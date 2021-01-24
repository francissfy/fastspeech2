import torch
import torch.nn as nn
import numpy as np
from network.utils import ConvNorm, get_mask_from_lengths
from network.utils import pad as utils_pad, get_device
from collections import OrderedDict


class LengthRegulator(nn.Module):
    def __init__(self, device=None):
        super(LengthRegulator, self).__init__()
        self.device = get_device() if device is None else device

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
        out = torch.cat(out, 0)
        return out

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])
        if max_len is not None:
            output = utils_pad(output, max_len)
        else:
            output = utils_pad(output)
        return output, torch.LongTensor(mel_len).to(self.device)

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    def __init__(self, cfg):
        super(VariancePredictor, self).__init__()

        self.input_size = cfg.MODEL.ENCODER.HIDDEN
        variance_param = cfg.MODEL.VARIANCE_PREDICTOR
        self.filter_size = variance_param.FILTER_SIZE
        self.kernel_size = variance_param.KERNEL_SIZE
        self.conv_output_size = variance_param.FILTER_SIZE
        self.dropout = variance_param.DROPOUT

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", ConvNorm(
                in_channels=self.input_size,
                out_channels=self.filter_size,
                kernel_size=self.kernel_size,
                padding=(self.kernel_size - 1) // 2
            )),
            ("relu_1", nn.ReLU()),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("conv1d_2", ConvNorm(
                in_channels=self.filter_size,
                out_channels=self.filter_size,
                kernel_size=self.kernel_size,
                padding=1
            )),
            ("relu_2", nn.ReLU()),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("dropout_2", nn.Dropout(self.dropout)),
        ]))
        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


class VarianceAdaptor(nn.Module):
    def __init__(self, cfg, device=None):
        super(VarianceAdaptor, self).__init__()

        self.device = get_device() if device is None else device

        self.duration_predictor = VariancePredictor(cfg)
        self.pitch_predictor = VariancePredictor(cfg)
        self.energy_predictor = VariancePredictor(cfg)
        self.length_regulator = LengthRegulator(device=self.device)

        quant_param = cfg.QUANTIZATION
        self.log_offset = quant_param.LOG_OFFSET
        encoder_param = cfg.MODEL.ENCODER

        self.pitch_bins = nn.Parameter(
            torch.exp(
                torch.linspace(np.log(quant_param.F0_MIN), np.log(quant_param.F0_MAX), quant_param.N_BINS - 1)
            ),
            requires_grad=False
        )
        self.energy_bins = nn.Parameter(
            torch.exp(
                torch.linspace(np.log(quant_param.ENERGY_MIN), np.log(quant_param.ENERGY_MAX), quant_param.N_BINS - 1)
            ),
            requires_grad=False
        )
        self.pitch_embedding = nn.Embedding(quant_param.N_BINS, encoder_param.HIDDEN)
        self.energy_embedding = nn.Embedding(quant_param.N_BINS, encoder_param.HIDDEN)

    def forward(self,
                x,
                src_mask,
                mel_mask=None,
                duration_target=None,
                pitch_target=None,
                energy_target=None,
                max_len=None,
                d_control=1.0,
                p_control=1.0,
                e_control=1.0):
        log_duration_prediction = self.duration_predictor(x, src_mask)
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
        else:
            duration_rounded = torch.clamp(
                torch.round(torch.exp(log_duration_prediction) - self.log_offset) * d_control,
                min=0
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len, device=self.device)

        pitch_prediction = self.pitch_predictor(x, mel_mask)
        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding(
                torch.bucketize(pitch_target, self.pitch_bins)
            )
        else:
            pitch_prediction = pitch_prediction * p_control
            pitch_embedding = self.pitch_embedding(
                torch.bucketize(pitch_prediction, self.pitch_bins)
            )
        energy_prediction = self.energy_predictor(x, mel_mask)
        if energy_target is not None:
            energy_embedding = self.energy_embedding(
                torch.bucketize(energy_target, self.energy_bins)
            )
        else:
            energy_prediction = energy_prediction * e_control
            energy_embedding = self.energy_embedding(
                torch.bucketize(energy_prediction, self.energy_bins)
            )
        x = x + pitch_embedding + energy_embedding

        return x, log_duration_prediction, pitch_prediction, energy_prediction, mel_len, mel_mask
