import torch
import torch.nn as nn
import numpy as np
from network.utils import ConvNorm, get_mask_from_lengths
from network.utils import get_device
from collections import OrderedDict
from utils.utils import const_pad_tensors_dim1


class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def forward(self, x, duration, max_mel_len=None):
        expanded_tensors = []
        for d, t in zip(duration, x):   # t: [L, W]
            target_len = int(d.item())
            t = t.expand((target_len, -1))
            expanded_tensors.append(t)
        padded_tensor = const_pad_tensors_dim1(expanded_tensors, max_mel_len)
        return padded_tensor


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
    def __init__(self, cfg):
        super(VarianceAdaptor, self).__init__()

        self.duration_predictor = VariancePredictor(cfg)
        self.pitch_predictor = VariancePredictor(cfg)
        self.energy_predictor = VariancePredictor(cfg)
        self.length_regulator = LengthRegulator()

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
                mel_mask,
                duration_target,
                pitch_target,
                energy_target,
                max_mel_len
                ):
        # use for loss
        log_duration_prediction = self.duration_predictor(x, src_mask)
        # use target duration
        x, mel_len = self.length_regulator(x, duration_target, max_mel_len)

        pitch_prediction = self.pitch_predictor(x, mel_mask)
        pitch_embedding = self.pitch_embedding(
            torch.bucketize(pitch_target, self.pitch_bins)
        )

        energy_prediction = self.energy_predictor(x, mel_mask)
        energy_embedding = self.energy_embedding(
            torch.bucketize(energy_target, self.energy_bins)
        )

        x = x + pitch_embedding + energy_embedding

        return x, log_duration_prediction, pitch_prediction, energy_prediction, mel_len, mel_mask

    def inference(self,
                  x,
                  src_mask,
                  duration_control=1.0,
                  pitch_control=1.0,
                  energy_control=1.0):
        # duration
        log_duration_prediction = self.duration_predictor(x, src_mask)
        duration_rounded = torch.clamp(
            torch.round(torch.exp(log_duration_prediction) - self.log_offset) * duration_control,
            min=0
        )
        # use duration as mel_lengths
        mel_lengths = duration_rounded
        x = self.length_regulator(x, duration_rounded)
        mel_mask = get_mask_from_lengths(mel_lengths)
        # pitch
        pitch_prediction = self.pitch_predictor(x, mel_mask) * pitch_control
        pitch_embedding = self.pitch_embedding(
            torch.bucketize(pitch_prediction, self.pitch_bins)
        )
        energy_prediction = self.energy_predictor(x, mel_mask) * energy_control
        energy_embedding = self.energy_embedding(
            torch.bucketize(energy_prediction, self.energy_bins)
        )
        x = x + pitch_embedding + energy_embedding
        return x, log_duration_prediction, pitch_prediction, energy_prediction, mel_lengths, mel_mask
