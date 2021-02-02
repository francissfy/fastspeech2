import torch
import torch.nn as nn
import numpy as np
from network.utils import ConvNorm, get_mask_from_lengths, pad_list
from network.utils import get_device
from collections import OrderedDict
from tools.utils import const_pad_tensors_dim1


# use LR from espnet
class LengthRegulator(nn.Module):
    def __init__(self,
                 pad_value: float = 0.0):
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self,
                xs: torch.Tensor,
                durations: torch.LongTensor,
                input_lengths: torch.LongTensor,
                duration_control: float = 1.0):
        if durations != 1.0:
            duration_control = torch.round(durations.float() * duration_control).long()
        xs = [x[:in_len] for x, in_len in zip(xs, input_lengths)]
        ds = [d[:in_len] for d, in_len in zip(durations, input_lengths)]
        xs = [self._repeat_one_sequence(x, d) for x, d in zip(xs, ds)]
        return pad_list(xs, self.pad_value)

    def _repeat_one_sequence(self,
                             x: torch.Tensor,
                             d: torch.LongTensor):
        if d.sum() == 0:
            # FIXME
            d = d.fill_(1)
        return torch.cat([x_.repeat(int(d_), 1) for x_, d_ in zip(x, d) if d_ != 0], dim=0)


# espnet implementation, don't use torch.Embedding
class VarianceEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, embed_kernel_size):
        super(VarianceEmbedding, self).__init__()
        self.conv = ConvNorm(in_channels=in_dim,
                             out_channels=out_dim,
                             kernel_size=embed_kernel_size,
                             padding=(embed_kernel_size-1)//2)
        self.dropout = nn.Dropout(embed_kernel_size)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.dropout(x)
        return x


# use espnet implementation
class VariancePredictor(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 n_layers: int = 2,
                 n_chans: int = 384,
                 kernel_size: int = 3,
                 bias: bool = True,
                 dropout_rate: float = 0.5):
        super(VariancePredictor, self).__init__()

        self.conv = nn.ModuleList([
            nn.Sequential(
                ConvNorm(in_channels=in_dim,
                         out_channels=n_chans,
                         kernel_size=kernel_size,
                         padding=(kernel_size-1)//2,
                         bias=bias),
                nn.ReLU(),
                # FIXME
                nn.LayerNorm(n_chans),
                nn.Dropout(dropout_rate)
            ) for in_dim in ([in_dim]+[n_chans] * (n_layers-1))
        ])
        self.linear = nn.Linear(n_chans, out_dim)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor = None):
        x = x.transpose(1, -1)
        for f in self.conv:
            x = f(x)
        x = self.linear(x.transpose(1, -1))

        if mask is not None:
            x = x.masked_fill(mask, 0.0)
        return x


# espnet variance adapter
class VA(nn.Module):
    def __init__(self):
        super(VA, self).__init__()

    def forward(self):
        pass

    def inference(self):
        pass


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
