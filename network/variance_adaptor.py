import torch
import torch.nn as nn
import numpy as np
from network.utils import ConvNorm, LayerNorm, get_mask_from_lengths, pad_list
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
        if duration_control != 1.0:
            durations = torch.round(durations.float() * duration_control).long()
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
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 embed_kernel_size: int,
                 embed_droput_rate: float):
        super(VarianceEmbedding, self).__init__()
        self.conv = ConvNorm(in_channels=in_dim,
                             out_channels=out_dim,
                             kernel_size=embed_kernel_size,
                             padding=(embed_kernel_size-1)//2)
        self.dropout = nn.Dropout(embed_droput_rate)

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
                LayerNorm(n_chans, dim=1),
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


class DurationPredictor(nn.Module):
    def __init__(self,
                 in_dim: int,
                 n_layers: int = 2,
                 n_chans: int = 384,
                 kernel_size: int = 3,
                 dropout_rate: float = 0.1,
                 offset: float = 1.0):
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = nn.ModuleList([
            nn.Sequential(
                ConvNorm(in_channels=in_chans,
                         out_channels=n_chans,
                         kernel_size=kernel_size,
                         padding=(kernel_size-1)//2),
                nn.ReLU(),
                LayerNorm(n_chans, dim=-1),
                nn.Dropout(dropout_rate)
            )
            for in_chans in ([in_dim]+[n_chans]*(n_layers-1))
        ])
        self.linear = nn.Linear(n_chans, 1)

    def forward(self,
                xs: torch.Tensor,
                x_masks: torch.Tensor = None):
        xs = xs.transpose(1, 2)
        for f in self.conv:
            xs = f(xs)
        xs = self.linear(xs)
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)
        return xs

    def inference(self,
                xs: torch.Tensor,
                x_masks: torch.Tensor = None):
        xs = xs.transpose(1, 2)
        for f in self.conv:
            xs = f(xs)
        xs = self.linear(xs)
        xs = torch.clamp(torch.round(xs.exp()-self.offset), min=0).long()
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)
        return xs


# espnet variance adapter
class VarianceAdaptor(nn.Module):
    def __init__(self,
                 adim: int,
                 pitch_dim: int = 4,
                 energy_dim: int = 1,
                 pitch_embed_kernel_size: int = 1,
                 pitch_embed_dropout_rate: float = 0.0,
                 energy_embed_kernel_size: int = 1,
                 energy_embed_dropout_rate: float = 0.0,
                 duration_predictor_layers: int = 2,
                 duration_predictor_chans: int = 256,
                 duration_predictor_kernel_size: int = 3,
                 duration_predictor_dropout_rate: float = 0.1):

        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = DurationPredictor(in_dim=adim,
                                                    n_layers=duration_predictor_layers,
                                                    n_chans=duration_predictor_chans,
                                                    kernel_size=duration_predictor_kernel_size,
                                                    dropout_rate=duration_predictor_dropout_rate)
        self.length_regulator = LengthRegulator()

        self.pitch_predictor = VariancePredictor(in_dim=adim,
                                                 out_dim=pitch_dim)
        self.pitch_embed = VarianceEmbedding(in_dim=pitch_dim,
                                             out_dim=adim,
                                             embed_kernel_size=pitch_embed_kernel_size,
                                             embed_droput_rate=pitch_embed_dropout_rate)

        self.energy_predictor = VariancePredictor(in_dim=adim,
                                                  out_dim=energy_dim)
        self.energy_embed = VarianceEmbedding(in_dim=energy_dim,
                                              out_dim=adim,
                                              embed_kernel_size=energy_embed_kernel_size,
                                              embed_droput_rate=energy_embed_dropout_rate)

    def forward(self,
                hs: torch.Tensor,
                durations: torch.LongTensor,
                input_lengths: torch.LongTensor,
                pitch_target: torch.Tensor,
                energy_target: torch.Tensor,

                duration_mask: torch.Tensor,
                variance_mask: torch.Tensor):
        hs = self.length_regulator.forward(hs, durations, input_lengths)
        # TODO
        pitch_embed = self.pitch_embed.forward(pitch_target.transpose(1, 2)).transpose(1, 2)
        energy_embed = self.energy_embed(energy_target.transpose(1, 2)).transpose(1, 2)
        hs += pitch_embed + energy_embed
        return hs

    def inference(self,
                  hs: torch.Tensor,
                  input_lengths: torch.LongTensor,
                  duration_masks: torch.Tensor,
                  variance_masks: torch.Tensor):
        duration_outs = self.duration_predictor.forward(hs, duration_masks)
        hs = self.length_regulator.forward(hs, duration_outs, input_lengths)
        pitch_outs = self.pitch_predictor.forward(hs, variance_masks)
        pitch_embed = self.pitch_embed.forward(pitch_outs.transpose(1, 2)).transpose(1, 2)
        energy_outs = self.energy_predictor.forward(hs, variance_masks)
        energy_embed = self.energy_embed.forward(energy_outs.transpose(1, 2)).transpose(1, 2)
        hs += energy_embed + pitch_embed
        return hs, pitch_outs, energy_outs


"""deprecated
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
    """
