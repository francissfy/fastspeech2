import torch
import torch.nn as nn
from network.utils import ConvNorm, LayerNorm, pad_list


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
        duration_outs = self.duration_predictor.forward(hs, duration_mask)
        hs = self.length_regulator.forward(hs, durations, input_lengths)
        pitch_outs = self.pitch_predictor.forward(hs, variance_mask)
        pitch_embed = self.pitch_embed.forward(pitch_target.transpose(1, 2)).transpose(1, 2)
        energy_outs = self.energy_predictor.forward(hs, variance_mask)
        energy_embed = self.energy_embed(energy_target.transpose(1, 2)).transpose(1, 2)
        hs += pitch_embed + energy_embed
        return hs, duration_outs, pitch_outs, energy_outs

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
        return hs, duration_outs, pitch_outs, energy_outs
