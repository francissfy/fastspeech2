import chainer
import torch
import torch.nn as nn
from typing import List, Union


class ConvNorm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 bias=True,
                 w_init_gain="linear"):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int((dilation * (kernel_size - 1)) / 2)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        nn.init.xavier_normal_(self.conv.weight,
                               gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        output = self.conv(x)
        return output


class LayerNorm(nn.LayerNorm):
    def __init__(self, nout, dim=-1):
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.shape[0] for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].shape[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].shape[0]] = xs[i]
    return pad


def make_pad_mask(lengths: Union[torch.LongTensor, List[int]], xs: torch.FloatTensor, length_dim: int):
    if length_dim == 0:
        raise ValueError(f"length_dim cannot be {length_dim}")
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.shape[length_dim]

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.shape[0] == bs, (xs.shape[0], bs)
        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        ind = tuple(slice(None) if i in (0, length_dim) else None
                    for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths: Union[torch.LongTensor, List[int]], xs: torch.FloatTensor = None, length_dim: int = -1):
    return ~make_pad_mask(lengths, xs, length_dim)


def initialize(model: nn.Module, init_type: str = "pytorch"):
    if init_type == "pytorch":
        return
    for p in model.parameters():
        if p.dim() > 1:
            if init_type == "xavier_uniform":
                nn.init.xavier_uniform_(p.data)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(p.data)
            elif init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(p.data, nonlinearity="relu")
            else:
                raise ValueError(f"unknown initialization: {init_type}")
    for p in model.parameters():
        if p.dim == 1:
            p.data.zero_()
    for m in model.modules():
        if isinstance(m, (nn.Embedding, LayerNorm)):
            m.reset_parameters()


class Reporter(chainer.Chain):
    def report(self, dicts):
        for d in dicts:
            chainer.reporter.report(d, self)