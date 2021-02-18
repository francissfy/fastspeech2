import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List


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
            padding = int((dilation * (kernel_size-1))/2)
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


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def cal_angle(position, hid_idx):
        return position/np.power(10000, 2*(hid_idx//2)/d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, ::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.0
    return torch.FloatTensor(sinusoid_table)


def get_mask_from_lengths(lengths, max_len=None):
    """
    lengths: tensor
    """
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    if isinstance(max_len, torch.Tensor):
        max_len = max_len.item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = (ids > lengths.unsqueeze(1).expand(-1, max_len))
    return mask


def pad_list(xs: List[torch.Tensor],
             pad_value: float):
    n_batch = len(xs)
    max_len = max(x.shape[0] for x in xs)

    pad = xs[0].new(n_batch, max_len, *(xs[0].shape[1:])).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].shape[0]] = xs[i]
    return pad


def pad(input_ele, mel_max_length=None):
    max_len = mel_max_length if mel_max_length \
        else max([input_ele[i].size()[0] for i in range(len(input_ele))])
    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(batch, (0, max_len-batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0)
        else:
            assert 0, "network: tools: pad: len(batch.shape) error!"
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

