import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network.utils import ConvNorm


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# transformer
# # # # # # # # # # # # # # # # # # # # # # # # # # # #


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.sm = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        atten = torch.bmm(q, k.transpose(1, 2))
        atten = atten / self.temperature

        if mask is not None:
            atten = atten.masked_fill_(mask, -np.inf)

        atten = self.sm(atten)
        output = torch.bmm(atten, v)
        return output, atten


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        # TODO n_head?
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        mask = mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, cfg, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(
            in_channels=d_in,
            out_channels=d_hid,
            kernel_size=cfg.MODEL.FFT.CONV1D_KERNEL_SIZE[0],
            padding=((cfg.MODEL.FFT.CONV1D_KERNEL_SIZE[0]-1)//2)
        )
        self.w_2 = nn.Conv1d(
            in_channels=d_hid,
            out_channels=d_in,
            kernel_size=cfg.MODEL.FFT.CONV1D_KERNEL_SIZE[1],
            padding=((cfg.MODEL.FFT.CONV1D_KERNEL_SIZE[1]-1)//2)
        )
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class FFTBlock(nn.Module):
    def __init__(self, cfg, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.post_ffn = PositionwiseFeedForward(cfg, d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, mask=None, self_attn_mask=None):
        enc_output, enc_self_attn = self.self_attn(enc_input, enc_input, enc_input, mask=self_attn_mask)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.post_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_self_attn


class PostNet(nn.Module):
    def __init__(self, cfg):
        super(PostNet, self).__init__()

        n_mel_channels = cfg.AUDIO.N_MEL_CHANNELS

        post_param = cfg.MODEL.POSTNET
        postnet_embedding_dim = post_param.EMBEDDING_DIM
        postnet_kernel_size = post_param.KERNEL_SIZE
        postnet_n_convolutions = post_param.N_CONVOLUTIONS

        self.convolutions = nn.ModuleList()

        in_channels_list = [n_mel_channels] + [postnet_embedding_dim]*(postnet_n_convolutions-2) + [n_mel_channels]
        out_channels_list = [postnet_embedding_dim]*(postnet_n_convolutions-1) + [n_mel_channels]
        init_gain_list = ["tanh"]*(postnet_n_convolutions-1) + ["linear"]

        for in_dim, out_dim, gain in zip(in_channels_list, out_channels_list, init_gain_list):
            self.convolutions.append(nn.Sequential(
                ConvNorm(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                ),
                nn.BatchNorm1d(out_dim)
            ))

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)

        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        x = x.contiguous().transpose(1, 2)
        return x
