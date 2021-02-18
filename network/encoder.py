import numpy as np
import math
import torch
import torch.nn as nn
from network.utils import get_sinusoid_encoding_table, LayerNorm
from network.transformer import FFTBlock
from typing import Union


class MultiLayeredConv1d(nn.Module):
    def __init__(self,
                 in_channs: int,
                 hidden_chans: int,
                 kernel_size: int,
                 dropout_rate: float):
        super(MultiLayeredConv1d, self).__init__()
        self.w1 = nn.Conv1d(in_channels=in_channs,
                            out_channels=hidden_chans,
                            kernel_size=kernel_size,
                            padding=(kernel_size - 1) // 2,
                            stride=1)
        self.w2 = nn.Conv1d(in_channels=hidden_chans,
                            out_channels=in_channs,
                            kernel_size=kernel_size,
                            padding=(kernel_size - 1) // 2,
                            stride=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        # TODO transpose
        x = torch.relu(self.w1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)


class PositionwiseFeedForward(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_units: int,
                 dropout_rate: float):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(in_dim, hidden_units)
        self.w2 = nn.Linear(hidden_units, in_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        return self.w2(self.dropout(torch.relu(self.w1(x))))


class MultiHeadedAttention(nn.Module):
    """
    TODO n_head?
    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0

        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.atten = None
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                query: torch.FloatTensor,
                key: torch.FloatTensor,
                value: torch.FloatTensor,
                mask: torch.Tensor):
        n_batch = query.shape[0]
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1) / math.sqrt(self.d_k))
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.atten = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            self.atten = torch.softmax(scores, dim=-1)

        p_atten = self.dropout(self.atten)
        x = torch.matmul(p_atten, v)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)


def _pre_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 # FIXME
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x: torch.Tensor):
        if self.pe is not None and \
                self.pe.shape[1] >= x.shape[1] and \
                (self.pe.dtype != x.dtype or self.pe.device != x.device):
            return
        pe = torch.zeros(x.shape[1], self.d_model)
        position = torch.arange(0, x.shape[1], dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) *
                             -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.shape[1]]
        return self.dropout(x)


class ScaledPositionalEncoding(PositionalEncoding):
    def __int__(self,
                d_model: int,
                dropout_rate: float,
                max_len: int = 5000):
        super(ScaledPositionalEncoding, self).__int__(d_model, dropout_rate, max_len)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        x = x + self.alpha*self.pe[:, :x.shape[1]]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """
    TODO 这是个嘛
    """

    def __init__(self,
                 in_dim: int,
                 self_atten: nn.Module,
                 feed_forward: nn.Module,
                 dropout_rate: float,
                 normalized_before: bool = True,
                 concat_after: bool = False):
        super(EncoderLayer, self).__init__()
        self.self_atten = self_atten
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_dim)
        self.norm2 = LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_dim = in_dim
        self.normalized_before = normalized_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(in_dim + in_dim, in_dim)

    def forward(self,
                x: torch.FloatTensor,
                # TODO mask type
                mask: torch.LongTensor):
        residual = x
        if self.normalized_before:
            x = self.norm1(x)
        if self.concat_after:
            x_concat = torch.cat((x, self.self_atten(x, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_atten(x, x, x, mask))
        if not self.normalized_before:
            x = self.norm1(x)

        residual = x
        if self.normalized_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalized_before:
            x = self.norm2(x)

        return x, mask


class Encoder(nn.Module):
    def __init__(self,
                 input_layer: Union[str, nn.Module] = None,
                 pos_enc_class=PositionalEncoding,
                 attention_dim: int = 256,
                 attention_heads: int = 4,
                 attention_dropout_rate: float = 0.0,
                 linear_units: int = 2048,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 normalized_before: bool = True,
                 concate_after: bool = False,
                 positionwise_layer_type: str = "linear",
                 positionwise_conv_kernel_size: int = 1,
                 num_blocks: int = 6,

                 ):
        super(Encoder, self).__init__()
        # only implemented two cases used in fastspeech in espnet
        if isinstance(input_layer, nn.Module):
            self.embed = nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        elif input_layer is None:
            self.embed = nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            # conv2d, embd not implemented
            raise NotImplementedError("not implemented")

        self.normalized_before = normalized_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate)
        else:
            raise NotImplementedError("not implemented")

        self.encoders = [
            EncoderLayer(attention_dim,
                         MultiHeadedAttention(n_head=attention_heads,
                                              n_feat=attention_dim,
                                              dropout_rate=attention_dropout_rate),
                         positionwise_layer(*positionwise_layer_args),
                         dropout_rate,
                         normalized_before,
                         concate_after)
            for _ in range(num_blocks)
        ]
        if self.normalized_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks):
        xs = self.embed(xs)
        for encoder in self.encoders:
            xs = encoder(xs, masks)
        if self.normalized_before:
            xs = self.after_norm(xs)
        return xs, masks


"""The following method is deprecated
class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()

        model_ns = cfg.MODEL
        n_src_vocab = model_ns.N_SRC_VOCAB
        self.len_max_seq = model_ns.MAX_SEQ_LEN

        encoder_ns = cfg.MODEL.ENCODER
        self.encoder_hidden = encoder_ns.HIDDEN
        d_word_vec = encoder_ns.HIDDEN
        n_layers = encoder_ns.LAYER
        n_head = encoder_ns.HEAD
        d_k = encoder_ns.HIDDEN
        d_v = encoder_ns.HIDDEN
        d_model = encoder_ns.HIDDEN
        dropout = encoder_ns.DROPOUT

        fft_ns = cfg.MODEL.FFT
        d_inner = fft_ns.CONV1D_FILTER_SIZE

        n_position = self.len_max_seq + 1

        const_ns = cfg.CONST

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=const_ns.PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0), requires_grad=False
        )

        self.layer_stack = nn.ModuleList(
            [FFTBlock(cfg, d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, src_seq, mask, return_attns=False):
        enc_self_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        self_atten_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        if not self.training and src_seq.shape[1] > self.len_max_seq:
            enc_output = self.src_word_emb(src_seq) + \
                         get_sinusoid_encoding_table(src_seq.shape[1], self.encoder_hidden)[:src_seq.shape[1], :] \
                             .unsqueeze(0).expand(batch_size, -1, -1).to(src_seq.device)
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_self_attn = enc_layer(enc_output, mask=mask, self_attn_mask=self_atten_mask)
            if return_attns:
                enc_self_attn_list.append(enc_self_attn)

        return enc_output
"""
