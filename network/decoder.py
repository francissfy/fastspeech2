import torch.nn as nn
from network.utils import get_sinusoid_encoding_table
from network.transformer import FFTBlock


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        model_ns = cfg.MODEL
        self.len_max_seq = model_ns.MAX_SEQ_LEN

        decoder_ns = cfg.MODEL.DECODER
        self.decoder_hidden = decoder_ns.HIDDEN
        d_word_vec = decoder_ns.HIDDEN
        n_layers = decoder_ns.LAYER
        n_head = decoder_ns.N_HEAD
        d_k = decoder_ns.HIDDEN
        d_v = decoder_ns.HIDDEN
        d_model = decoder_ns.HIDDEN
        dropout = decoder_ns.DROPOUT

        fft_ns = cfg.MODEL.FFT
        d_inner = fft_ns.CONV1D_FILTER_SIZE

        n_position = self.len_max_seq + 1

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0), requires_grad=False
        )

        self.layer_stack = nn.ModuleList(
            [FFTBlock(cfg, d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, enc_seq, mask, return_attns=False):
        dec_self_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]
        self_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        if not self.training and enc_seq.shape[1] > self.len_max_seq:
            dec_output = enc_seq + \
                get_sinusoid_encoding_table(enc_seq.shape[1], self.decoder_hidden)[:enc_seq.shape[1], :]\
                .unsqueeze(0).expand(batch_size, -1, -1).to(enc_seq.decode)
        else:
            dec_output = enc_seq + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)

        for dec_layer in self.layer_stack:
            dec_output, dec_self_attn = dec_layer(dec_output, mask=mask, self_attn_mask=self_attn_mask)
            if return_attns:
                dec_self_attn_list.append(dec_self_attn)

        return dec_output
