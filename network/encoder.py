import torch.nn as nn
from network.utils import get_sinusoid_encoding_table
from network.transformer import FFTBlock


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
        n_head = encoder_ns.N_HEAD
        d_k = encoder_ns.HIDDEN
        d_v = encoder_ns.HIDDEN
        d_model = encoder_ns.HIDDEN
        dropout = encoder_ns.DROPOUT

        fft_ns = cfg.MODEL.FFT
        d_inner = fft_ns.CONV1D_FILTER_SIZE

        n_position = self.len_max_seq+1

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
                get_sinusoid_encoding_table(src_seq.shape[1], self.encoder_hidden)[:src_seq.shape[1], :]\
                .unsqueeze(0).expand(batch_size, -1, -1).to(src_seq.device)
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_self_attn = enc_layer(enc_output, mask=mask, self_atten_mask=self_atten_mask)
            if return_attns:
                enc_self_attn_list.append(enc_self_attn)

        return enc_output
