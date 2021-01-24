import torch.nn as nn
from network.encoder import Encoder
from network.decoder import Decoder
from network.transformer import PostNet
from network.variance_adaptor import VarianceAdaptor
from network.utils import get_device, get_mask_from_lengths


class Fastspeech2(nn.Module):
    def __init__(self, cfg):
        super(Fastspeech2, self).__init__()
        # NOTE use naive fastspeech2

        self.use_postnet = False
        self.device = get_device()

        self.encoder = Encoder(cfg)
        self.variance_adaptor = VarianceAdaptor(cfg, device=self.device)

        self.decoder = Decoder(cfg)

        self.mel_linear = nn.Linear(cfg.MODEL.DECODER.HIDDEN, cfg.AUDIO.N_MEL_CHANNELS)

        self.postnet = PostNet(cfg) if self.use_postnet else None

    def forward(self,
                src_seq,
                src_len,
                mel_len=None,
                d_target=None,
                p_target=None,
                e_target=None,
                max_src_len=None,
                max_mel_len=None,
                d_control=1.0,
                p_control=1.0,
                e_control=1.0):
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None

        encoder_output = self.encoder(src_seq, src_mask)

        variance_adaptor_output, d_prediction, p_prediction, e_prediction, mel_len_pred, mel_mask_pred = \
            self.variance_adaptor(
                encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len,
                d_control, p_control, e_control
            )
        if d_target is None:
            mel_len, mel_mask = mel_len_pred, mel_mask_pred

        decoder_output = self.decoder(variance_adaptor_output, mel_mask)
        mel_output = self.mel_linear(decoder_output)

        if self.use_postnet:
            mel_output = self.postnet(mel_output) + mel_output
        # NOTE: changed the return sequence
        return mel_output, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len
