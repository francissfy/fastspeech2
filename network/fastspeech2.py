import torch.nn as nn
from network.encoder import Encoder
from network.decoder import Decoder
from network.transformer import PostNet
from network.variance_adaptor import VarianceAdaptor
from network.utils import get_device, get_mask_from_lengths


class Fastspeech2(nn.Module):
    def __init__(self, cfg):
        super(Fastspeech2, self).__init__()

        self.use_postnet = True
        self.device = get_device()

        self.encoder = Encoder(cfg)
        self.variance_adaptor = VarianceAdaptor(cfg)

        self.decoder = Decoder(cfg)

        self.mel_linear = nn.Linear(cfg.MODEL.DECODER.HIDDEN, cfg.AUDIO.N_MEL_CHANNELS)

        self.postnet = PostNet(cfg) if self.use_postnet else None

    def forward(self,
                src_seq,
                src_len,
                mel_len,
                d_target,
                p_target,
                e_target,
                max_src_len,
                max_mel_len):
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len)

        encoder_output = self.encoder(src_seq, src_mask)

        variance_adaptor_output, d_prediction, p_prediction, e_prediction, _, _ = \
            self.variance_adaptor(encoder_output,
                                  src_mask,
                                  mel_mask,
                                  d_target,
                                  p_target,
                                  e_target,
                                  max_mel_len)

        decoder_output = self.decoder(variance_adaptor_output, mel_mask)
        mel_output = self.mel_linear(decoder_output)

        if self.use_postnet:
            mel_output_postnet = self.postnet(mel_output) + mel_output
        else:
            mel_output_postnet = mel_output

        return mel_output, mel_output_postnet, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len

    def inference(self,
                  src_seq,
                  src_len,
                  duration_control=1.0,
                  pitch_control=1.0,
                  energy_control=1.0
                  ):
        src_mask = get_mask_from_lengths(src_len)

        encoder_output = self.encoder(src_seq, src_mask)

        variance_adaptor_output, duration_prediction, pitch_prediction, energy_prediction, mel_len, mel_mask = \
            self.variance_adaptor.inference(encoder_output,
                                            src_mask,
                                            duration_control,
                                            pitch_control,
                                            energy_control)

        decoder_output = self.decoder(variance_adaptor_output, mel_mask)
        mel_output = self.mel_linear(decoder_output)

        if self.use_postnet:
            mel_output_postnet = self.postnet(mel_output) + mel_output
        else:
            mel_output_postnet = mel_output

        return mel_output, mel_output_postnet, duration_prediction, pitch_prediction, \
            energy_prediction, src_mask, mel_mask, mel_len
