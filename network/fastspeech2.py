import torch
import torch.nn as nn
from network.encoder import Encoder
from network.encoder import PositionalEncoding, ScaledPositionalEncoding
from network.variance_adaptor import VarianceAdaptor, VariancePredictor
from network.utils import LayerNorm
from network.loss import DurationPredictorLoss
from typing import Union, Optional


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


class PostNet(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 n_layers: int = 5,
                 n_chans: int = 512,
                 n_filts: int = 5,
                 dropout_rate: float = 0.5,
                 use_batch_norm: bool = True):
        super(PostNet, self).__init__()
        self.postnet = nn.ModuleList()
        for layer in range(n_layers-1):
            ichans = in_dim if layer == 0 else n_chans
            ochans = out_dim if layer == n_layers-1 else n_chans
            if use_batch_norm:
                self.postnet.append(nn.Sequential(
                    nn.Conv1d(in_channels=ichans,
                              out_channels=ochans,
                              kernel_size=n_filts,
                              stride=1,
                              padding=(n_filts-1)//2,
                              bias=False),
                    nn.BatchNorm1d(ochans),
                    nn.Tanh(),
                    nn.Dropout(dropout_rate)
                ))
            else:
                self.postnet.append(nn.Sequential(
                    nn.Conv1d(in_channels=ichans,
                              out_channels=ochans,
                              kernel_size=n_filts,
                              stride=1,
                              padding=(n_filts - 1) // 2,
                              bias=False),
                    nn.Tanh(),
                    nn.Dropout(dropout_rate)
                ))

    def forward(self, xs: torch.Tensor):
        for layer in self.postnet:
            xs = layer(xs)
        return xs


class FeedForwardTransformer(nn.Module):
    def __init__(self,
                 idim: int,
                 odim: int,
                 adim: int = 384,
                 aheads: int = 4,
                 positionwise_layer_type: str = "linear",
                 positionwise_conv_kernel_size: int = 3,
                 # encoder
                 eunits: int = 1536,
                 elayers: int = 6,
                 transformer_enc_dropout_rate: float = 0.0,
                 transformer_enc_positional_dropout_rate: float = 0.0,
                 transformer_enc_atten_dropout_rate: float = 0.0,
                 encoder_normalized_before: bool = True,
                 encoder_concat_after: bool = False,

                 # variance
                 pitch_embed_kernel_size: int = 1,
                 pitch_embed_dropout: float = 0.0,
                 energy_embed_kernel_size: int = 1,
                 energy_embed_dropout: float = 0.0,
                 duration_predictor_layers: int = 2,
                 duration_predictor_chans: int = 256,
                 duration_predictor_kernel_size: int = 3,
                 duration_predictor_dropout_rate: float = 0.1,
                 # decoder
                 dlayers: int = 6,
                 dunits: int = 1536,
                 transformer_dec_dropout_rate: float = 0.1,
                 transformer_dec_positional_dropout_rate: float = 0.1,
                 transformer_dec_atten_dropout_rate: float = 0.1,
                 decoder_normalized_before: bool = False,
                 decoder_concat_after: bool = False,

                 reduction_factor: int = 1,
                 # postnet
                 postnet_layers: int = 5,
                 postnet_filts: int = 5,
                 postnet_chans: int = 256,
                 postnet_dropout_rate: float = 0.5,

                 use_batch_norm: bool = True,
                 use_scaled_pos_enc: bool = True,

                 # init
                 transformer_init: str = "pytorch",
                 initial_encoder_alpha: float = 1.0,
                 initial_decoder_alpha: float = 1.0,


                 ):
        super(FeedForwardTransformer, self).__init__()
        self.use_scaled_pos_enc = use_scaled_pos_enc

        # encoder
        pos_enc_class = ScaledPositionalEncoding if use_scaled_pos_enc else PositionalEncoding

        padding_idx: int = 0
        encoder_input_layer = nn.Embedding(
            num_embeddings=idim,
            embedding_dim=adim,
            padding_idx=padding_idx)
        self.encoder = Encoder(input_layer=encoder_input_layer,
                               attention_dim=adim,
                               attention_heads=aheads,
                               linear_units=eunits,
                               num_blocks=elayers,
                               dropout_rate=transformer_enc_dropout_rate,
                               positional_dropout_rate=transformer_enc_positional_dropout_rate,
                               attention_dropout_rate=transformer_enc_atten_dropout_rate,
                               pos_enc_class=pos_enc_class,
                               normalized_before=encoder_normalized_before,
                               concate_after=encoder_concat_after,
                               positionwise_layer_type=positionwise_layer_type,
                               positionwise_conv_kernel_size=positionwise_conv_kernel_size)
        self.variance_adaptor = VarianceAdaptor(adim=adim,
                                                pitch_dim=4,
                                                energy_dim=1,
                                                pitch_embed_kernel_size=pitch_embed_kernel_size,
                                                pitch_embed_dropout_rate=pitch_embed_dropout,
                                                energy_embed_kernel_size=energy_embed_kernel_size,
                                                energy_embed_dropout_rate=energy_embed_dropout,
                                                duration_predictor_layers=duration_predictor_layers,
                                                duration_predictor_chans=duration_predictor_chans,
                                                duration_predictor_kernel_size=duration_predictor_kernel_size,
                                                duration_predictor_dropout_rate=duration_predictor_dropout_rate)
        self.decoder = Encoder(input_layer=None,
                               attention_dim=adim,
                               attention_heads=aheads,
                               linear_units=dunits,
                               num_blocks=dlayers,
                               dropout_rate=transformer_dec_dropout_rate,
                               positional_dropout_rate=transformer_dec_positional_dropout_rate,
                               attention_dropout_rate=transformer_dec_atten_dropout_rate,
                               pos_enc_class=pos_enc_class,
                               normalized_before=decoder_normalized_before,
                               concate_after=decoder_concat_after,
                               positionwise_layer_type=positionwise_layer_type,
                               positionwise_conv_kernel_size=positionwise_conv_kernel_size)
        self.feat_out = nn.Linear(in_features=adim,
                                  out_features=odim*reduction_factor)

        self.postnet = None if postnet_layers == 0 else PostNet(in_dim=idim,
                                                                out_dim=odim,
                                                                n_layers=postnet_layers,
                                                                n_chans=postnet_chans,
                                                                n_filts=postnet_filts,
                                                                use_batch_norm=use_batch_norm,
                                                                dropout_rate=postnet_dropout_rate)
        self._reset_parameters(init_type=transformer_init,
                               init_enc_alpha=initial_encoder_alpha,
                               init_dec_alpha=initial_decoder_alpha)
        self.duration_criterion = DurationPredictorLoss()
        self.mse_criterion = nn.MSELoss()

    def _forward(self,
                 xs: torch.FloatTensor,
                 ilens: torch.LongTensor,
                 olens: torch.LongTensor = None,
                 ds: torch.LongTensor = None,
                 ps: torch.FloatTensor = None,
                 es: torch.FloatTensor = None,
                 in_masks: torch.LongTensor = None,
                 out_masks: torch.LongTensor = None,
                 is_inference: bool = False):
        x_masks =


    def _source_mask(self, ilens: torch.FloatTensor):
        x_masks =

    def _reset_parameters(self,
                          init_type,
                          init_enc_alpha: float = 1.0,
                          init_dec_alpha: float = 1.0):
        initialize(self, init_type)
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)



"""deprecated
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
"""
