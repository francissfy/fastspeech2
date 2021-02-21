import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.encoder import Encoder, MultiHeadedAttention
from network.encoder import PositionalEncoding, ScaledPositionalEncoding
from network.variance_adaptor import VarianceAdaptor
from network.utils import make_non_pad_mask, initialize, Reporter
from network.loss import DurationPredictorLoss
from tools.plot import TTSPlot


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
                 # init
                 transformer_init: str = "pytorch",
                 initial_encoder_alpha: float = 1.0,
                 initial_decoder_alpha: float = 1.0,
                 # other
                 use_masking: bool = True,
                 use_batch_norm: bool = True,
                 use_scaled_pos_enc: bool = True,
                 ):
        super(FeedForwardTransformer, self).__init__()
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.reduction_factor = reduction_factor
        self.odim = odim
        self.use_masking = use_masking

        self.reporter = Reporter()

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

    def _source_mask(self, ilens: torch.LongTensor):
        x_masks = make_non_pad_mask(ilens).to(self.feat_out.weight.device)
        return x_masks.unsqueeze(-2) & x_masks.unsqueeze(-1)

    def _reset_parameters(self,
                          init_type,
                          init_enc_alpha: float = 1.0,
                          init_dec_alpha: float = 1.0):
        initialize(self, init_type)
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)

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
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder.forward(xs, x_masks)

        # ignore spk embedding

        d_masks = ~in_masks if in_masks is not None else None
        v_masks = ~out_masks if out_masks is not None else None
        if is_inference:
            hs, d_outs, p_outs, e_outs = self.variance_adaptor.inference(hs, ilens, d_masks, v_masks)
        else:
            hs, d_outs, p_outs, e_outs = self.variance_adaptor.forward(hs, ds, ilens, ps, es, d_masks, v_masks)

        # forward decoder
        if olens is not None:
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None
        zs, _ = self.decoder.forward(hs, h_masks)
        before_outs = self.feat_out.forward(zs).view(zs.shape[0], -1, self.odim)

        # postnet
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)

        if is_inference:
            return before_outs, after_outs
        else:
            return before_outs, after_outs, d_outs, p_outs, e_outs

    def forward(self,
                xs: torch.FloatTensor,
                ilens: torch.LongTensor,
                ys: torch.FloatTensor,
                olens: torch.LongTensor,
                ds: torch.FloatTensor,
                ps: torch.FloatTensor,
                es: torch.FloatTensor):
        # rm padded part
        xs = xs[:, :max(ilens)]
        ys = ys[:, :max(olens)]
        ds = ds[:, :max(ilens)]
        ps = ps[:, :max(olens)]
        es = es[:, :max(olens)]

        in_masks = make_non_pad_mask(ilens).to(xs.device)
        out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
        # ignore spk embedding

        before_outs, after_outs, d_outs, p_outs, e_outs = \
            self._forward(xs, ilens, olens, ds, ps, es, in_masks=in_masks, out_masks=out_masks, is_inference=False)

        if self.reduction_factor > 1:
            olens = olens.new([olen-olen%self.reduction_factor for olen in olens])
            max_olen = max(olens)
            ys = ys[:, :max_olen]

        if self.use_masking:
            d_outs = d_outs.masked_select(in_masks)
            ds = ds.masked_select(in_masks)
            before_outs = before_outs.masked_select(out_masks)
            after_outs = after_outs.masked_select(out_masks)
            ys = ys.masked_select(out_masks)
            p_outs = p_outs.masked_select(out_masks)
            e_outs = e_outs.masked_select(out_masks)
            ps = ps.masked_select(out_masks)
            es = es.masked_select(out_masks)

        # calculate loss
        if self.postnet is None:
            l1_loss = F.l1_loss(after_outs, ys)
        else:
            l1_loss = F.l1_loss(after_outs, ys) + F.l1_loss(before_outs, ys)
        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)

        loss = l1_loss + duration_loss + pitch_loss + energy_loss
        # report loss
        report_keys = [
            {"l1_loss": l1_loss.item()},
            {"duration_loss": duration_loss.item()},
            {"pitch_loss": pitch_loss.item()},
            {"energy_loss": energy_loss.item()},
            {"loss": loss.item()}
        ]

        if self.use_scaled_pos_enc:
            report_keys += [
                {"encoder_alpha": self.encoder.embed[-1].alpha.data.item()},
                {"decoder_alpha": self.decoder.embed[-1].alpha.data.item()},
            ]
        self.reporter.report(report_keys)
        return loss

    def inference(self, x: torch.LongTensor, y: torch.FloatTensor):
        ilens = torch.LongTensor([x.shape[0]]).to(x.device)
        xs = x.unsqueeze(0)
        in_masks = make_non_pad_mask(ilens).to(xs.device)
        _, outs = self._forward(xs, ilens, in_masks=in_masks, is_inference=True)
        return outs[0]

    # for reporting attentions
    def calculate_all_attentions(self,
                                 xs: torch.FloatTensor,
                                 ilens: torch.LongTensor,
                                 ys: torch.FloatTensor,
                                 olens: torch.LongTensor,
                                 ds: torch.LongTensor,
                                 ps: torch.FloatTensor,
                                 es: torch.FloatTensor):
        with torch.no_grad():
            # remove unnecessary padded part
            xs = xs[:, :max(ilens)]
            ds = ds[:, :max(ilens)]
            ys = ys[:, :max(olens)]
            ps = ps[:, :max(olens)]
            es = es[:, :max(olens)]
            in_masks = make_non_pad_mask(ilens).to(xs.device)
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(xs.device)
            outs = self._forward(xs, ilens, olens, ds, ps, es, in_masks, out_masks, is_inference=False)[0]

        att_ws_dict = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                atten = m.atten.cpu().numpy()
                if "encoder" in name:
                    atten = [a[:, :l, :l] for a, l in zip(atten, ilens.tolist())]
                elif "decoder" in name:
                    if "src" in name:
                        atten = [a[:, :ol, :il] for a, il, ol in zip(atten, ilens.tolist(), olens.tolist())]
                    elif "self" in name:
                        atten = [a[:, :l, :l] for a, l in zip(atten, olens.tolist())]
                    else:
                        logging.warning(f"unknown attention module: {name}")
                else:
                    logging.warning(f"unknown attention module: {name}")
                att_ws_dict[name] = atten
        att_ws_dict["predicted_fbank"] = [m[:l].T for m, l in zip(outs.cpu().numpy(), olens.tolist())]
        return att_ws_dict

    @property
    def attention_plot_class(self):
        return TTSPlot

    @property
    def base_plot_keys(self):
        plot_keys = ["loss", "l1_loss", "duration_loss"]
        if self.use_scaled_pos_enc:
            plot_keys += ["encoder_alpha", "decoder_alpha"]
        return plot_keys



