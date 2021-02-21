import os
import copy
import logging
import numpy as np
from os.path import basename
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from chainer.training import Trainer
from chainer.training.extension import Extension


def _plot_and_save_attention(att_w, filename):
    d = os.path.dirname(filename)
    if not os.path.exists(d):
        os.makedirs(d)
    w, h = plt.figaspect(1.0 / len(att_w))
    fig = plt.Figure(figsize=(w * 2, h * 2))
    axes = fig.subplots(1, len(att_w))
    if len(att_w) == 1:
        axes = [axes]
    for ax, aw in zip(axes, att_w):
        # plt.subplot(1, len(att_w), h)
        ax.imshow(aw.astype(np.float32), aspect="auto")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def savefig(plot, filename):
    plot.savefig(filename)
    plt.clf()


def plot_multi_head_attention(data, attn_dict, outdir, suffix="png", savefn=savefig):
    for name, att_ws in attn_dict.items():
        for idx, att_w in enumerate(att_ws):
            filename = "%s/%s.%s.%s" % (
                outdir, data[idx][0], name, suffix)
            dec_len = int(data[idx][1]['output'][0]['shape'][0])
            enc_len = int(data[idx][1]['input'][0]['shape'][0])
            if "encoder" in name:
                att_w = att_w[:, :enc_len, :enc_len]
            elif "decoder" in name:
                if "self" in name:
                    att_w = att_w[:, :dec_len, :dec_len]
                else:
                    att_w = att_w[:, :dec_len, :enc_len]
            else:
                logging.warning("unknown name for shaping attention")
            fig = _plot_and_save_attention(att_w, filename)
            savefn(fig, filename)


# removed some overrided methods
class TTSPlotAttentionReport(Extension):
    def __init__(self, att_vis_fn, data, outdir, converter, transform, device, reverse=False,
                 ikey="input", iaxis=0, okey="output", oaxis=0):
        self.att_vis_fn = att_vis_fn
        self.data = copy.deepcopy(data)
        self.outdir = outdir
        self.converter = converter
        self.transform = transform
        self.device = device
        self.reverse = reverse
        self.ikey = ikey
        self.iaxis = iaxis
        self.okey = okey
        self.oaxis = oaxis
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

    def get_attention_weight(self, idx, att_w):
        if self.reverse:
            dec_len = int(self.data[idx][1][self.ikey][self.iaxis]["shape"][0])
            enc_len = int(self.data[idx][1][self.okey][self.oaxis]["shape"][0])
        else:
            dec_len = int(self.data[idx][1][self.okey][self.oaxis]["shape"][0])
            enc_len = int(self.data[idx][1][self.ikey][self.iaxis]["shape"][0])
        if len(att_w.shape) == 3:
            att_w = att_w[:, :dec_len, :enc_len]
        else:
            att_w = att_w[:dec_len, :enc_len]
        return att_w

    def get_attention_weights(self):
        batch = self.converter([self.transform(self.data)], self.device)
        if isinstance(batch, tuple):
            att_ws = self.att_vis_fn(*batch)
        else:
            att_ws = self.att_vis_fn(**batch)
        return att_ws

    def draw_han_plot(self, att_w):
        if len(att_w.shape) == 3:
            for h, aw in enumerate(att_w, 1):
                legends = []
                plt.subplot(1, len(att_w), h)
                for i in range(aw.shape[1]):
                    plt.plot(aw[:, i])
                    legends.append(f"Att{i}")
                plt.ylim([0, 1.0])
                plt.xlim([0, aw.shape[0]])
                plt.grid(True)
                plt.ylabel("Attention Weight")
                plt.xlabel("Decoder Index")
                plt.legend(legends)
        else:
            legends = []
            for i in range(att_w.shape[1]):
                plt.plot(att_w[:, i])
                legends.append(f"Att{i}")
            plt.ylim([0, 1.0])
            plt.xlim([0, att_w.shape[0]])
            plt.grid(True)
            plt.ylabel("Attention Weight")
            plt.xlabel("Decoder Index")
            plt.legend(legends)
        plt.tight_layout()
        return plt

    def draw_attention_plot(self, att_w):
        att_w = att_w.astype(np.float32)
        if len(att_w.shape) == 3:
            for h, aw in enumerate(att_w, 1):
                plt.subplot(1, len(att_w), h)
                plt.imshow(aw, aspect="auto")
                plt.xlabel("Encoder Index")
                plt.ylabel("Decoder Index")
        else:
            plt.imshow(att_w, aspect="auto")
            plt.xlabel("Encoder Index")
            plt.ylabel("Decoder Index")
        plt.tight_layout()
        return plt

    def _plot_and_save_attention(self, att_w, filename, han_mode=False):
        if han_mode:
            plt = self.draw_han_plot(att_w)
        else:
            plt = self.draw_attention_plot(att_w)
        plt.savefig(filename)
        plt.close()

    def plotfn(self, *args, **kwargs):
        plot_multi_head_attention(*args, **kwargs)

    def __call__(self, trainer: Trainer):
        attn_dict = self.get_attention_weights()
        suffix = f"ep.{trainer.updater.epoch}.png"
        self.plotfn(self.data, attn_dict, self.outdir, suffix, savefig)

    def log_attentions(self, logger, step):
        def log_fig(plot, filename):
            logger.add_figure(basename(filename), plot, step)
            plt.clf()

        attn_dict = self.get_attention_weights()
        self.plotfn(self.data, attn_dict, self.outdir, "", log_fig)


class TTSPlot(TTSPlotAttentionReport):
    def plotfn(self, data, attn_dict, outdir, suffix="png", savefn=None):
        for name, att_ws in attn_dict.items():
            for idx, att_w in enumerate(att_ws):
                filename = "%s/%s.%s.%s" % (outdir, data[idx][0], name, suffix)
                if "fbank" in name:
                    fig = plt.Figure()
                    ax = fig.subplots(1, 1)
                    ax.imshow(att_w, aspect="auto")
                    ax.set_xlabel("frames")
                    ax.set_ylabel("fbank coeff")
                    fig.tight_layout()
                else:
                    fig = _plot_and_save_attention(att_w, filename)
                savefn(fig, filename)
