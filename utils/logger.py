import torch
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from torch.utils.tensorboard import SummaryWriter
matplotlib.use("Agg")


# here use the logging method in NVIDIA Tacotron2
def save_fig_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_fig_to_numpy(fig)
    plt.close()
    return data


class TensorBoardLogger(SummaryWriter):
    def __init__(self, logdir):
        super(TensorBoardLogger, self).__init__(logdir)

    def log_loss(self, current_step, total_loss, mel_loss, duration_loss, pitch_loss, energy_loss):
        self.add_scalar("loss/total", total_loss, current_step)
        self.add_scalar("loss/mel", mel_loss, current_step)
        self.add_scalar("loss/duration", duration_loss, current_step)
        self.add_scalar("loss/pitch", pitch_loss, current_step)
        self.add_scalar("loss/energy", energy_loss, current_step)
