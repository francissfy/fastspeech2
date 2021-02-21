import torch
import numpy as np


class NoamOpt(object):
    def __init__(self,
                 model_size,
                 factor,
                 warmup,
                 optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_group(self):
        return self.optimizer.param_groups

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * self.model_size ** (-0.5) \
            * min(step ** (-0.5), step * self.warmup ** (-1.5))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict: dict):
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


def get_std_opt(model: torch.nn.Module, d_model, warmup, factor):
    base = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    return NoamOpt(d_model, factor, warmup, base)


class ScheduledOptimizer:
    def __init__(self, optimizer, decoder_hidden_dim, warm_up_steps, current_step):
        self.optimizer = optimizer
        self.decoder_hidden_dim = decoder_hidden_dim
        self.warm_up_steps = warm_up_steps
        self.current_steps = current_step
        self.init_lr = np.power(decoder_hidden_dim, -0.5)

    def update_lr(self):
        self.current_steps += 1
        lr_scale = np.min([
            np.power(self.current_steps, -0.5),
            np.power(self.warm_up_steps, -1.5) * self.current_steps
        ])
        lr = self.init_lr * lr_scale
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step_update_lr(self):
        self.update_lr()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
