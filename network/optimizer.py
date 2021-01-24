import numpy as np


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
            np.power(self.warm_up_steps, -1.5)*self.current_steps
        ])
        lr = self.init_lr * lr_scale
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step_update_lr(self):
        self.update_lr()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
