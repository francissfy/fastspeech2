import torch
import torch.nn as nn


class DurationPredictorLoss(nn.Module):
    def __init__(self, offset: float = 1.0):
        super(DurationPredictorLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.offset = offset

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)
        return loss
