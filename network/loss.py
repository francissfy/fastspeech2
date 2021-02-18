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


class Fastspeech2Loss(nn.Module):
    def __init__(self):
        super(Fastspeech2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self,
                log_d_predicted,
                log_d_target,
                p_predicted,
                p_target,
                e_predicted,
                e_target,
                mel,
                mel_target,
                src_mask,
                mel_mask):
        log_d_target.requires_grad = False
        p_target.requires_grad = False
        e_target.requires_grad = False
        mel_target.requires_grad = False

        log_d_predicted = log_d_predicted.masked_select(src_mask)
        log_d_target = log_d_target.masked_select(src_mask)

        p_predicted = p_predicted.masked_select(mel_mask)
        p_target = p_target.masked_select(mel_mask)
        e_predicted = e_predicted.masked_select(mel_mask)
        e_target = e_target.masked_select(mel_mask)

        mel = mel.masked_select(mel_mask.unsqueeze(-1))
        mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1))

        mel_loss = self.mse_loss(mel, mel_target)

        d_loss = self.mae_loss(log_d_predicted, log_d_target)
        p_loss = self.mae_loss(p_predicted, p_target)
        e_loss = self.mae_loss(e_predicted, e_target)

        return mel_loss, d_loss, p_loss, e_loss
