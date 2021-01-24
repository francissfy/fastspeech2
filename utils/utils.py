import torch
from torch.nn import ConstantPad1d


def const_pad_tensors_dim1(tensors, padding_value=0):
    """
    tensors: list of tensor in shape: [L, W]
    """
    max_len = max([t.shape[0] for t in tensors])
    print(max_len)
    for i, t in enumerate(tensors):
        pad_len = max_len-t.shape[0]
        t = t.unsqueeze(dim=0)
        t = t.transpose(1, 2)
        pad = ConstantPad1d((0, pad_len), padding_value)
        tensors[i] = pad(t).transpose(1, 2)
    tensors = torch.cat(tensors, dim=0)
    return tensors


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_num_params(model):
    num_params = sum(param.numel() for param in model.parameters())
    return num_params


def load_from_ckpt(ckpt_file, model, optimizer):
    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimzer"])


if __name__ == "__main__":
    t1 = torch.ones((3, 5))
    t1[1, 2] = 12
    t2 = torch.ones((12, 5))
    t2[2, 3] = 12
    t3 = torch.ones((6, 5))
    t3[3, 4] = 12
    tt = const_pad_tensors_dim1([t1, t2, t3], 0)
    print(tt.shape)
    print(tt[0, 1, 2], tt[1, 2, 3], tt[2, 3, 4])
