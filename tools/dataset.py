import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from tools.kaldi_io_utils import KaldiFeats
from tools.utils import const_pad_tensors_dim1


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        super(TransformDataset, self).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])


class ChainerDataLoader(object):
    def __init__(self, **kwargs):
        self.loader = torch.utils.data.DataLoader(**kwargs)
        self.len = len(kwargs["dataset"])
        self.current_position = 0
        self.epoch = 0
        self.iter = None
        self.kwargs = kwargs

    def next(self):
        if self.iter is None:
            self.iter = iter(self.loader)
        try:
            ret = next(self.iter)
        except StopIteration:
            self.iter = None
            return self.next()
        self.current_position += 1
        if self.current_position == self.len:
            self.epoch += 1
            self.current_position = 0
        return ret

    def __iter__(self):
        for batch in self.loader:
            yield batch

    @property
    def epoch_detail(self):
        return self.epoch+self.current_position/self.len

    def serialize(self, serializer):
        epoch = serializer("epoch", self.epoch)
        current_position = serializer("current_position", self.current_position)
        self.epoch = epoch
        self.current_position = current_position

    def start_shuffle(self):
        self.kwargs["shuffle"] = True
        self.loader = torch.utils.data.dataloader.DataLoader(**self.kwargs)

    def finalize(self):
        del self.loader


class FSDataset(Dataset):
    def __init__(self, mel_scp, variance_scp, json_file):
        json_data = open(json_file, "r").read()
        json_data = json.loads(json_data)
        self.utts_data = json_data["utts"]
        self.utt_ids = self.utts_data.keys()
        self.kaldi_feats = KaldiFeats(mel_scp, variance_scp)

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        key = self.utt_ids[idx]
        return self.pack_data(key)

    def pack_data(self, key):
        # mel_feat: [L, 320]; variance_feat: [L, 5]
        mel_feat, variance_feat = self.kaldi_feats[key]
        energy = variance_feat[:, :4]
        pitch = variance_feat[:, 4]

        json_data = self.utts_data[key]
        duration = [t for t in json_data["input"] if t["name"] == "duration"][0]
        duration = [int(t) for t in duration.split(" ")]

        target1 = [t for t in json_data["output"] if t["name"] == "target1"][0]
        text, token, token_id = target1["text"], target1["token"], target1["token_id"]

        tp = (token_id,
              mel_feat,
              duration, pitch, energy,
              key)

        return tp

    # FIXME simplify the logic
    def pack_batch(self, batch, idxs):
        """
        pack the raw list of item from __getitem__ into smaller sub batches, used by collate_fn
        :param batch: list of item from __getitem__
        :param idxs: indexs
        :return: list of sub-batch
        """

        token_ids = [batch[i][0] for i in idxs]
        mels = [batch[i][1] for i in idxs]
        durations = [batch[i][2] for i in idxs]
        pitchs = [batch[i][3] for i in idxs]
        energies = [batch[i][4] for i in idxs]
        wav_ids = [batch[i][5] for i in idxs]

        mel_lengths = [m.shape[0] for m in mels]
        src_lengths = [s.shape[0] for s in token_ids]

        mels = const_pad_tensors_dim1(mels)
        durations = const_pad_tensors_dim1(durations)
        pitchs = const_pad_tensors_dim1(pitchs)
        energies = const_pad_tensors_dim1(energies)
        token_ids = const_pad_tensors_dim1(token_ids)

        tp = (token_ids, src_lengths,
              mels, mel_lengths,
              durations, pitchs, energies,
              wav_ids)

        return tp

    def collate_fn(self, batch):
        batch_size = len(batch)
        token_lengths = np.array([t[0].shape[0] for t in batch])
        argsort_idxs = np.argsort(-token_lengths)

        # cut the batch into sub batches
        step = int(math.sqrt(batch_size))
        s_idxs = [i for i in range(0, batch_size, step)]
        t_idxs = [min(i + step, batch_size) for i in s_idxs]
        sub_batches = [
            self.pack_batch(batch, argsort_idxs[s: t]) for s, t in zip(s_idxs, t_idxs)
        ]
        return sub_batches


if __name__ == "__main__":
    data_json = "/Users/francis/code/fastspeech2/local_test/updated_data.json"
    dataset = FSDataset("", "", data_json)
