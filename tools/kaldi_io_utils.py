import os
from tqdm import tqdm
from kaldi_io.kaldi_io import read_mat_scp, write_mat, open_or_fd

"""
decode the data from kaldi align
based on espnet

variance encoding: ([B, 5])
pitch_dim: :4
energy_dim: 4:

feat: ([B, 320])
320 dim out mel features

"""


class KaldiFeats:
    def __init__(self, mel_scp, variance_scp):
        self.mel_feats = {}
        self.variance_feats = {}
        self.load_variance_scp(variance_scp)
        self.load_mel_scp(mel_scp)
        assert len(self.mel_feats) == len(self.variance_feats), "number variance feats not equal to mel feats"

    def load_mel_scp(self, mel_scp):
        if len(self.mel_feats) == 0:
            data = read_mat_scp(mel_scp)
            for k, v in data:
                self.mel_feats[k] = v

    def load_variance_scp(self, variance_scp):
        if len(self.variance_feats) == 0:
            data = read_mat_scp(variance_scp)
            for k, v in data:
                self.variance_feats[k] = v

    def get_keys(self):
        keys = self.mel_feats.keys()
        keys = [k for k in keys]
        return keys

    def __len__(self):
        return len(self.mel_feats)

    def __getitem__(self, key):
        mel_feat = self.mel_feats[key] if key in self.mel_feats else None
        variance_feat = self.variance_feats[key] if key in self.variance_feats else None
        return mel_feat, variance_feat


def write_ark_scp(output_name, kmat_dict: dict):
    ark_scp_output = f"ark:| copy-feats --compress=true ark:- ark,scp:{output_name}.ark,{output_name}.scp"
    with open_or_fd(ark_scp_output, "wb") as f:
        for k, mat in kmat_dict.items():
            write_mat(f, mat, key=k)


def preprocess_kaldi_scp(kaldi_scp, fixed_scp, ark_base_dir):
    kaldi_scp_f = open(kaldi_scp, "r")
    fixed_scp_f = open(fixed_scp, "w")

    lines = kaldi_scp_f.readlines()
    num_lines = len(lines)

    with tqdm(total=num_lines) as pbar:
        for line in lines:
            tmp = line.strip().split(" ", maxsplit=1)
            wav_id = tmp[0]
            ark_path = tmp[1].rsplit("/", maxsplit=1)[1]
            ark_path = os.path.join(ark_base_dir, ark_path)
            fixed_scp_f.write(f"{wav_id} {ark_path}\n")
            pbar.update(1)

    kaldi_scp_f.close()
    fixed_scp_f.close()


if __name__ == "__main__":
    kaldi_scp = "/Users/francis/code/fastspeech2/local_test/mel/feats.1.scp"
    fixed_kaldi_scp = "/Users/francis/code/fastspeech2/local_test/mel/fixed_feats.1.scp"
    ark_base = "/Users/francis/code/fastspeech2/local_test/mel/"
    preprocess_kaldi_scp(kaldi_scp, fixed_kaldi_scp, ark_base)
