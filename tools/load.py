import logging
import numpy as np
from kaldi_io.kaldi_io import read_mat_scp, read_mat
from collections import OrderedDict


class LoadInputsAndTargets(object):
    def __init__(self,
                 mode="tts",
                 preprocess_conf=None,
                 load_input=True,
                 load_output=True,
                 sort_in_input_length=True,
                 use_speaker_embedding=False,
                 use_second_target=False,
                 preprocess_args=None,
                 keep_all_data_on_mem=False):
        self._loaders = {}
        if mode not in ["tts"]:
            raise NotImplementedError("only tts is implemented")
        # TODO
        if preprocess_conf is not None:
            raise NotImplementedError("preprocess_conf=None is not implemented")
        else:
            self.preprocessing = None

        if use_second_target and use_speaker_embedding and mode == 'tts':
            raise ValueError("Choose one of use_second_target and use_speaker_embedding")

        self.mode = mode
        self.load_output = load_output
        self.load_input = load_input
        self.sort_in_input_length = sort_in_input_length
        self.use_speaker_embedding = use_speaker_embedding
        self.use_second_target = use_second_target
        if preprocess_args is None:
            self.preprocess_args = {}
        else:
            assert isinstance(preprocess_args, dict), type(preprocess_args)
            self.preprocess_args = dict(preprocess_args)

        self.keep_all_data_on_mem = keep_all_data_on_mem

    def _get_from_loader(self, filepath, filetype):
        if filetype in ["mat", "vec"]:
            if not self.keep_all_data_on_mem:
                return read_mat(filepath)
            if filepath not in self._loaders:
                self._loaders[filepath] = read_mat(filepath)
            return self._loaders[filepath]
        elif filetype == "scp":
            filepath, key = filepath.split(":", 1)
            loader = self._loaders.get(filepath)
            if loader is None:
                loader = read_mat_scp(filepath)
                self._loaders[filepath] = loader
            return loader[key]
        else:
            raise NotImplementedError(f"Not supported: loader_type={filetype}")

    def _create_batch_tts(self, x_feats_dict, y_feats_dict, uttid_list, eos):
        xs = list(y_feats_dict.values())[0]
        nonzero_idx = list(filter(lambda i: len(xs[i]>0), range(len(xs))))

        if self.sort_in_input_length:
            nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(xs[i]))
        else:
            nonzero_sorted_idx = nonzero_idx

        xs = [xs[i] for i in nonzero_sorted_idx]
        uttid_list = [uttid_list[i] for i in nonzero_sorted_idx]
        x_name = list(y_feats_dict.keys())[0]

        spkid_name = "spkid"
        spkids = y_feats_dict.get("speaker_id")
        spkids = [spkids[i] for i in nonzero_sorted_idx]

        if self.load_input:
            y_name = "input1"
            ys = list(x_feats_dict.values())[0]
            assert len(xs) == len(ys), (len(xs), len(ys))
            ys = [ys[i] for i in nonzero_sorted_idx]

            durs_name = "duration"
            durs = x_feats_dict[durs_name]
            durs = [durs[i] for i in nonzero_sorted_idx]

            variance_name = "variance"
            variances = x_feats_dict[variance_name]
            variances = [variances[i] for i in nonzero_sorted_idx]

            assert spkids is not None

            return_batch = OrderedDict([(x_name, xs),
                                        (spkid_name, spkids),
                                        (y_name, ys),
                                        (durs_name, durs),
                                        (variance_name, variances)])
        else:
            return_batch = OrderedDict([(x_name, xs),
                                        (spkid_name, spkids)])
        return return_batch, uttid_list

    def __call__(self, batch):
        x_feats_dict = OrderedDict()
        y_feats_dict = OrderedDict()
        uttid_list = []

        for uttid, info in batch:
            for idx, inp in enumerate(info["input"]):
                if inp["name"] == "duration":
                    x = np.array(list(map(int, inp["duration"].split())), dtype=np.int64)
                elif "feat" in inp:
                    x = self._get_from_loader(filepath=inp["feat"], filetype=inp.get("filetype", "mat"))
                x_feats_dict.setdefault(inp["name"], []).append(x)

            if self.load_output:
                if self.mode == "mt":
                    raise NotImplementedError("mode=mt is not implemented")
                for idx, inp in enumerate(info["output"]):
                    if inp["name"] == "speaker_id":
                        x = int(inp["speaker_id"])
                    elif "tokenid" in inp:
                        x = np.fromiter(map(int, inp["tokenid"].split()), dtype=np.int64)
                    else:
                        x = self._get_from_loader(filepath=inp["feat"], filetype=inp.get("filetype", "mat"))
                    y_feats_dict.setdefault(inp["name"], []).append(x)

        if self.mode == "tts":
            _, info = batch[0]
            eos = int(info["output"][0]["shape"][1])-1
            return_batch, uttid_list = self._create_batch_tts(x_feats_dict, y_feats_dict, uttid_list)
        else:
            raise NotImplementedError("mode other than tts not implemented")

        if self.preprocessing is not None:
            for x_name in return_batch.keys():
                if x_name.startswith("input"):
                    return_batch[x_name] = self.preprocessing(return_batch[x_name], uttid_list, **self.preprocess_args)
        return tuple(return_batch.values())
