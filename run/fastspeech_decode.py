import os
import torch
from utils.dataset import FSDataset
from torch.utils.data import DataLoader
from utils.logger import TensorBoardLogger
from utils.kaldi_io_utils import write_ark_scp


def decode(cfg, model, current_step, criterion):
    logger = TensorBoardLogger(cfg.DIR_PATH.VAL_LOG)

    model.eval()
    with torch.no_grad():
        val_param = cfg.DATASET.VAL
        val_dataset = FSDataset(val_param.MEL_SCP,
                                val_param.VARIANCE_SCP,
                                val_param.JSON_DATA)
        opt_param = cfg.OPT
        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    collate_fn=val_dataset.collate_fn,
                                    num_workers=1)

        for batch in val_dataloader:
            token_ids, src_lengths, mel_targets, mel_lengths, duration_targets, pitch_targets, energy_targets, wav_ids \
                = batch

            wav_id = wav_ids[0]

            _, mel_output_postnet, duration_output, pitch_output, \
                energy_output, src_mask, mel_mask, mel_lengths_output = model.inference(token_ids, src_lengths,
                                                                                        duration_control=1.0,
                                                                                        pitch_control=1.0,
                                                                                        energy_control=1.0)





    model.train()