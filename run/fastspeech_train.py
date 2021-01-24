import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from network.fastspeech2 import Fastspeech2
from network.loss import Fastspeech2Loss
from utils.logger import TensorBoardLogger
from utils.dataset import FSDataset
from utils.utils import get_device, get_model_num_params, load_from_ckpt
from network.optimizer import ScheduledOptimizer


def train(cfg):
    torch.manual_seed(1234)
    device = get_device()

    # dataset and dataloader
    traindata_param = cfg.DATASET.TRAIN
    dataset = FSDataset(traindata_param.MEL_SCP,
                        traindata_param.VARIANCE_SCP,
                        traindata_param.JSON_DATA
                        )
    opt_param = cfg.OPT
    dataloder = DataLoader(dataset=dataset,
                           batch_size=opt_param.BATCH_SIZE**2,
                           shuffle=True,
                           collate_fn=dataset.collate_fn,
                           drop_last=True,
                           num_workers=0
                           )
    print("loaded dataset and dataloader")

    # model setup
    # FIXME check for multigpu and do parallelization
    model = nn.DataParallel(Fastspeech2(cfg)).to(device)
    print("loaded model")
    num_params = get_model_num_params(model)
    print(f"model parameters: {num_params}")

    # uptimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 betas=opt_param.BETAS,
                                 eps=opt_param.EPS,
                                 weight_decay=opt_param.WEIGHT_DECAY
                                )
    decoder_hidden_dim = cfg.MODEL.DECODER.HIDDEN
    restore_step = opt_param.RESTORE_STEP
    scheduled_optimizer = ScheduledOptimizer(optimizer,
                                             decoder_hidden_dim,
                                             opt_param.N_WARM_UP_STEP,
                                             restore_step
                                             )
    loss_criteria = Fastspeech2Loss().to(device)
    print("loaded optimizer and loss function")

    # load previous ckpt
    dir_param = cfg.DIR_PATH
    ckpt_file = dir_param.CKPT
    if os.path.isfile(ckpt_file):
        load_from_ckpt(ckpt_file, model, optimizer)
        print(f"load state for model and optimizer from {ckpt_file} from step: {restore_step}")
    print("start new training")

    # TODO mkdir for ckpt and synthesis
    logdir = dir_param.LOG
    tensorboard_logger = TensorBoardLogger(logdir)

    # time_points = []
    # start = time.perf_counter()

    print("============ start training ============")
    model = model.train()

    num_epoch = opt_param.EPOCH
    batch_size = opt_param.BATCH_SIZE
    accum_steps = opt_param.ACCUM_STEPS
    grad_clip_thred = opt_param.GRAD_CLIP_THRESH

    total_steps = num_epoch*len(dataloder)*batch_size
    for epoch in range(num_epoch):

        for big_batch_idx, batch in enumerate(dataloder):
            for sub_batch_idx, sub_batch in enumerate(batch):

                current_step = big_batch_idx*batch_size + sub_batch_idx + restore_step + \
                                epoch*len(dataloder)*batch_size + 1

                # start_time = time.perf_counter()

                token_ids = sub_batch[0].long().to(device)
                src_lengths = torch.LongTensor(sub_batch[1]).to(device)
                mels = sub_batch[2].float().to(device)
                mel_lengths = torch.LongTensor(sub_batch[3]).to(device)
                durations = sub_batch[4].float().to(device)
                # TODO log duration in dataset
                log_durations = None
                pitchs = sub_batch[5].float().to(device)
                energies = sub_batch[6].float().to(device)

                max_src_len = max(sub_batch[1])
                max_mel_len = max(sub_batch[3])

                mel_output, log_duration_output, pitch_output, energy_output, src_mask, mel_mask, mel_len = \
                    model(token_ids, src_lengths, mel_lengths, durations, pitchs, energies, max_src_len, max_mel_len)

                mel_loss, duration_loss, pitch_loss, energy_loss = loss_criteria(log_duration_output,
                                                                                 log_durations,
                                                                                 pitch_output,
                                                                                 pitchs,
                                                                                 energy_output,
                                                                                 energies,
                                                                                 mel_output,
                                                                                 mels,
                                                                                 ~src_mask,
                                                                                 ~mel_mask)
                total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

                # accumulate gradient for large-batch backward
                total_loss /= accum_steps
                total_loss.backward()
                if current_step % accum_steps != 0:
                    continue

                # gradient clipping
                nn.utils.clip_grad_norm(model.parameters(), grad_clip_thred)

                # update weight
                scheduled_optimizer.step_update_lr()
                scheduled_optimizer.zero_grad()

                tensorboard_logger.log_loss(current_step, total_loss, mel_loss, duration_loss, pitch_loss, energy_loss)
                # end_time = time.perf_counter()
