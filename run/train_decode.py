import os
import json
import math
import copy
import logging
import chainer
import torch
import torch.nn as nn
import numpy as np
from chainer import training
from chainer.training.extension import Extension
from chainer.training.extensions import Evaluator, PlotReport, LogReport, PrintReport, ProgressBar
from chainer.training import StandardUpdater
from network.fastspeech2 import FeedForwardTransformer
from network.optimizer import get_std_opt
from network.utils import pad_list
from run.batchfy import make_batchset
from tools.load import LoadInputsAndTargets
from tools.dataset import ChainerDataLoader, TransformDataset
from tools.logger import TensorboardLogger
from tools.snapshot import torch_snapshot, snapshot_object
from torch.utils.tensorboard import SummaryWriter


class CustomEvaluator(Evaluator):
    def __init__(self, model: nn.Module, iterator, target, device):
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.device = device

    def evaluate(self):
        iterator = self._iterators["main"]
        if self.eval_hook:
            self.eval_hook(self)
        if hasattr(iterator, "reset"):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = chainer.reporter.DictSummary()
        self.model.eval()
        with torch.no_grad():
            for batch in it:
                if isinstance(batch, tuple):
                    x = tuple(arr.to(self.device) for arr in batch)
                else:
                    x = batch
                    for k in x.keys():
                        x[k] = x[k].to(self.device)
                observation = {}
                with chainer.reporter.report_scope(observation):
                    if isinstance(x, tuple):
                        self.model(*x)
                    else:
                        self.model(**x)
                summary.add(observation)
        self.model.train()
        return summary.compute_mean()

    def __call__(self, trainer=None):
        ret = super().__call__(trainer)
        try:
            if trainer is not None:
                tb_logger = trainer.get_extension(TensorboardLogger.default_name)
                tb_logger(trainer)
        except ValueError:
            pass
        return ret


class CustomUpdater(StandardUpdater):
    def __init__(self, model, grad_clip, iterator, optimizer, device, accum_grad=1):
        super(CustomUpdater, self).__init__(iterator, optimizer)
        self.model = model
        self.grad_clip = grad_clip
        self.device = device
        self.clip_grad_norm = nn.utils.clip_grad_norm_
        self.accum_grad = accum_grad
        self.forward_count = 0

    def update_core(self):
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")

        batch = train_iter.next()
        if isinstance(batch, tuple):
            x = tuple(arr.to(self.device) for arr in batch)
        else:
            x = batch
            for k in x.keys():
                x[k] = x[k].to(self.device)
        if isinstance(x, tuple):
            loss = self.model(*x).mean() / self.accum_grad
        else:
            loss = self.model(**x).mean() / self.accum_grad
        loss.backward()

        self.forward_count += 1
        if self.forward_count != self.accum_grad:
            return
        self.forward_count = 0

        grad_norm = self.clip_grad_norm(self.model.parameters(), self.grad_clip)
        logging.debug(f"grad norm={grad_norm}")
        if math.isnan(grad_norm):
            logging.warning("grad norm is nan. Do not update model")
        else:
            optimizer.step()
        optimizer.zero_grad()

    def upddate(self):
        self.update_core()
        if self.forward_count == 0:
            self.iteration += 1


class CustomConverter(object):
    def __init__(self):
        pass

    def __call__(self, batch, device=torch.device("cpu")):
        assert len(batch) == 1
        xs, spkids, ys, durs, variances = batch[0]

        ilens = torch.from_numpy(np.array([x.shape[0] for x in xs])).long().to(device)
        olens = torch.from_numpy(np.array([y.shape[0] for y in ys])).long().to(device)

        xs = pad_list([torch.from_numpy(x).long() for x in xs], 0).to(device)
        ys = pad_list([torch.from_numpy(y).long() for y in ys], 0).to(device)

        labels = ys.new_zeros(ys.shape[0], ys.shape[1])
        for i, l in enumerate(olens):
            labels[i, l - 1:] = 1.0

        new_batch = {
            "xs": xs,
            "ilens": ilens,
            "ys": ys,
            "labels": labels,
            "olens": olens
        }

        durs = pad_list([torch.from_numpy(dur).long() for dur in durs], 0).to(device)
        new_batch["ds"] = durs
        new_batch["spkids"] = torch.LongTensor(spkids).to(device)
        variances = pad_list([torch.from_numpy(v).float() for v in variances], 0).to(device)
        new_batch["ps"] = variances[:, :, :4]
        new_batch["es"] = variances[:, :, 4:]
        return new_batch


def build_model(idim, odim, args):
    model = FeedForwardTransformer(idim=idim,
                                   odim=odim,
                                   adim=args.adim,
                                   aheads=args.aheads,
                                   positionwise_layer_type=args.positionwise_layer_type,
                                   positionwise_conv_kernel_size=args.positionwise_conv_kernel_size,
                                   eunits=args.eunits,
                                   transformer_enc_dropout_rate=args.transformer_enc_dropout_rate,
                                   transformer_enc_positional_dropout_rate=args.transformer_enc_positional_dropout_rate,
                                   transformer_enc_atten_dropout_rate=args.transformer_enc_atten_dropout_rate,
                                   encoder_normalized_before=args.encoder_normalized_before,
                                   encoder_concat_after=args.encoder_concat_after,
                                   pitch_embed_kernel_size=args.pitch_embed_kernel_size,
                                   pitch_embed_dropout=args.pitch_embed_dropout,
                                   energy_embed_kernel_size=args.energy_embed_kernel_size,
                                   energy_embed_dropout=args.energy_embed_dropout,
                                   duration_predictor_layers=args.duration_predictor_layers,
                                   duration_predictor_chans=args.duration_predictor_chans,
                                   duration_predictor_kernel_size=args.duration_predictor_kernel_size,
                                   duration_predictor_dropout_rate=args.duration_predictor_dropout_rate,
                                   dlayers=args.dlayers,
                                   dunits=args.dunits,
                                   transformer_dec_dropout_rate=args.transformer_dec_dropout_rate,
                                   transformer_dec_positional_dropout_rate=args.transformer_dec_positional_dropout_rate,
                                   transformer_dec_atten_dropout_rate=args.transformer_dec_atten_dropout_rate,
                                   decoder_normalized_before=args.decoder_normalized_before,
                                   decoder_concat_after=args.decoder_concat_after,
                                   reduction_factor=args.reduction_factor,
                                   postnet_layers=args.postnet_layers,
                                   postnet_filts=args.postnet_filts,
                                   postnet_chans=args.postnet_chans,
                                   postnet_dropout_rate=args.postnet_dropout_rate,
                                   transformer_init=args.transformer_init,
                                   initial_encoder_alpha=args.initial_encoder_alpha,
                                   initial_decoder_alpha=args.initial_decoder_alpha,
                                   use_masking=args.use_masking,
                                   use_batch_norm=args.use_batch_norm,
                                   use_scaled_pos_enc=args.use_scaled_pos_enc)
    return model


def check_early_stop(trainer: training.Trainer, epochs):
    end_epoch = trainer.updater.get_iterator("main").epoch
    if end_epoch < (epochs-1):
        logging.warning(f"Hit early stop at epoch {end_epoch}\n"
                        f"You can change the patience or set it to 0 to run all epochs")


def set_early_stop(trainer: training.Trainer, args, is_lm=False):
    patience = args.patience
    criterion = args.early_stop_criterion
    epochs = args.epoch if is_lm else args.epochs
    mode = 'max' if 'acc' in criterion else 'min'
    if patience > 0:
        trainer.stop_trigger = chainer.training.triggers.EarlyStoppingTrigger(monitor=criterion,
                                                                              mode=mode,
                                                                              patients=patience,
                                                                              max_trigger=(epochs, 'epoch'))


class ShufflingEnable(Extension):
    def __init__(self, iterators):
        self.set = False
        self.iterators = iterators

    def __call__(self, trainer):
        if not self.set:
            for iterator in self.iterators:
                iterator.start_shuffle()
            self.set = True


def train(args):
    # cuda check
    if not torch.cuda.is_available():
        logging.warning("cuda is not available")
    # valid json
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]
    utts = list(valid_json.keys())
    # inout dims
    idim = int(valid_json[utts[0]]['output'][0]['shape'][1])
    odim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    logging.info(f"#input dims : {idim}")
    logging.info(f"#output dims : {odim}")
    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + "/model.json"
    with open(model_conf, "wb") as f:
        logging.info(f"writing a model config file to {model_conf}")
        f.write(
            json.dumps((idim, odim, vars(args)), indent=4, ensure_ascii=False, sort_keys=True).encode("utf_8")
        )
    for key in sorted(vars(args).keys()):
        logging.info(f"ARGS: {key}:{vars(args)[key]}")

    # model arch
    model = build_model(idim, odim, args)
    logging.info(model)
    reporter = model.reporter

    # check multi-gpus
    if args.ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        if args.batch_size != 0:
            logging.warning(
                f"batch size is automatically increased ({args.batch_size} -> {args.batch_size * args.ngpu}")
            args.batch_size *= args.ngpu
    # set torch device
    device = torch.device("gpu" if args.ngpu > 1 else "cpu")
    model = model.to(device)

    # optimizer
    if args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr, eps=args.eps, weight_decay=args.weight_decay)
    elif args.opt == "noam":
        optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
    else:
        raise NotImplementedError(f"unknown optimizer: {args.opt}")

    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # read json data
    with open(args.train_json, "rb") as f:
        train_json = json.load(f)["utts"]
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    if use_sortagrad:
        args.batch_sort_key = "input"

    # make minibatch list
    train_batchset = make_batchset(data=train_json,
                                   num_batches=args.minibatches,
                                   min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                                   shortest_first=use_sortagrad,
                                   batch_sort_key=args.batch_sort_key,
                                   swap_io=True,
                                   count=args.batch_count,
                                   batch_bins=args.batch_bins)
    valid_batchset = make_batchset(data=valid_json,
                                   num_batches=args.minibatches,
                                   min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                                   shortest_first=use_sortagrad,
                                   batch_sort_key=args.batch_sort_key,
                                   swap_io=True,
                                   count=args.batch_count,
                                   batch_bins=args.batch_bins)
    # preprocess_args: dummy
    load_tr = LoadInputsAndTargets(mode="tts",
                                   preprocess_conf=args.preprocess_conf,
                                   use_speaker_embedding=args.use_speaker_embedding,
                                   use_second_target=args.use_second_target,
                                   preprocess_args={"train": True},
                                   keep_all_data_on_mem=args.keep_all_data_on_mem)
    load_cv = LoadInputsAndTargets(mode="tts",
                                   preprocess_conf=args.preprocess_conf,
                                   use_speaker_embedding=args.use_speaker_embedding,
                                   use_second_target=args.use_second_target,
                                   preprocess_args={"train": True},
                                   keep_all_data_on_mem=args.keep_all_data_on_mem)
    converter = CustomConverter()
    train_iter = {
        "main": ChainerDataLoader(dataset=TransformDataset(train_batchset, lambda data: converter([load_tr(data)])),
                                  batch_size=1, num_workers=args.num_iter_processes,
                                  shuffle=not use_sortagrad, collate_fn=lambda x: x[0])
    }
    valid_iter = {
        "main": ChainerDataLoader(dataset=TransformDataset(valid_batchset, lambda data: converter([load_tr(data)])),
                                  batch_size=1, num_workers=args.num_iter_processes,
                                  shuffle=not use_sortagrad, collate_fn=lambda x: x[0])
    }

    # setup trainer
    updater = CustomUpdater(model, args.grad_clip, train_iter, optimizer, device, args.accum_grad)
    trainer = training.Trainer(updater, (args.epochs, "epoch"), out=args.outdir)

    if args.resume:
        raise NotImplementedError("resume not implemented")

    eval_interval = (args.eval_interval_epochs, "epoch")
    save_interval = (args.save_interval_epochs, "epoch")
    report_interval = (args.report_interval_iters, "iteration")

    # evaluate the model
    trainer.extend(CustomEvaluator(model, valid_iter, reporter, device), trigger=eval_interval)
    # save sanpshot
    trainer.extend(torch_snapshot(), trigger=save_interval)
    # save the best model
    trainer.extend(snapshot_object(model, "model.loss.best"),
                   trigger=training.triggers.MinValueTrigger("validation/main/loss", trigger=eval_interval))

    # save attention figure
    if args.num_save_attention > 0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]["input"][0]["shape"][1]), reverse=True)
        if hasattr(model, "module"):
            att_vis_fn = model.module.calculate_all_attentions
            plot_class = model.module.attention_plot_class
        else:
            att_vis_fn = model.calculate_all_attentions
            plot_class = model.attention_plot_class
        att_reporter = plot_class(att_vis_fn, data, args.outdir+"/att_ws", converter=converter,
                                  transform=load_cv, device=device, reverse=True)
        trainer.extend(att_reporter, trigger=eval_interval)
    else:
        att_reporter = None

    # training and validation loss
    if hasattr(model, "module"):
        base_plot_keys = model.module.base_plot_keys
    else:
        base_plot_keys = model.base_plot_keys
    plot_keys = []
    for key in base_plot_keys:
        plot_key = ["main/"+key, "validation/main/"+key]
        trainer.extend(PlotReport(plot_key, "epoch", filename=key+".png"), trigger=eval_interval)
        plot_keys += plot_key
    trainer.extend(PlotReport(plot_keys, "epoch", filename="all_loss.png"), trigger=eval_interval)

    # write evaluation statistics
    trainer.extend(LogReport(trigger=report_interval))
    report_keys = ["epoch", "iteration", "elapsed_time"] + plot_keys
    trainer.extend(PrintReport(report_keys), trigger=report_interval)
    trainer.extend(ProgressBar(), trigger=report_interval)

    set_early_stop(trainer, args)
    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        writer = SummaryWriter(args.tensorboard_dir)
        trainer.extend(TensorboardLogger(writer, att_reporter), trigger=report_interval)

    if use_sortagrad:
        trainer.extend(ShufflingEnable([train_iter]),
                       trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, 'epoch'))

    trainer.run()
    check_early_stop(trainer, args.epochs)




