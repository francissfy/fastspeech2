import os
import shutil
import chainer
import torch
import tempfile
from chainer.serializers.npz import DictionarySerializer
from chainer.training import extension
from chainer.training import Trainer


def torch_snapshot(save_func=torch.save, filename="snapshot.sp.{.updator.epoch}"):
    @extension.make_extension(trigger=(1, "epoch"), priority=-100)
    def torch_snapshot(trainer):
        _torch_snapshot_object(trainer, trainer, filename.format(trainer), save_func)
    return torch_snapshot


def _torch_snapshot_object(trainer: Trainer, target, filename: str, savefunc):
    s = DictionarySerializer()
    s.save(trainer)
    # fixme remove parts for asr
    if hasattr(trainer.updater.model, "model"):
        if hasattr(trainer.updater.model.model, "module"):
            model_state_dict = trainer.updater.model.model.module.state_dict()
        else:
            model_state_dict = trainer.updater.model.model.state_dict()
    else:
        if hasattr(trainer.updater.model, "module"):
            model_state_dict = trainer.updater.model.module.state_dict()
        else:
            model_state_dict = trainer.updater.model.state_dict()
    snapshot_dict = {
        "trainer": s.target,
        "model": model_state_dict,
        "optimizer": trainer.updater.get_optimizer("main").state_dict()
    }

    # why such complex path
    fn = filename.format(trainer)
    prefix = "tmp" + fn
    tmpdir = tempfile.mktemp(prefix, trainer.out)
    tmppath = os.path.join(tmpdir, fn)
    try:
        savefunc(snapshot_dict, tmppath)
        shutil.move(tmppath, os.path.join(trainer.out, fn))
    finally:
        shutil.rmtree(tmpdir)


def snapshot_object(target, filename):
    @extension.make_extension(trigger=(1, "epoch"), priority=-100)
    def snapshot_object(trainer):
        torch.save(os.path.join(trainer.out, filename.format(trainer)), target)
    return snapshot_object

