<<<<<<< HEAD
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from torch.distributions import laplace
from torch.distributions import uniform
from torch.nn.modules.loss import _Loss

from ..utils import clamp
from ..utils import clamp_by_pnorm
from ..utils import batch_multiply
from ..utils import normalize_by_pnorm
from ..utils import predict_from_logits
from ..attacks import Attack, LabelMixin


def rand_init_delta(delta, x, ord, eps, clip_min, clip_max):

    if isinstance(eps, torch.Tensor):
        assert len(eps) == len(delta)

    if ord == np.inf:
        delta.data.uniform_(-1, 1)
    elif ord == 2:
        delta.data.uniform_(clip_min, clip_max)
        delta.data = delta.data - x
        delta.data = clamp_by_pnorm(delta.data, ord, eps)
    elif ord == 1:
        ini = laplace.Laplace(
            loc=delta.new_tensor(0), scale=delta.new_tensor(1))
        delta.data = ini.sample(delta.data.shape)
        delta.data = normalize_by_pnorm(delta.data, p=1)
        ray = uniform.Uniform(0, eps).sample()
        delta.data *= ray
        delta.data = clamp(x.data + delta.data, clip_min, clip_max) - x.data
    else:
        error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
        raise NotImplementedError(error)

    delta.data = clamp(
        x + delta.data, min=clip_min, max=clip_max) - x
    return delta.data


def is_successful(y1, y2, targeted):
    if targeted is True:
        return y1 == y2
    else:
        return y1 != y2


class AttackConfig(object):
    def __init__(self):
        self.kwargs = {}

        for mro in reversed(self.__class__.__mro__):
            if mro in (AttackConfig, object):
                continue
            for kwarg in mro.__dict__:
                if kwarg in self.AttackClass.__init__.__code__.co_varnames:
                    self.kwargs[kwarg] = mro.__dict__[kwarg]
                else:
                    # make sure we don't specify wrong kwargs
                    assert kwarg in ["__module__", "AttackClass", "__doc__"]

    def __call__(self, *args):
        adversary = self.AttackClass(*args, **self.kwargs)
        print(self.AttackClass, args, self.kwargs)
        return adversary


def multiple_mini_batch_attack(
        adversary, loader, device="cuda", save_adv=False,
        norm=None, num_batch=None, num_pred_output=1):
    lst_label = []
    lst_pred = []
    lst_advpred = []
    lst_dist = []

    _norm_convert_dict = {"Linf": "inf", "L2": 2, "L1": 1}
    if norm in _norm_convert_dict:
        norm = _norm_convert_dict[norm]

    if norm == "inf":
        def dist_func(x, y):
            return (x - y).view(x.size(0), -1).max(dim=1)[0]
    elif norm == 1 or norm == 2:
        from ..utils import _get_norm_batch

        def dist_func(x, y):
            return _get_norm_batch(x - y, norm)
    else:
        assert norm is None


    idx_batch = 0
    from tqdm import tqdm
    for data, label in tqdm(loader):
        data, label = data.to(device), label.to(device)
        adv = adversary.perturb(data, label)
        if num_pred_output == 1:
            advpred = predict_from_logits(adversary.predict(adv))
            pred = predict_from_logits(adversary.predict(data))
        else:
            advpred = predict_from_logits(adversary.predict(adv)[0])
            pred = predict_from_logits(adversary.predict(data)[0])
            
        lst_label.append(label)
        lst_pred.append(pred)
        lst_advpred.append(advpred)
        if norm is not None:
            lst_dist.append(dist_func(data, adv))

        idx_batch += 1
        if idx_batch == num_batch:
            break

    return torch.cat(lst_label), torch.cat(lst_pred), torch.cat(lst_advpred), \
        torch.cat(lst_dist) if norm is not None else None


class MarginalLoss(_Loss):
    def forward(self, logits, targets):  # pylint: disable=arguments-differ
        assert logits.shape[-1] >= 2
        top_logits, top_classes = torch.topk(logits, 2, dim=-1)
        target_logits = logits[torch.arange(logits.shape[0]), targets]
        max_nontarget_logits = torch.where(
            top_classes[..., 0] == targets,
            top_logits[..., 1],
            top_logits[..., 0],
        )

        loss = max_nontarget_logits - target_logits
        if self.reduction == "none":
            pass
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        else:
            raise ValueError("unknown reduction: '%s'" % (self.recution,))

        return loss


class ChooseBestAttack(Attack, LabelMixin):
    def __init__(self, predict, base_adversaries, loss_fn=None,
                 targeted=False):
        self.predict = predict
        self.base_adversaries = base_adversaries
        self.loss_fn = loss_fn
        self.targeted = targeted

        for adversary in self.base_adversaries:
            assert self.targeted == adversary.targeted

    def perturb(self, x, y=None):

        x, y = self._verify_and_process_inputs(x, y)

        with torch.no_grad():
            maxloss = self.loss_fn(self.predict(x), y)
        final_adv = torch.zeros_like(x)
        for adversary in self.base_adversaries:
            adv = adversary.perturb(x, y)
            loss = self.loss_fn(self.predict(adv), y)
            to_replace = maxloss < loss
            final_adv[to_replace] = adv[to_replace]
            maxloss[to_replace] = loss[to_replace]

        return final_adv


def attack_whole_dataset(adversary, loader, device="cuda"):
    lst_adv = []
    lst_label = []
    lst_pred = []
    lst_advpred = []
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        pred = predict_from_logits(adversary.predict(data))
        adv = adversary.perturb(data, label)
        advpred = predict_from_logits(adversary.predict(adv))
        lst_label.append(label)
        lst_pred.append(pred)
        lst_advpred.append(advpred)
        lst_adv.append(adv)
    return torch.cat(lst_adv), torch.cat(lst_label), torch.cat(lst_pred), \
        torch.cat(lst_advpred)
=======
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from torch.distributions import laplace
from torch.distributions import uniform
from torch.nn.modules.loss import _Loss

from ..utils import clamp
from ..utils import clamp_by_pnorm
from ..utils import batch_multiply
from ..utils import normalize_by_pnorm
from ..utils import predict_from_logits
from ..attacks import Attack, LabelMixin


def rand_init_delta(delta, x, ord, eps, clip_min, clip_max):

    if isinstance(eps, torch.Tensor):
        assert len(eps) == len(delta)

    if ord == np.inf:
        delta.data.uniform_(-1, 1)
    elif ord == 2:
        delta.data.uniform_(clip_min, clip_max)
        delta.data = delta.data - x
        delta.data = clamp_by_pnorm(delta.data, ord, eps)
    elif ord == 1:
        ini = laplace.Laplace(
            loc=delta.new_tensor(0), scale=delta.new_tensor(1))
        delta.data = ini.sample(delta.data.shape)
        delta.data = normalize_by_pnorm(delta.data, p=1)
        ray = uniform.Uniform(0, eps).sample()
        delta.data *= ray
        delta.data = clamp(x.data + delta.data, clip_min, clip_max) - x.data
    else:
        error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
        raise NotImplementedError(error)

    delta.data = clamp(
        x + delta.data, min=clip_min, max=clip_max) - x
    return delta.data


def is_successful(y1, y2, targeted):
    if targeted is True:
        return y1 == y2
    else:
        return y1 != y2


class AttackConfig(object):
    def __init__(self):
        self.kwargs = {}

        for mro in reversed(self.__class__.__mro__):
            if mro in (AttackConfig, object):
                continue
            for kwarg in mro.__dict__:
                if kwarg in self.AttackClass.__init__.__code__.co_varnames:
                    self.kwargs[kwarg] = mro.__dict__[kwarg]
                else:
                    # make sure we don't specify wrong kwargs
                    assert kwarg in ["__module__", "AttackClass", "__doc__"]

    def __call__(self, *args):
        adversary = self.AttackClass(*args, **self.kwargs)
        print(self.AttackClass, args, self.kwargs)
        return adversary


def multiple_mini_batch_attack(
        adversary, loader, device="cuda", save_adv=False,
        norm=None, num_batch=None, num_pred_output=1):
    lst_label = []
    lst_pred = []
    lst_advpred = []
    lst_dist = []

    _norm_convert_dict = {"Linf": "inf", "L2": 2, "L1": 1}
    if norm in _norm_convert_dict:
        norm = _norm_convert_dict[norm]

    if norm == "inf":
        def dist_func(x, y):
            return (x - y).view(x.size(0), -1).max(dim=1)[0]
    elif norm == 1 or norm == 2:
        from ..utils import _get_norm_batch

        def dist_func(x, y):
            return _get_norm_batch(x - y, norm)
    else:
        assert norm is None


    idx_batch = 0
    from tqdm import tqdm
    for data, label in tqdm(loader):
        data, label = data.to(device), label.to(device)
        adv = adversary.perturb(data, label)
        if num_pred_output == 1:
            advpred = predict_from_logits(adversary.predict(adv))
            pred = predict_from_logits(adversary.predict(data))
        else:
            advpred = predict_from_logits(adversary.predict(adv)[0])
            pred = predict_from_logits(adversary.predict(data)[0])
            
        lst_label.append(label)
        lst_pred.append(pred)
        lst_advpred.append(advpred)
        if norm is not None:
            lst_dist.append(dist_func(data, adv))

        idx_batch += 1
        if idx_batch == num_batch:
            break

    return torch.cat(lst_label), torch.cat(lst_pred), torch.cat(lst_advpred), \
        torch.cat(lst_dist) if norm is not None else None


class MarginalLoss(_Loss):
    def forward(self, logits, targets):  # pylint: disable=arguments-differ
        assert logits.shape[-1] >= 2
        top_logits, top_classes = torch.topk(logits, 2, dim=-1)
        target_logits = logits[torch.arange(logits.shape[0]), targets]
        max_nontarget_logits = torch.where(
            top_classes[..., 0] == targets,
            top_logits[..., 1],
            top_logits[..., 0],
        )

        loss = max_nontarget_logits - target_logits
        if self.reduction == "none":
            pass
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        else:
            raise ValueError("unknown reduction: '%s'" % (self.recution,))

        return loss


class ChooseBestAttack(Attack, LabelMixin):
    def __init__(self, predict, base_adversaries, loss_fn=None,
                 targeted=False):
        self.predict = predict
        self.base_adversaries = base_adversaries
        self.loss_fn = loss_fn
        self.targeted = targeted

        for adversary in self.base_adversaries:
            assert self.targeted == adversary.targeted

    def perturb(self, x, y=None):

        x, y = self._verify_and_process_inputs(x, y)

        with torch.no_grad():
            maxloss = self.loss_fn(self.predict(x), y)
        final_adv = torch.zeros_like(x)
        for adversary in self.base_adversaries:
            adv = adversary.perturb(x, y)
            loss = self.loss_fn(self.predict(adv), y)
            to_replace = maxloss < loss
            final_adv[to_replace] = adv[to_replace]
            maxloss[to_replace] = loss[to_replace]

        return final_adv


def attack_whole_dataset(adversary, loader, device="cuda"):
    lst_adv = []
    lst_label = []
    lst_pred = []
    lst_advpred = []
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        pred = predict_from_logits(adversary.predict(data))
        adv = adversary.perturb(data, label)
        advpred = predict_from_logits(adversary.predict(adv))
        lst_label.append(label)
        lst_pred.append(pred)
        lst_advpred.append(advpred)
        lst_adv.append(adv)
    return torch.cat(lst_adv), torch.cat(lst_label), torch.cat(lst_pred), \
        torch.cat(lst_advpred)
>>>>>>> 4b57aae98d230f0282c345e6775888feda641368
