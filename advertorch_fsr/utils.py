<<<<<<< HEAD
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_allclose(x, y, rtol=1.e-5, atol=1.e-8):
    import numpy as np
    return np.allclose(x.detach().cpu().numpy(), y.detach().cpu().numpy(),
                       rtol=rtol, atol=atol)


def single_dim_flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    indices = torch.arange(
        x.size(dim) - 1, -1, -1,
        dtype=torch.long, device=x.device, requires_grad=x.requires_grad)
    return x.index_select(dim, indices)


def torch_flip(x, dims):
    for dim in dims:
        x = single_dim_flip(x, dim)
    return x


def replicate_input(x):
    return x.detach().clone()


def replicate_input_withgrad(x):
    return x.detach().clone().requires_grad_()


def calc_l2distsq(x, y):
    d = (x - y)**2
    return d.view(d.shape[0], -1).sum(dim=1)


def calc_l1dist(x, y):
    d = torch.abs(x - y)
    return d.view(d.shape[0], -1).sum(dim=1)


def tanh_rescale(x, x_min=-1., x_max=1.):
    return (torch.tanh(x)) * 0.5 * (x_max - x_min) + (x_max + x_min) * 0.5


def torch_arctanh(x, eps=1e-6):
    return (torch.log((1 + x) / (1 - x))) * 0.5


def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input


def to_one_hot(y, num_classes=10):
    y = replicate_input(y).view(-1, 1)
    y_one_hot = y.new_zeros((y.size()[0], num_classes)).scatter_(1, y, 1)
    return y_one_hot


class CarliniWagnerLoss(nn.Module):
    def __init__(self):
        super(CarliniWagnerLoss, self).__init__()

    def forward(self, input, target):
        num_classes = input.size(1)
        label_mask = to_one_hot(target, num_classes=num_classes).float()
        correct_logit = torch.sum(label_mask * input, dim=1)
        wrong_logit = torch.max((1. - label_mask) * input, dim=1)[0]
        loss = -F.relu(correct_logit - wrong_logit + 50.).sum()
        return loss


def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    return (
        batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()


def _batch_clamp_tensor_by_vector(vector, batch_tensor):
    return torch.min(
        torch.max(batch_tensor.transpose(0, -1), -vector), vector
    ).transpose(0, -1).contiguous()


def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def batch_clamp(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_clamp_tensor_by_vector(float_or_vector, tensor)
        return tensor
    elif isinstance(float_or_vector, float):
        tensor = clamp(tensor, -float_or_vector, float_or_vector)
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def _get_norm_batch(x, p):
    batch_size = x.size(0)
    return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)


def _thresh_by_magnitude(theta, x):
    return torch.relu(torch.abs(x) - theta) * x.sign()


def batch_l1_proj_flat(x, z=1):
    v = torch.abs(x)
    v = v.sum(dim=1)

    indexes_b = torch.nonzero(v > z).view(-1)
    if isinstance(z, torch.Tensor):
        z = z[indexes_b][:, None]
    x_b = x[indexes_b]
    batch_size_b = x_b.size(0)

    if batch_size_b == 0:
        return x

    view = x_b
    view_size = view.size(1)
    mu = view.abs().sort(1, descending=True)[0]
    vv = torch.arange(view_size).float().to(x.device)
    st = (mu.cumsum(1) - z) / (vv + 1)
    u = (mu - st) > 0
    if u.dtype.__str__() == "torch.bool":  # after and including torch 1.2
        rho = (~u).cumsum(dim=1).eq(0).sum(1) - 1
    else:
        rho = (1 - u).cumsum(dim=1).eq(0).sum(1) - 1
    theta = st.gather(1, rho.unsqueeze(1))
    proj_x_b = _thresh_by_magnitude(theta, x_b)

    proj_x = x.detach().clone()
    proj_x[indexes_b] = proj_x_b
    return proj_x


def batch_l1_proj(x, eps):
    batch_size = x.size(0)
    view = x.view(batch_size, -1)
    proj_flat = batch_l1_proj_flat(view, z=eps)
    return proj_flat.view_as(x)


def clamp_by_pnorm(x, p, r):
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    if isinstance(r, torch.Tensor):
        assert norm.size() == r.size()
    else:
        assert isinstance(r, float)
    factor = torch.min(r / norm, torch.ones_like(norm))
    return batch_multiply(factor, x)


def is_float_or_torch_tensor(x):
    return isinstance(x, torch.Tensor) or isinstance(x, float)


def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return batch_multiply(1. / norm, x)


def jacobian(model, x, output_class):
    xvar = replicate_input_withgrad(x)
    scores = model(xvar)

    torch.sum(scores[:, output_class]).backward()

    return xvar.grad.detach().clone()


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def normalize_fn(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


def batch_per_image_standardization(imgs):
    assert imgs.ndimension() == 4
    mean = imgs.view(imgs.shape[0], -1).mean(dim=1).view(
        imgs.shape[0], 1, 1, 1)
    return (imgs - mean) / batch_adjusted_stddev(imgs)


def batch_adjusted_stddev(imgs):
    std = imgs.view(imgs.shape[0], -1).std(dim=1).view(imgs.shape[0], 1, 1, 1)
    std_min = 1. / imgs.new_tensor(imgs.shape[1:]).prod().float().sqrt()
    return torch.max(std, std_min)


class PerImageStandardize(nn.Module):
    def __init__(self):
        super(PerImageStandardize, self).__init__()

    def forward(self, tensor):
        return batch_per_image_standardization(tensor)


def predict_from_logits(logits, dim=1):
    return logits.max(dim=dim, keepdim=False)[1]


def get_accuracy(pred, target):
    return pred.eq(target).float().mean().item()


def set_torch_deterministic():
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def set_seed(seed=None):
    import torch
    import numpy as np
    import random
    if seed is not None:
        torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
=======
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_allclose(x, y, rtol=1.e-5, atol=1.e-8):
    import numpy as np
    return np.allclose(x.detach().cpu().numpy(), y.detach().cpu().numpy(),
                       rtol=rtol, atol=atol)


def single_dim_flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    indices = torch.arange(
        x.size(dim) - 1, -1, -1,
        dtype=torch.long, device=x.device, requires_grad=x.requires_grad)
    return x.index_select(dim, indices)


def torch_flip(x, dims):
    for dim in dims:
        x = single_dim_flip(x, dim)
    return x


def replicate_input(x):
    return x.detach().clone()


def replicate_input_withgrad(x):
    return x.detach().clone().requires_grad_()


def calc_l2distsq(x, y):
    d = (x - y)**2
    return d.view(d.shape[0], -1).sum(dim=1)


def calc_l1dist(x, y):
    d = torch.abs(x - y)
    return d.view(d.shape[0], -1).sum(dim=1)


def tanh_rescale(x, x_min=-1., x_max=1.):
    return (torch.tanh(x)) * 0.5 * (x_max - x_min) + (x_max + x_min) * 0.5


def torch_arctanh(x, eps=1e-6):
    return (torch.log((1 + x) / (1 - x))) * 0.5


def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input


def to_one_hot(y, num_classes=10):
    y = replicate_input(y).view(-1, 1)
    y_one_hot = y.new_zeros((y.size()[0], num_classes)).scatter_(1, y, 1)
    return y_one_hot


class CarliniWagnerLoss(nn.Module):
    def __init__(self):
        super(CarliniWagnerLoss, self).__init__()

    def forward(self, input, target):
        num_classes = input.size(1)
        label_mask = to_one_hot(target, num_classes=num_classes).float()
        correct_logit = torch.sum(label_mask * input, dim=1)
        wrong_logit = torch.max((1. - label_mask) * input, dim=1)[0]
        loss = -F.relu(correct_logit - wrong_logit + 50.).sum()
        return loss


def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    return (
        batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()


def _batch_clamp_tensor_by_vector(vector, batch_tensor):
    return torch.min(
        torch.max(batch_tensor.transpose(0, -1), -vector), vector
    ).transpose(0, -1).contiguous()


def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def batch_clamp(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_clamp_tensor_by_vector(float_or_vector, tensor)
        return tensor
    elif isinstance(float_or_vector, float):
        tensor = clamp(tensor, -float_or_vector, float_or_vector)
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def _get_norm_batch(x, p):
    batch_size = x.size(0)
    return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)


def _thresh_by_magnitude(theta, x):
    return torch.relu(torch.abs(x) - theta) * x.sign()


def batch_l1_proj_flat(x, z=1):
    v = torch.abs(x)
    v = v.sum(dim=1)

    indexes_b = torch.nonzero(v > z).view(-1)
    if isinstance(z, torch.Tensor):
        z = z[indexes_b][:, None]
    x_b = x[indexes_b]
    batch_size_b = x_b.size(0)

    if batch_size_b == 0:
        return x

    view = x_b
    view_size = view.size(1)
    mu = view.abs().sort(1, descending=True)[0]
    vv = torch.arange(view_size).float().to(x.device)
    st = (mu.cumsum(1) - z) / (vv + 1)
    u = (mu - st) > 0
    if u.dtype.__str__() == "torch.bool":  # after and including torch 1.2
        rho = (~u).cumsum(dim=1).eq(0).sum(1) - 1
    else:
        rho = (1 - u).cumsum(dim=1).eq(0).sum(1) - 1
    theta = st.gather(1, rho.unsqueeze(1))
    proj_x_b = _thresh_by_magnitude(theta, x_b)

    proj_x = x.detach().clone()
    proj_x[indexes_b] = proj_x_b
    return proj_x


def batch_l1_proj(x, eps):
    batch_size = x.size(0)
    view = x.view(batch_size, -1)
    proj_flat = batch_l1_proj_flat(view, z=eps)
    return proj_flat.view_as(x)


def clamp_by_pnorm(x, p, r):
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    if isinstance(r, torch.Tensor):
        assert norm.size() == r.size()
    else:
        assert isinstance(r, float)
    factor = torch.min(r / norm, torch.ones_like(norm))
    return batch_multiply(factor, x)


def is_float_or_torch_tensor(x):
    return isinstance(x, torch.Tensor) or isinstance(x, float)


def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return batch_multiply(1. / norm, x)


def jacobian(model, x, output_class):
    xvar = replicate_input_withgrad(x)
    scores = model(xvar)

    torch.sum(scores[:, output_class]).backward()

    return xvar.grad.detach().clone()


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def normalize_fn(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


def batch_per_image_standardization(imgs):
    assert imgs.ndimension() == 4
    mean = imgs.view(imgs.shape[0], -1).mean(dim=1).view(
        imgs.shape[0], 1, 1, 1)
    return (imgs - mean) / batch_adjusted_stddev(imgs)


def batch_adjusted_stddev(imgs):
    std = imgs.view(imgs.shape[0], -1).std(dim=1).view(imgs.shape[0], 1, 1, 1)
    std_min = 1. / imgs.new_tensor(imgs.shape[1:]).prod().float().sqrt()
    return torch.max(std, std_min)


class PerImageStandardize(nn.Module):
    def __init__(self):
        super(PerImageStandardize, self).__init__()

    def forward(self, tensor):
        return batch_per_image_standardization(tensor)


def predict_from_logits(logits, dim=1):
    return logits.max(dim=dim, keepdim=False)[1]


def get_accuracy(pred, target):
    return pred.eq(target).float().mean().item()


def set_torch_deterministic():
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def set_seed(seed=None):
    import torch
    import numpy as np
    import random
    if seed is not None:
        torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
>>>>>>> 4b57aae98d230f0282c345e6775888feda641368
