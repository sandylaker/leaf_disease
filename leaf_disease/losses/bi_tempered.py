import torch
from torch import Tensor
import torch.nn as nn
from .utils import weight_reduce_loss


def log_t(u, t):
    if t == 1.0:
        return u.log()
    else:
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    if t == 1:
        return u.exp()
    else:
        return (1.0 + (1.0-t)*u).relu().pow(1.0 / (1.0 - t))


def compute_normalization_fixed_point(pred, t, num_iters):

    """Returns the normalization value for each example (t > 1.0).

    Args:
      pred: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness).
      num_iters: Number of iterations to run the method.

    Return:
        A tensor of same shape as activation with the last dimension being 1.
    """
    mu, _ = torch.max(pred, -1, keepdim=True)
    init_norm_pred = pred - mu

    norm_pred = init_norm_pred

    for _ in range(num_iters):
        logt_partition = torch.sum(
                exp_t(norm_pred, t), -1, keepdim=True)
        norm_pred = init_norm_pred * \
                logt_partition.pow(1.0-t)

    logt_partition = torch.sum(
            exp_t(norm_pred, t), -1, keepdim=True)
    norm_const = - log_t(1.0 / logt_partition, t) + mu

    return norm_const


def compute_normalization_binary_search(pred, t, num_iters):

    """Returns the normalization value for each example (t < 1.0).

    Args:
      pred: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (< 1.0 for finite support).
      num_iters: Number of iterations to run the method.

    Return:
        A tensor of same rank as activation with the last dimension being 1.
    """

    mu, _ = torch.max(pred, -1, keepdim=True)
    norm_pred = pred - mu

    effective_dim = \
        torch.sum(
                (norm_pred > -1.0 / (1.0-t)).to(torch.int32),
            dim=-1, keepdim=True).to(pred.dtype)

    shape_partition = pred.shape[:-1] + (1,)
    lower = torch.zeros(shape_partition, dtype=pred.dtype, device=pred.device)
    upper = -log_t(1.0/effective_dim, t) * torch.ones_like(lower)

    for _ in range(num_iters):
        logt_partition = (upper + lower)/2.0
        sum_probs = torch.sum(
                exp_t(norm_pred - logt_partition, t),
                dim=-1, keepdim=True)
        update = (sum_probs < 1.0).type_as(pred)
        lower = torch.reshape(
                lower * update + (1.0-update) * logt_partition,
                shape_partition)
        upper = torch.reshape(
                upper * (1.0 - update) + update * logt_partition,
                shape_partition)

    logt_partition = (upper + lower)/2.0
    return logt_partition + mu


class ComputeNormalization(torch.autograd.Function):
    """
    Class implementing custom backward pass for compute_normalization. See compute_normalization.
    """

    @staticmethod
    def forward(ctx, pred, t, num_iters):
        if t < 1.0:
            norm_const = compute_normalization_binary_search(pred, t, num_iters)
        else:
            norm_const = compute_normalization_fixed_point(pred, t, num_iters)

        ctx.save_for_backward(pred, norm_const)
        ctx.t = t
        return norm_const

    @staticmethod
    def backward(ctx, grad_output):
        pred, norm_const = ctx.saved_tensors
        t = ctx.t
        norm_pred = pred - norm_const
        probabilities = exp_t(norm_pred, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output

        return grad_input, None, None


def compute_normalization(pred, t, num_iters=5):
    """Returns the normalization value for each example.
    Backward pass is implemented.

    Args:
      pred: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      num_iters: Number of iterations to run the method.

    Return:
        A tensor of same rank as activation with the last dimension being 1.
    """
    return ComputeNormalization.apply(pred, t, num_iters)


def tempered_softmax(pred, t, num_iters=5):
    """Tempered softmax function.

    Args:
      pred: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature > 1.0.
      num_iters: Number of iterations to run the method.

    Returns:
      A probabilities tensor.
    """
    if t == 1.0:
        return pred.softmax(dim=-1)

    norm_const = compute_normalization(pred, t, num_iters)
    return exp_t(pred - norm_const, t)


def bi_tempered_loss(pred: Tensor,
                     label: Tensor,
                     t1: float,
                     t2: float,
                     label_smooth_val: float = 0.0,
                     avg_smooth_val: float = 0.0,
                     num_iters: int = 5,
                     weight=None,
                     reduction='mean',
                     avg_factor=None) -> Tensor:
    one_hot = torch.zeros_like(pred)
    one_hot.fill_(avg_smooth_val)
    label = label.view(-1, 1)
    one_hot.scatter_(1, label, 1 - label_smooth_val + avg_smooth_val)

    prob = tempered_softmax(pred, t2, num_iters)
    loss = one_hot * log_t(one_hot + 1e-10, t1) - \
        one_hot * log_t(prob, t1) - \
        one_hot.pow(2.0 - t1) / (2.0 - t1) + \
        prob.pow(2.0 - t1) / (2.0 - t1)
    loss = loss.sum(-1)

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)
    return loss


class BiTemperedLoss(nn.Module):
    """Bi-Tempered Logistic Loss. If t1 = 1.0 and t2 = 1.0, this is equivalent
    to the cross entropy loss.

    Args:
        t1 (float): temperature 1 ( < 1.0 for boundedness).
        t2 (float): temperatur 2 ( > 1.0 for tail heaviness, < 1.0 for finite support).
        label_smooth_val (float): label smoothing parameter.
        num_iters (int): number of iterations to run the method.
        reduction (str): reduction method, can be one of 'mean', 'sum', 'none'.
        loss_weight (float): weight of the loss.
    """

    def __init__(self,
                 t1,
                 t2,
                 num_classes,
                 label_smooth_val=0.0,
                 num_iters=5,
                 reduction='mean',
                 loss_weight=1.0):
        super(BiTemperedLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.num_iters = num_iters
        self.label_smooth_val = label_smooth_val
        self.avg_smooth_val = label_smooth_val / num_classes
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.criterion = bi_tempered_loss

    def forward(self,
                pred,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.criterion(
            pred,
            label,
            self.t1,
            self.t2,
            self.label_smooth_val,
            self.avg_smooth_val,
            num_iters=self.num_iters,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return self.loss_weight * loss