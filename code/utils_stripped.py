import torch
import torch.distributed as dist
from model import DeepSpeech

# function to consolidate tensor on a single processor


def reduce_tensor(tensor, world_size, reduce_op_max=False):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.MAX if reduce_op_max is True else dist.reduce_op.SUM)  # Default to sum
    if not reduce_op_max:
        rt /= world_size
    return rt


# function to check CTC loss is valid
def check_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """

    # initialize values
    loss_valid = True
    error = ''

    # If the loss grows to infinity, return error message
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"

    # if there are NaN losses, return error message
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'

    # if loss is negative, return error message
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"

    # return T/F of loss validity and potential error message
    return loss_valid, error


# function to load model from pth file
def load_model(device, model_path, use_half):

    # use load_model method from DeepSpeech class
    model = DeepSpeech.load_model(model_path)

    # set model to eval
    model.eval()

    # put model on device (GPU/CPU)
    model = model.to(device)

    # if the model is using half-precision sampling, use the half method of the model to indicate so
    if use_half:
        model = model.half()

    # return the model
    return model


# function to un-parallelize the model
def remove_parallel_wrapper(model):
    """
    Return the model or extract the model out of the parallel wrapper
    :param model: The training model
    :return: The model without parallel wrapper
    """
    # Take care of distributed/data-parallel wrapper
    model_no_wrapper = model.module if hasattr(model, "module") else model
    return model_no_wrapper
