
import math
import numpy
import lpips
import torch
import time
import fast_soft_sort.pytorch_ops as ops
import scipy.io as scio

def DataShuffleBox(org_num=200, shuffle_rate=0.8, fix=False):

    seq = numpy.arange(org_num)
    if not fix:
        numpy.random.shuffle(seq)
    # else:
    #     seq = numpy.arange(org_num-1, -1, -1)
    seq_s = seq.tolist()
    train_num = numpy.round(org_num * (1-shuffle_rate))
    # train_num = numpy.floor(org_num * shuffle_rate)
    seq_eval = seq_s[0:int(train_num)]
    seq_train = seq_s[int(train_num):]

    # train_num = numpy.round(org_num * shuffle_rate)
    # D = scio.loadmat('/home/vista/Documents/ZhouZehong/VISOR_plus/version3/data/LIVECinfo.mat')
    # seq_s = D['index'][0][:]
    # seq_train = seq_s[0:int(train_num)]
    # seq_eval = seq_s[int(train_num):]

    return seq_eval, seq_train


def PerceptualLoss(img1, img2, cuda_flg=False):

    loss_fn = lpips.LPIPS(net='alex', spatial=True, verbose=False)
    if cuda_flg:
        loss_fn = loss_fn.cuda()

    loss_fn.requires_grad = False
    # with torch.no_grad():
    loss = loss_fn(img1, img2)
    # loss = loss_fn.forward(img1, img2)

    return loss.mean()


def cal_time(func):

    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print("%s running time: %s mins." % (func.__name__, (t2-t1) / 60))
        return result

    return wrapper


