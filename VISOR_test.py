# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:33:18 2020

@author: marsh26macro
"""

import argparse, os
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from VISORNet import VISORNet
from DATA_READ import ReadIQAFolder
import torchvision.transforms as transforms
import ResultEvaluate as RE
from Utilizes import *
import numpy as np

# Testing settings
parser = argparse.ArgumentParser(description="PyTorch Regression")
parser.add_argument("--cuda", action="store_false", help="use cuda?")
parser.add_argument("--database", default='live', type=str, help="momentum")
parser.add_argument("--gpuid", default='0', type=str, help="id of GPU")
parser.add_argument("--nexp", default=0, type=int, help="number of the experiment")


# @cal_time
def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuid

    local_dir = '/mnt/hdd1/zzh/IQAdataset/'
    spaq_dir = '/mnt/hdd1/zzh/IQAdataset/'

    matpath = {
        'live': ['./data/LIVEinfo.mat', local_dir + 'LIVE/databaserelease2', 100],
        'csiq': ['./data/CSIQinfo.mat', local_dir + 'CSIQ', 1],
        'tid2013': ['./data/TID2013info.mat', local_dir + 'TID2013', 10],
        'livec': ['./data/LIVECinfo_new.mat', local_dir + 'LIVEC/ChallengeDB_release', 100],
        'kadid10k': ['./data/KADID10Kinfo.mat', local_dir + 'KADID10k/kadid10k/kadid10k', 10],
        'koniq10k': ['./data/KonIQ10Kinfo.mat', local_dir + 'KonIQ10K', 10],
        'spaq': ['./data/SPAQinfo.mat', spaq_dir + 'SPAQ/SPAQ zip', 100],
        'flive': ['./data/LIVEFBinfo.mat', local_dir + 'FLIVE/database', 100],
    }


    eval_img_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = 19980720
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    test_set = ReadIQAFolder(matpath=matpath[opt.database], nexp=opt.nexp, aug=False, random_crop=False,
                             img_transform=eval_img_tf, status='eval')
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    valid_set = ReadIQAFolder(matpath=matpath[opt.database], nexp=opt.nexp, aug=False, random_crop=False,
                             img_transform=eval_img_tf, status='valid')
    valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    print("===> Building model and Setting GPU")

    model = VISORNet(clen=16).cuda()
    checkpoint = torch.load('./checkpoint_finetune/fmodel_best_live_exp0.pth')
    model.load_state_dict(checkpoint["QAmodel"].state_dict())

    criterion = nn.MSELoss()

    print("===> Testing")
    best_res = test(valid_loader, model, criterion, status='valid')
    best_res = test(test_loader, model, criterion, status='test')


@cal_time
def test(test_loader, model, criterion, status='test'):

    model.CEnc.eval()
    model.DEnc.eval()
    model.Reg.eval()

    out_box = np.zeros(shape=(len(test_loader)))
    mos_box = np.zeros(shape=(len(test_loader)))

    tmp_out = []
    tmp_mos = []

    avg_loss = 0
    with torch.no_grad():
        for iteration, load_data in enumerate(test_loader, 1):
            img, org, mos, ids = load_data
            mos = mos.type(torch.FloatTensor)
            if opt.cuda:
                img = img.cuda()
                mos = mos.cuda()
            score = model(img)
            loss = criterion(score.squeeze(1), mos)
            avg_loss += loss.item()

            tmp_out.append(score.cpu())
            tmp_mos.append(mos.cpu())

            if iteration % 1 == 0:
                out_box[iteration - 1] = np.mean(np.array(tmp_out))
                mos_box[iteration - 1] = np.mean(np.array(tmp_mos))
                tmp_out = []
                tmp_mos = []

    # print(out_box[:20])
    # print(mos_box[:20])

    srcc = RE.srocc(out_box, mos_box)
    # srcc, _ = spearmanr(out_box, mos_box)
    krcc = RE.kendallcc(out_box, mos_box)
    plcc = RE.pearsoncc(out_box, mos_box)
    # plcc, _ = pearsonr(out_box, mos_box)
    rmse = RE.rootMSE(out_box, mos_box)

    res = [srcc, krcc, plcc, rmse]

    if status == 'test':
        print("===> TEST Loss: {:.10f} SRCC: {:.4f} KRCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f}".format((avg_loss / iteration),
                                                                                                         srcc, krcc,
                                                                                                         plcc, rmse))
    else:
        print("===> VALID Loss: {:.10f} SRCC: {:.4f} KRCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f}".format((avg_loss / iteration),
                                                                                                             srcc, krcc,
                                                                                                             plcc, rmse))

    return res


def save_checkpoint(model, epoch):
    model_folder = "checkpoint_finetune/"
    model_out_path = model_folder + "fmodel_best.pth"
    state = {"epoch": epoch, "enc_dist": model, "reg_net": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()











