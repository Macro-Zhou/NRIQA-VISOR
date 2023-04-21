# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:33:18 2020

@author: marsh26macro
"""

import argparse, os
import torch.backends.cudnn as cudnn
from VISORNet import VISORNet
import torchvision.transforms as transforms
import ResultEvaluate as RE
from Utilizes import *
import numpy as np
from PIL import Image

# Testing settings
parser = argparse.ArgumentParser(description="PyTorch Regression")
parser.add_argument("--cuda", action="store_false", help="use cuda?")
parser.add_argument("--gpuid", default='0', type=str, help="id of GPU")

# @cal_time
def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuid

    eval_img_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    img_info = [['img/img01.bmp', 26.14],
                ['img/img02.bmp', 56.02],
                ['img/img03.bmp', 68.05],
                ]

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = 19980720
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    model = VISORNet(clen=16).cuda()
    checkpoint = torch.load('./checkpoint_finetune/fmodel_best_live_exp0.pth')
    model.load_state_dict(checkpoint["QAmodel"].state_dict())

    model.eval()
    out_box = np.zeros(shape=(len(img_info)))
    mos_box = np.zeros(shape=(len(img_info)))
    with torch.no_grad():
        for iteration, data in enumerate(img_info):
            imgname, dmos = data
            with open(imgname, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                img = eval_img_tf(img)
                dmos = np.array(dmos).astype(np.float32)
                dmos = torch.Tensor(dmos)
                if opt.cuda:
                    img = img.cuda()
                    dmos = dmos.cuda()
                score = model(img.unsqueeze(0))
                out_box[iteration] = np.array(score.item())
                mos_box[iteration] = np.array(dmos.item())
                print("Test Image: {}, MOS/DMOS: {:.4f}, Predicted Score: {:.4f}".format(imgname, dmos.item(), score.item()))

        srcc = RE.srocc(out_box, mos_box)
        krcc = RE.kendallcc(out_box, mos_box)
        plcc = RE.pearsoncc(out_box, mos_box)
        rmse = RE.rootMSE(out_box, mos_box)

        print("===> Performance: SRCC: {:.4f} KRCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f}".format(srcc, krcc, plcc, rmse))


if __name__ == "__main__":
    main()











