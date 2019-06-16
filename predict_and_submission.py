#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 08:20:09 2019

@author: ai
"""
import os
#import torch.utils.data as data

from utils import process_test_data
import torch
import torch.utils.data
from data_loader import load_checkpoint,Test_loader
from parser import parse_args
from Effecnet import Effect_netI,params_dict
import utils
from transforms import (Scale,DualCompose,Normalize)
import numpy as np
import pandas as pd
import scipy.io as sio


def predict(args):

    labels = sio.loadmat(args.data_dir + '/devkit/cars_test_annos.mat')
    x = []
    y=[]
    for i in range(8041):
        x.append(np.transpose(np.array(labels['annotations']['fname']))[i][0][0])
        y.append(0)

    D={"iamge_name":x, "label":y}
    sub=pd.DataFrame.from_dict(D)
    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True
    
    testdir = os.path.join(args.data_dir, 'test')

    print("Loading training data")
    size = params_dict[args.model_name][2]
    test_transform = DualCompose([Scale(size=size),Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    dataset_test = Test_loader(testdir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=16,shuffle=False,
        num_workers=args.workers, pin_memory=True)
    model = Effect_netI(num_classes=196, num_channels=3, pretrained=True, model_name=args.model_name, device =device)
    model.to(device)
    

    #criterion = utils.softmax_cross_entropy_criterion

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    load_checkpoint(args.output_dir +'{}.pth.tar'.format(args.model_name), model, optimizer)
    
    model.eval()

    for (image, target, name) in test_loader:
        image = image.to(device)
        logit = model(image)
        logit = logit.cpu().detach().numpy()
        predict = utils.softmax(logit, axis=1)
        for i, (e, n) in enumerate(list(zip(predict, name))):

            sub.loc[sub['iamge_name'] == n, 'label'] =np.argmax(e)+1
        
    sub.to_csv(args.data_dir + '/submission.csv', index=False)

args = parse_args()
process_test_data(args)
predict(args)
    
