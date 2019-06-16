#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:11:42 2019

@author: NGUYEN Hoang-Phuong
"""


import argparse
import os
#ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
#print("root_of_project: {}".format(ROOT_DIR))

def parse_args():
    """
    A method to parse up command line parameters. By default it learns on the Watts-Strogatz dataset.
    The default hyperparameters give good results without cross-validation.
    """
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
    print("root_of_project --:-- {}".format(ROOT_DIR))
    
    
    parser = argparse.ArgumentParser(description='PyTorch Dataset_car_stanford')
	
    parser.add_argument('--download-path', 
                        default=ROOT_DIR, help='path of the dataset which we downloaded in the first step')
    parser.add_argument('--data-dir', 
                        default=ROOT_DIR+"/data", help='path of the dataset which is processed')
    parser.add_argument('--model_name', default='efficientnet-b3', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default= 0.1, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--output-dir', default= ROOT_DIR+'/weight/', help='path of weights')
    parser.add_argument('--resume', default=False, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--tensorboard', default=True,
                        help='Use tensorboad')
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    
    args = parser.parse_args()
    
    
    return args 
