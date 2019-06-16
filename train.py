from __future__ import print_function
import datetime
import os
import time
import sys
import torch
import torch.utils.data
from data_loader import Dataset_car_stanford,save_checkpoint,load_checkpoint
from parser import parse_args
from Effecnet import Effect_netI,params_dict
import utils
from logger import Logger
try:
    from apex import amp
except ImportError:
    amp = None
from learning_rate import CyclicLR
from transforms import (Scale,DualCompose,OneOf,Normalize,ShiftScale,
                        ShiftScaleRotate,VerticalFlip)

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch,  apex=True):
    model.train()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top3 = utils.AverageMeter()
    start_time = time.time()
    for batch_idx,(image, target) in enumerate(data_loader):
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)        
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        lr=optimizer.param_groups[0]["lr"]
        prec1, prec3 = utils.accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), image.size(0))
        top1.update(prec1[0], image.size(0))
        top3.update(prec3[0], image.size(0))
    print("learning_rate:", lr,"---time:",time.time() - start_time)


def evaluate(model, valid_loader, criterion,device):
    model.eval()
    losses = utils.AverageMeter() 
    top1 = utils.AverageMeter()
    top3 = utils.AverageMeter()

    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device),target.to(device)
            output = model(data)
            loss = criterion(output, target)
            prec1, prec3 = utils.accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1[0], data.size(0))
            top3.update(prec3[0], data.size(0))

    print(' *Val_ Loss ({loss.avg:.4f}) Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(loss=losses, top1=top1, top3=top3))
    return top3.avg, top1.avg, loss




def main(args):
    
        
    if args.apex:
        if sys.version_info < (3, 0):
            raise RuntimeError("Apex currently only supports Python 3. Aborting.")
        if amp is None:
            raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                               "to enable mixed-precision training.")
    if args.output_dir:
        utils.mkdir(args.output_dir)
    if args.tensorboard:
        utils.mkdir(os.path.join(args.output_dir,"logs"))
        logger = Logger(os.path.join(args.output_dir,"logs"))
    

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    print("Loading data")
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'valid')

    print("Loading training data")
    size = params_dict[args.model_name][2]
    print("resize image by{} to adapter model {}".format(size,args.model_name))
    ### transform ###
    valid_transform = DualCompose([Scale(size=size),Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_transform = DualCompose([VerticalFlip(prob=0.5),OneOf([ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30),
                                                            ShiftScale(limit=4,prob=0.5)#,
                                                            #do_horizontal_shear(limit=0.3)     
                                                                  ]),
#                        OneOf([Brightness_shift(limit=0.1),
#                               Brightness_multiply(limit=0.1),
#                               do_Gamma(limit=0.1),
#                               RandomContrast(limit=0.1),
#                               RandomSaturation(limit=0.1)],prob=0.5),   
                        Scale(size=size),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                        ])
    ## data loader ###
    dataset_train = Dataset_car_stanford(traindir, transform=train_transform, target_transform=None)
    dataset_test = Dataset_car_stanford(valdir, transform=valid_transform, target_transform=None)

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,shuffle=True,
        num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,shuffle=False,
         num_workers=args.workers, pin_memory=True)

    print("Creating model")
    model = Effect_netI(num_classes=196, num_channels=3, pretrained=True, model_name=args.model_name, device =device)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    criterion = utils.softmax_cross_entropy_criterion

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    if args.resume:
        load_checkpoint(args.output_dir +'{}.pth.tar'.format(args.model_name), model, optimizer)

    if args.test_only:
        evaluate(model,data_loader_test, criterion, device=device)
        return

    print("Start training")
    start_time = time.time()
    best_prec1=0
    ####################### FINE TUNE STEP 1 ################################
    print('----------FINETUNE BY  STEPLR--------------------')
    for epoch in range(args.start_epoch, args.epochs):
        print('epoch:',epoch)
        is_best=True
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch,  args.apex)
        lr_scheduler.step()
        top3_error,top1_error,loss_val = evaluate(model,data_loader_test, criterion,device=device)
        is_best = top1_error > best_prec1
        best_prec1 = max(top1_error, best_prec1)
        if (is_best):
            save_checkpoint({
                'model': model_without_ddp.state_dict(),
                'args': args,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename= args.output_dir +'{}.pth.tar'.format(args.model_name))

        
            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #
        if args.tensorboard:
 
            info = { 'loss_val': loss_val.item(),
                 'top3_error': top3_error.item(),
                 'top1_error': top1_error.item()}
            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch+1)
#        # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)

            
    ###################### FINETUNE STEP2 ##################################
    lr_scheduler = CyclicLR(optimizer,base_lr=5e-6, max_lr=5e-5,
                   step_size=1500, mode='triangular', gamma=1.,
                   scale_fn=None, scale_mode='cycle', last_batch_iteration=-1)
    print('----------FINETUNE BY  CyclicLR--------------------')
    for epoch in range( args.epochs, 2*args.epochs):
        print('epoch:',epoch)
        is_best=True
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch,  args.apex)
        lr_scheduler.step()
        top3_error,top1_error,loss_val = evaluate(model,data_loader_test, criterion,device=device)
        is_best = top1_error > best_prec1
        best_prec1 = max(top1_error, best_prec1)
        if (is_best):
            save_checkpoint({
                'model': model_without_ddp.state_dict(),
                'args': args,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename= args.output_dir +'{}.tar'.format(args.model_name))

    
            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #
        if args.tensorboard:
        # 1. Log scalar values (scalar summary)   
            info = { 'loss_val': loss_val.item(),
                 'top3_error': top3_error.item(),
                 'top1_error': top1_error.item()}
            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch+1)
#        # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    args = parse_args()
    main(args)
