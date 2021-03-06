# Copyright (c) 2020 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""\
colorectal polyp classification (Adenoma patches) training example.
base: dhelth/pylibs-toolkit 0.12.2
"""

import argparse
import os

import numpy as np
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import numpy as np
from models import resnet18, Resnet18_onnx, Resnet50_onnx
import utils
import pandas as pd
import yaml
import time
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_recall_fscore_support

MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")

def main(args):

    if args.wandb:
        import wandb
        wandb.login()

    np.random.seed(args.seed)
    ecvl.AugmentationParam.SetSeed(args.seed)

    if args.checkpoints:
        os.makedirs(args.checkpoints, exist_ok=True)

    #Adenoma classifier: Not_adenoma, Tubular adenoma, Tubulo-villous adenoma
    num_classes = 3
    size = [args.input_size, args.input_size]  # size of the images
    
    # ECVL works in BGR
    mean = utils.adenoma_mean[[2,1,0]]
    std = utils.adenoma_mean[[2,1,0]]

    in_ = eddl.Input([3, size[0], size[1]])

    if args.pretrain == -1:
        out = resnet18(in_, num_classes)
        net = eddl.Model([in_], [out])
        args.pretrain = 18
    elif args.pretrain == 50:
        net, out = Resnet50_onnx(num_classes, pretreined=True)
    elif args.pretrain == 18:
        net, out = Resnet18_onnx(num_classes, pretreined=True)
    elif args.pretrain == 51:
        net, out = Resnet50_onnx(num_classes, pretreined=False)
    else: 
        net, out = Resnet18_onnx(num_classes, pretreined=False)
   
    if args.wandb:
        config = dict (
            learning_rate = args.lr,
            momentum = 0.99,
            batch_size = args.batch_size,
            weight_decay = args.weight_decay,
            architecture = "resnet{}".format(args.pretrain),
            dataset_id = "unitopatho_7000_224",
            infra = 'HPC4AI',
        )

        run = wandb.init(
           name=args.name,
           project="uc2-eddl",
           notes="uc2-eddl",
           tags=["adenoma_classifier","unitopatho","7000","224"],
           config=config,
        )
        
    eddl.build(
        net,
        eddl.sgd(args.lr, args.momentum, weight_decay=args.weight_decay),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU(args.gpu,args.lsb,mem=args.mem) if args.gpu else eddl.CS_CPU(),
        init_weights = args.pretrain == -1
    )
    currentlr = args.lr
    eddl.summary(net)
    eddl.setlogfile(net, "colorectal_polyp_classification")

    # note: 1812 pixel original resolution need progressive rescaling to avoid artifacts

    training_augs = ecvl.SequentialAugmentationContainer([
        #ecvl.AugResizeDim([906,906]), #uncomment if you train at original size (800 micron)
        #ecvl.AugResizeDim([453,453]), #uncomment if you train at original size (800 micron)
        #ecvl.AugResizeDim(size),
        ecvl.AugMirror(.5),
        ecvl.AugFlip(.5),
        ecvl.AugRotate([-90, 90]),
        ecvl.AugBrightness([30, 60]),
        ecvl.AugGammaContrast([0.5, 2]),
        ecvl.AugToFloat32(),
        ecvl.AugNormalize(mean, std)
    ])
    test_augs = ecvl.SequentialAugmentationContainer([
        #ecvl.AugResizeDim([906,906]), #uncomment if you train at original size (800 micron)
        #ecvl.AugResizeDim([453,453]), #uncomment if you train at original size (800 micron)
        #ecvl.AugResizeDim(size),
        ecvl.AugToFloat32(),
        ecvl.AugNormalize(mean, std)
    ])

    print("Reading dataset")  

    ds_file = os.path.join(args.in_ds,args.yml_name)
    if not os.path.isfile(ds_file):
        raise Exception('missing Dataset yaml file {}'.format(ds_file))
        exit()

    # Dataloader arguments [training,validation,test] 
    augs = [training_augs,test_augs,test_augs]
    drop_last = [True,False,False]

    # this yml describes splits in [test,training,validation] order
    yml_order = [2,0,1]
    augs = [augs[i] for i in yml_order]
    drop_last = [drop_last[i] for i in yml_order]

    dataset_augs = ecvl.DatasetAugmentations(augs)
    d = ecvl.DLDataset(ds_file, args.batch_size, dataset_augs, drop_last = drop_last , num_workers = len(args.gpu) , queue_ratio_size = 4*len(args.gpu) )

    print('Classes: ',d.classes_)

    d.SetSplit(ecvl.SplitType.training)

    num_samples_train = len(d.GetSplit())
    num_batches_train = d.GetNumBatches(ecvl.SplitType.training)

    print('Train samples:',num_samples_train)
    print('Train batches:',num_batches_train)

    #d.SetSplit(ecvl.SplitType.validation)
    d.SetSplit(ecvl.SplitType.test)
    num_samples_val = len(d.GetSplit())

    num_batches_val = d.GetNumBatches()

    print('Validation samples:',num_samples_val)
    best_val = -1

    # restore model
    if os.path.exists(args.ckpts):
        print('Restore model: ', args.ckpts)
        eddl.load(net, args.ckpts, "bin")

    print("Starting training")
    start_time = time.time()
    for e in range(args.epochs):
        print("Epoch {:d}/{:d} - Training".format(e + 1, args.epochs), flush=True)
        eddl.set_mode(net, 1)
        eddl.reset_loss(net)
        d.SetSplit(ecvl.SplitType.training)

        # dataset shuffling        
        d.ResetAllBatches(shuffle=True)
        
        d.Start()
        train_loss, train_acc = 0.0 , 0.0

        start_e_time = time.time()
        for b in range(num_batches_train):
            print("Epoch {:d}/{:d} (batch {:d}/{:d}) - ".format(
                e + 1, args.epochs, b + 1, num_batches_train
            ), end="", flush=True)
            
            _,x,y = d.GetBatch()

            tx, ty = [x], [y]
            eddl.train_batch(net, tx, ty)
            # These .get_* methods return a batch mean
            train_loss += eddl.get_losses(net)[0]
            train_acc  += eddl.get_metrics(net)[0]

            print()
            
        d.Stop()
        tttse = (time.time() - start_e_time)        
        train_loss = train_loss / num_batches_train
        train_acc  = train_acc / num_batches_train
        print("---Time to Train - Single Epoch: %s seconds ---" % tttse)
        print("---Train Loss:\t\t%s ---" % train_loss)
        print("---Train Accuracy:\t%s ---" % train_acc)

        # Scheduler
        #if (e+1) % 100 == 0:
        #    currentlr *= 0.1
        #    eddl.setlr(net, [currentlr, args.momentum, args.weight_decay])

        if (e+1) % args.val_epochs == 0:

            print("Epoch %d/%d - Evaluation" % (e + 1, args.epochs), flush=True)
            val_loss, val_acc = 0.0 , 0.0
            eddl.set_mode(net, 0)
            eddl.reset_loss(net)
            
            #d.SetSplit(ecvl.SplitType.validation)
            d.SetSplit(ecvl.SplitType.test)
            
            d.ResetAllBatches(shuffle=False)
            d.Start()

            metric = eddl.getMetric("categorical_accuracy")
            error = eddl.getLoss("soft_cross_entropy")
            np_out = None

            for b in range(num_batches_val):

                print("Epoch {:d}/{:d} (batch {:d}/{:d}) - ".format(
                e + 1, args.epochs, b + 1, num_batches_val
                ))


                samples,x,y = d.GetBatch()

                eddl.forward(net, [x])

                output = eddl.getOutput(out)

                # These .value return a sum
                val_acc += metric.value(y, output)
                val_loss += error.value(y, output)

                # output accumulation
                if np_out is None:
                   np_out = np.argmax(output.getdata(),axis=1)
                   np_y = np.argmax(y.getdata(),axis=1)
                else:
                   np_out, np_y = np.concatenate((np_out, np.argmax(output.getdata(),axis=1)), axis=0),  np.concatenate((np_y, np.argmax(y.getdata(),axis=1)), axis=0)

            d.Stop()
            val_loss = val_loss / (num_samples_val) 
            val_acc = val_acc / (num_samples_val) 

            # compute score weights because of class unbalance in the test/validation set
            scores = utils.comp_stats( confusion_matrix(np_y,np_out))

            weights = 1. / np.unique(np_y, return_counts=True)[1]
            weights = [ w / weights.sum() for w in weights] 

            ba_score = (scores['BA']*weights).sum()
            spec_score = (scores['TNR']*weights).sum()
            precision_score, sens_score, f1_score, _ =  precision_recall_fscore_support(np_y,np_out,average='weighted', zero_division = 0 )
 
            #save best model
            if best_val < ba_score:
                best_val = ba_score
                print("Saving weights")
                eddl.save( net, os.path.join(args.checkpoints, "{}.cpc_epoch_{}_val_{:.2f}.bin".format(args.name,e+1,ba_score)), "bin" )

            print("F1 Score:\t", f1_score)
            print("Recall/Sensitivity Score:\t", sens_score)
            print("Specificity Score:\t", spec_score)
            print("Precision Score:\t", precision_score)
            print("Categorical accuracy:\t", val_acc)
            print("---Validation Loss:\t%s ---" % val_loss)
            print("---Validation Balanced Accuracy:\t%s ---" % ba_score)
            print("---Validation Best Balanced Accuracy:\t%s ---" % best_val)

            if args.wandb:
                run.log({"train_time_epoch": tttse, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc })
        else:
            if args.wandb:
                run.log({"train_time_epoch": tttse, "train_loss": train_loss, "train_acc": train_acc, "val_loss": None, "val_acc": None})

    print("Saving weights")
    eddl.save( net, os.path.join(args.checkpoints, "{}.cpc_epoch_{}_val_{:.2f}.bin".format(args.name,args.epochs,val_acc)), "bin" )
    print("---Time to Train: %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_ds", help = 'path to the dataset',metavar="INPUT_DATASET")
    parser.add_argument("--epochs", help='number of training epochs', type=int, metavar="INT", default=100)
    parser.add_argument("--batch-size", help='batch-size', type=int, metavar="INT", default=120)
    parser.add_argument("--lr", help='learning rate' ,type=float, default=0.0001)
    parser.add_argument("--momentum", help='momentum' ,type=float, default=0.99)
    parser.add_argument("--weight-decay", help='weight-decay' ,type=float, default=5e-4)
    parser.add_argument("--val_epochs", help='validation set inference each (default=1) epochs', type=int, metavar="INT", default=1)
    parser.add_argument('--gpu', nargs='+', type=int, required=False, help='`--gpu 1 1` to use two GPUs')
    parser.add_argument("--checkpoints", metavar="DIR", default='checkpoints', help="if set, save checkpoints in this directory")
    parser.add_argument("--name", help='run name', type=str,  default='uc2_train')
    parser.add_argument("--pretrain", help='use pretrained resnet network: default=18, allows 50 and -1 (resnet 18 not pretrained)', type=int,  default=18)
    parser.add_argument("--input-size", type=int, help='224 px or original size', default=224)
    parser.add_argument("--seed", help='training seed', type=int, default=1990)
    parser.add_argument("--yml-name", help='yml name (default=deephealth-uc2-7000_balanced_adenoma.yml )', type=str, default='deephealth-uc2-7000_balanced_adenoma.yml')
    parser.add_argument("--ckpts", help='resume trining from a checkpoint', metavar='RESUME_PATH', default='')
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES), choices=MEM_CHOICES, default="full_mem")
    parser.add_argument("--lsb", help='multi-gpu update frequency', type=int, metavar="INT", default=1)
    parser.add_argument('--wandb', action='store_true', help='enable wandb logs', default=False)
    main(parser.parse_args())
