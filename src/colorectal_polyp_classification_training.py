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
colorectal polyp classification training example.
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

def main(args):

    if args.wandb:
        import wandb

    np.random.seed(args.seed)
    ecvl.AugmentationParam.SetSeed(args.seed)

    if args.checkpoints:
        os.makedirs(args.checkpoints, exist_ok=True)

    num_classes = 6
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
        net, out = Resnet50_onnx(pretreined=True)
    elif args.pretrain == 18:
        net, out = Resnet18_onnx(pretreined=True)
    else: 
        net, out = Resnet50_onnx(pretreined=False)
   
    if args.wandb:
        config = dict (
            learning_rate = args.lr,
            momentum = 0.99,
            batch_size = args.batch_size,
            weight_decay = args.weight_decay,
            architecture = "resnet{}".format(args.pretrain),
            dataset_id = "annotated_800_224",
            infra = 'HPC4AI',
        )

        run = wandb.init(
           name=args.name,
           project="uc2-eddl",
           notes="uc2-eddl",
           tags=["annotated","800","224"],
           config=config,
        )
    
    eddl.build(
        net,
        eddl.sgd(args.lr, 0.99, weight_decay=args.weight_decay),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU(args.gpu) if args.gpu else eddl.CS_CPU(),
        init_weights = args.pretrain == -1
    )
    currentlr = args.lr
    eddl.summary(net)
    eddl.setlogfile(net, "colorectal_polyp_classification")

    # 1812 original resolution need progressive rescaling to avoid artifacts

    training_augs = ecvl.SequentialAugmentationContainer([
        #ecvl.AugResizeDim([906,906]), #uncomment if you train at original size
        #ecvl.AugResizeDim([453,453]), #uncomment if you train at original size
        ecvl.AugResizeDim(size),
        ecvl.AugMirror(.5),
        ecvl.AugFlip(.5),
        ecvl.AugRotate([-90, 90]),
        ecvl.AugBrightness([30, 60]),
        ecvl.AugGammaContrast([0.5, 1.5]),
        ecvl.AugCoarseDropout([0.01, 0.05], [0.05, 0.15], 0.5),
        ecvl.AugToFloat32(),
        ecvl.AugNormalize(mean, std)
    ])
    test_augs = ecvl.SequentialAugmentationContainer([
        #ecvl.AugResizeDim([906,906]), #uncomment if you train at original size
        #ecvl.AugResizeDim([453,453]), #uncomment if you train at original size
        ecvl.AugResizeDim(size),
        ecvl.AugToFloat32(),
        ecvl.AugNormalize(mean, std)
    ])
    dataset_augs = ecvl.DatasetAugmentations(
        [training_augs, test_augs, test_augs]
    )

    print("Reading dataset")  

    ds_file = os.path.join(args.in_ds,args.yml_name)
    if not os.path.isfile(ds_file):
        raise Exception('missing Dataset yaml file {}'.format(ds_file))
        exit()

    d = ecvl.DLDataset(ds_file, args.batch_size, dataset_augs)

    print('Classes: ',d.classes_)
    x = Tensor([args.batch_size, d.n_channels_, size[0], size[1]])
    y = Tensor([args.batch_size, len(d.classes_)])
    num_samples_train = len(d.GetSplit())
    num_batches_train = num_samples_train // args.batch_size
    print('Train samples:',num_samples_train)

    d.SetSplit(ecvl.SplitType.validation)
    num_samples_val = len(d.GetSplit())
    num_batches_val = num_samples_val // args.batch_size
    indices = list(range(args.batch_size))
    print('Validation samples:',num_samples_val)
    best_val = -1


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

        s = d.GetSplit()
        np.random.shuffle(s)
        d.split_.training_ = s
        d.ResetAllBatches()

        train_loss, train_acc = 0.0 , 0.0

        start_e_time = time.time()
        for b in range(num_batches_train):
            print("Epoch {:d}/{:d} (batch {:d}/{:d}) - ".format(
                e + 1, args.epochs, b + 1, num_batches_train
            ), end="", flush=True)
            
            d.LoadBatch(x, y)
            #x.sub_(mean)
            #x.div_(std)
            tx, ty = [x], [y]
            eddl.train_batch(net, tx, ty, indices)
            # ! This .get_ return a batch mean
            train_loss += eddl.get_losses(net)[0]
            train_acc  += eddl.get_metrics(net)[0]

            eddl.print_loss(net, b)
            print()
            
        tttse = (time.time() - start_e_time)        
        train_loss = train_loss / num_batches_train
        train_acc  = train_acc / num_batches_train
        print("---Time to Train - Single Epoch: %s seconds ---" % tttse)
        print("---Train Loss:\t\t%s ---" % train_loss)
        print("---Train Accuracy:\t%s ---" % train_acc)

        # Scheduler
        #if (e+1) % 100 == 0:
        #    currentlr *= 0.1
        #    eddl.setlr(net, [currentlr, 0.99, args.weight_decay])

        if (e+1) % args.val_epochs == 0:

            print("Epoch %d/%d - Evaluation" % (e + 1, args.epochs), flush=True)
            val_loss, val_acc = 0.0 , 0.0
            eddl.set_mode(net, 0)
            eddl.reset_loss(net)
            d.SetSplit(ecvl.SplitType.validation)

            s = d.GetSplit()
            np.random.shuffle(s)
            d.split_.validation_ = s
            d.ResetAllBatches()

            metric = eddl.getMetric("categorical_accuracy")
            error = eddl.getLoss("soft_cross_entropy")
            for b in range(num_batches_val):

                print("Epoch {:d}/{:d} (batch {:d}/{:d}) - ".format(
                e + 1, args.epochs, b + 1, num_batches_val
                ))

                d.LoadBatch(x, y)
                #x.sub_(mean)
                #x.div_(std)

                eddl.forward(net, [x])
                output = eddl.getOutput(out)
                # ! This .value return a sum
                val_acc += metric.value(output, y)
                val_loss += error.value(output, y)
                
                
            val_loss = val_loss / (num_batches_val * args.batch_size)
            val_acc  = val_acc / (num_batches_val * args.batch_size)

            if best_val < val_acc:
                best_val = val_acc
                print("Saving weights")
                eddl.save( net, os.path.join(args.checkpoints, "{}.cpc_epoch_{}_val_{:.2f}.bin".format(args.name,e+1,val_acc)), "bin" )

            print("---Validation Loss:\t%s ---" % val_loss)
            print("---Validation Accuracy:\t%s ---" % val_acc)
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
    parser.add_argument("--epochs", help='number of training epochs', type=int, metavar="INT", default=300)
    parser.add_argument("--batch-size", help='batch-size', type=int, metavar="INT", default=128)
    parser.add_argument("--lr", help='learning rate' ,type=float, default=0.0001)
    parser.add_argument("--weight-decay", help='weight-decay' ,type=float, default=0.0005)
    parser.add_argument("--val_epochs", help='validation set inference each (default=1) epochs', type=int, metavar="INT", default=1)
    parser.add_argument('--gpu', nargs='+', type=int, required=False, help='`--gpu 1 1` to use two GPUs')
    parser.add_argument("--checkpoints", metavar="DIR", default='checkpoints', help="if set, save checkpoints in this directory")
    parser.add_argument("--name", help='run name', type=str,  default='uc2_train')
    parser.add_argument("--pretrain", help='use pretrained resnet network: default=18, allows 50 and -1 (resnet 18 not pretrained)', type=int,  default=18)
    parser.add_argument("--input-size", type=int, help='224 px or original size (1812 at 800um)', default=224)
    parser.add_argument("--seed", help='training seed', type=int, default=1990)
    parser.add_argument("--yml-name", help='yml name (default=deephealth-uc2-800_224_balanced.yml )', type=str, default='deephealth-uc2-800_224_balanced.yml')
    parser.add_argument("--ckpts", help='resume trining from a checkpoint', metavar='RESUME_PATH', default='')
    parser.add_argument('--wandb', action='store_true', help='enable wandb logs', default=False)
    main(parser.parse_args())
