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
colorectal polyp classification inference example.
"""

import argparse

import numpy as np
import os
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

from models import resnet18, Resnet18_onnx, Resnet50_onnx
import time
import utils
from sklearn.metrics import confusion_matrix
import time

def main(args):
    num_classes = 6
    size = [args.input_size, args.input_size]  # size of images
    
    # ECVL works in BGR
    mean = utils.adenoma_mean[[2,1,0]]
    std = utils.adenoma_mean[[2,1,0]]

    in_ = eddl.Input([3, size[0], size[1]])

    if args.pretrain == -1:
        out = resnet18(in_, num_classes)
        net = eddl.Model([in_], [out])
    elif args.pretrain == 50:
        net, out = Resnet50_onnx(pretreined=True)
    elif args.pretrain == 18:
        net, out = Resnet18_onnx(pretreined=True)
    else: 
        net, out = Resnet50_onnx(pretreined=False)

    eddl.build(
        net,
        eddl.sgd(0.001, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU(args.gpu) if args.gpu else eddl.CS_CPU(),
        init_weights = args.pretrain == -1
    )
    eddl.summary(net)

    test_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size),
        ecvl.AugToFloat32(),
        ecvl.AugNormalize(mean, std)
    ])
    dataset_augs = ecvl.DatasetAugmentations([None, None, test_augs])

    #already contains proper test set
    print("Reading dataset")
    ds_file = os.path.join(args.in_ds,args.yml_name)
    if not os.path.isfile(ds_file):
        raise Exception('missing Dataset yaml file')
    
    d = ecvl.DLDataset(ds_file, args.batch_size, dataset_augs)

    x = Tensor([args.batch_size, d.n_channels_, size[0], size[1]])
    y = Tensor([args.batch_size, len(d.classes_)])
    d.SetSplit(ecvl.SplitType.training)
    num_samples = len(d.GetSplit())

    d.SetSplit(ecvl.SplitType.test)
    num_samples = len(d.GetSplit())
    print(f'Testing {num_samples} images')
    num_batches = num_samples // args.batch_size
    eddl.set_mode(net, 0)

    if not os.path.exists(args.ckpts):
        raise RuntimeError('Checkpoint "{}" not found'.format(args.ckpts))
    eddl.load(net, args.ckpts, "bin")

    print("Testing")
    start_time = time.time()
    np_out,np_out = None, None
    for b in range(num_batches):

        print("Batch {:d}/{:d}".format(b + 1, num_batches))
        d.LoadBatch(x, y)
        eddl.forward(net, [x])
        output = eddl.getOutput(out)

        if np_out is None:
            np_out = np.argmax(output.getdata(),axis=1)
            np_y = np.argmax(y.getdata(),axis=1)
        else:
            np_out, np_y = np.concatenate((np_out, np.argmax(output.getdata(),axis=1)), axis=0),  np.concatenate((np_y, np.argmax(y.getdata(),axis=1)), axis=0)

    scores = utils.comp_stats( confusion_matrix(np_y,np_out) )
    acc_score = scores['ACC'].mean()
    ba_score = scores['BA'].mean()
    f1_score = scores['F1'].mean()
    sens_score = scores['TPR'].mean()
    spec_score = scores['TNR'].mean()
    precision_score = scores['PPV'].mean()

    print("F1 Score:\t", f1_score)
    print("Recall/Sensitivity Score:\t", sens_score)
    print("Specificity Score:\t", spec_score)
    print("Precision Score:\t", precision_score)
    print("Balanced accuracy:\t", ba_score)
    print("Categorical accuracy:\t", acc_score)


    print("---Time to Inference:\t%s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_ds", help = 'path to the dataset', metavar="INPUT_DATASET")
    parser.add_argument("--ckpts", help='checkpoint path', metavar='CHECKPOINTS_PATH', default='checkpoints/cpc_classification_checkpoint_epoch_4.bin')
    parser.add_argument("--batch-size", help='batch-size', type=int, metavar="INT", default=218)
    parser.add_argument('--gpu', nargs='+', type=int, required=False, help='`--gpu 1 1` to use two GPUs')
    parser.add_argument("--pretrain", help='use pretrained resnet network: default=18, allows 50 and -1 (resnet 18 not pretrained)', type=int,  default=18)
    parser.add_argument("--yml-name", help='yml name (default=deephealth-uc2-800_224_balanced.yml )', type=str, default='deephealth-uc2-800_224_balanced.yml')
    parser.add_argument("--input-size", type=int, help='224 px or original size (1812 at 800um)', default=224)
    main(parser.parse_args())
