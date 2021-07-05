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
colorectal polyp classification inference for UniToPatho.
"""

import argparse

import numpy as np
import os
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor, DEV_CPU
import pandas as pd
import models
import time
from utils import comp_stats, gen_sub_crops_tensors, SLIDE_RES ,get_sub_crops_tensors, gen_sub_crops
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import time
from tqdm import tqdm
import utils
import wandb


HP, NORM, TAHG, TALG, TVAHG, TVALG = 0, 1, 2, 3, 4, 5
NOT_ADEN, TA, TVA = 0, 2, 3

def main(args):

    if args.wandb:
        config = dict (
            batch_size = args.batch_size,
            architecture = "resnet18",
            dataset_id = "unitopato",
            infra = 'HPC4AI',
        )

        run = wandb.init(
           name=args.name,
           project="uc2-eddl",
           notes="icip-inference-eddl",
           tags=["icip"],
           config=config,
        )

    args.use_temp = args.temp_folder is not None and args.temp_folder != ''
    start_time = time.time()
    

    """
    ############################################################################
    ######################### ADENOMA PREDICTION ###############################
    ############################################################################
    """
    
    print('\n-------------------- ADENOMA PREDICTION --------------------')
    start_time_adenoma = time.time()
    image_size = 224
    size = [image_size, image_size]
    adenoma_mean = utils.adenoma_mean[None,:,None,None]
    adenoma_std = utils.adenoma_std[None,:,None,None]

    ## Set Adenoma Type preditcion test set image transformations
    test_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size),
        # Normalization done manually on the Tensor
        #ecvl.AugToFloat32(),
        #ecvl.AugNormalize(utils.adenoma_mean[[2,1,0]], utils.adenoma_mean[[2,1,0]])
    ])

    dataset_augs = ecvl.DatasetAugmentations([None, None, test_augs])

    ## Loading Adenoma Type preditcion test set
    print("Reading dataset")
    ds_file = os.path.join(args.in_ds,'7000_224','deephealth-uc2-7000_224.yml')
    if not os.path.isfile(ds_file):
        raise Exception('missing Dataset yaml file')
    d = ecvl.DLDataset(ds_file, args.batch_size, dataset_augs)
    print('Classes: ',d.classes_)
    d.SetSplit(ecvl.SplitType.test)
    num_samples = len(d.GetSplit())
    num_batches = num_samples // args.batch_size

    ## Enseble 3 models predictions for Adenoma
    preds_adenoma, gt_adenoma = [],[]
    x = Tensor([args.batch_size, d.n_channels_, size[0], size[1]])
    y = Tensor([args.batch_size, len(d.classes_)])
    

    for classifier_idx in [0,1,2]:
        print('\n-----------------------------')
        print(f'Running classifier {classifier_idx+1} of 3...')
        # Loading model
        net, out = models.load_onnx_models(f'adenoma_classifier.3c.{classifier_idx}')
        eddl.build(
            net, eddl.sgd(0.001, 0.9),["soft_cross_entropy"], ["categorical_accuracy"],
            eddl.CS_GPU(args.gpu,"low_mem") if args.gpu else eddl.CS_CPU(), init_weights=False
        )
        # Normalization done manually on the Tensor
        adenoma_mean = Tensor.fromarray(np.full((args.batch_size, d.n_channels_, size[0], size[1]), adenoma_mean), dev=net.dev)
        adenoma_std = Tensor.fromarray(np.full((args.batch_size, d.n_channels_, size[0], size[1]), adenoma_std), dev=net.dev)

        eddl.set_mode(net, 0)
        d.ResetAllBatches()

        # inference loop
        print('Inference Adenoma Type')
        preds = []
        for s in tqdm(range(num_batches)):
            d.LoadBatch(x, y)

            # !RGB model: (note ECVL works in BGR, so we have to change the input to feed network trained on pytorch)
            x = Tensor.fromarray(x.getdata()[:,[2,1,0]], dev=net.dev)
            x.sub_(adenoma_mean)
            x.div_(adenoma_std)
 
            eddl.forward(net, [x])
            output = eddl.getOutput(out)
            preds.append(output.getdata())

        preds_adenoma.append(np.stack(preds,axis=0))
      
    ## ensemble results
    preds_adenoma_sfm = np.squeeze(np.mean(preds_adenoma, axis=0))
    preds_adenoma = np.argmax(preds_adenoma_sfm, axis=1)


    preds_adenoma[preds_adenoma == 2] = TVA
    preds_adenoma[preds_adenoma == 1] = TA
    preds_adenoma[preds_adenoma == 0] = NOT_ADEN

    adenomas = (preds_adenoma == TA) | (preds_adenoma == TVA)
    data = {
        'predicted_adenoma': adenomas.astype(np.int),
        'predicted_type': preds_adenoma
    }
    adenoma_df = pd.DataFrame(data)
    adenoma_df.to_csv('predictions_adenoma.csv', index=False)
    print("---Time to Inference Adenoma - UnitoPatho: %s seconds ---" % (time.time() - start_time_adenoma))
    
    
    """
    ############################################################################
    ########################### GRADE PREDICTION ###############################
    ############################################################################
    """
    
    print('\n-------------------- GRADE PREDICTION ----------------------')
    start_time_grade = time.time()
    
    image_size, centersize, cropsize = int( 800/ SLIDE_RES ), 2000, 800 # changes to these values make temp_folder obsolete
    size = [image_size, image_size]
    max_num = ((centersize//(cropsize//2))-1) ** 2 #step == 2
    grade_mean = utils.grade_mean[None,:,None,None]
    grade_std = utils.grade_std[None,:,None,None]

    ## Loading Grade Type preditcion test set
    print("Reading dataset")
    ds_file = os.path.join(args.in_ds,'7000','deephealth-uc2-7000.yml')
    if not os.path.isfile(ds_file):
        raise Exception('missing Dataset yaml file')
    d = ecvl.Dataset(ds_file)
    print('Classes: ',d.classes_)

    samples = [ s.location_[0] for s in np.array(d.samples_)[d.split_.test_] ]
    #samples = [ s for s in samples if '164-' in s]

    if args.use_temp:
        print('Create Temp folder')
        gen_sub_crops( samples , os.path.join(args.temp_folder,'grade'), size = cropsize, um = centersize )
    
    num_samples = len(samples)

    ## Enseble 3 models predictions for Grade Type
    preds_grade = []
    for classifier_idx in [0,1,2]: 
        print('\n-----------------------------')
        print(f'Running classifier {classifier_idx+1} of 3...')
        # Loading model
        net, out = models.load_onnx_models(f'grade_classifier.800.{classifier_idx}')
        eddl.build(
            net, eddl.sgd(0.001, 0.9),["soft_cross_entropy"], ["categorical_accuracy"],
            eddl.CS_GPU(args.gp,"low_mem") if args.gpu else eddl.CS_CPU(), init_weights=False
        )
        eddl.set_mode(net, 0)

        # Normalization done manually on the Tensor
        grade_mean = Tensor.fromarray(np.full((max_num, 3, size[0], size[1]), grade_mean), dev=net.dev)
        grade_std = Tensor.fromarray(np.full((max_num, 3, size[0], size[1]), grade_std), dev=net.dev)

        # inference 1 sample each pass: we need to process images full-resolution subcrops
        preds=[]
        print('Inference Grade Type')
        for j,sample in enumerate(tqdm(samples)):
            #load subcrops at 800 micron for each 7000 micron sample

            if args.use_temp:
                batchLoader = get_sub_crops_tensors( sample , os.path.join(args.temp_folder,'grade'), grade_mean, grade_std, batch_size = args.fullres_batch_size , px_size = image_size, gpu = net.dev)
            else:
                batchLoader = gen_sub_crops_tensors( sample, grade_mean, grade_std, batch_size = args.fullres_batch_size , size = cropsize, um = centersize, px_size = image_size, gpu = net.dev)

            #initialize prediction to fixed dimension to the max number of subcrops
            s_preds = np.empty((max_num,2))
            s_preds.fill(np.nan)
            
            for i,x in enumerate(batchLoader):

                eddl.forward(net, [x])
                output = eddl.getOutput(out)
                idx = i*args.fullres_batch_size
                s_preds[idx:idx+x.shape[0]] = output.getdata()

            preds.append(np.stack(s_preds,axis=0))

        preds_grade.append(np.stack(preds,axis=0))
        
    ## ensemble results 
    preds_grade_sfm = np.squeeze(np.mean(preds_grade, axis=0))

    ## takes note of HG predictions for each subcrop
    preds_grade_sfm = np.squeeze(preds_grade_sfm[:,:,1])

    grade_df = pd.DataFrame(np.squeeze(preds_grade_sfm))
    grade_df.to_csv('predictions_grade.csv', index=False)
    print("---Time to Inference Grade - UnitoPatho: %s seconds ---" % (time.time() - start_time_grade))

    """
    ############################################################################
    ############################# HP PREDICTION ################################
    ############################################################################
    """
    print('\n-------------------- HP PREDICTION -------------------------')
    start_time_hp = time.time()
    
    image_size, centersize, cropsize = int( 800/ SLIDE_RES ), 1500, 800 # changes to these values make temp_folder obsolete
    size = [image_size, image_size]
    max_num = ((centersize//(cropsize//2))-1) ** 2 #step == 2 
    HP_mean = utils.HP_mean[None,:,None,None]
    HP_std = utils.HP_std[None,:,None,None]

    ## Loading Hyperplastic preditcion test set
    print("Reading dataset")
    ds_file = os.path.join(args.in_ds,'7000','deephealth-uc2-7000.yml')
    if not os.path.isfile(ds_file):
        raise Exception('missing Dataset yaml file')
    d = ecvl.Dataset(ds_file)
    print('Classes: ',d.classes_)

    samples = [ s.location_[0] for s in np.array(d.samples_)[d.split_.test_] ]

    if args.use_temp:
        print('Create Temp folder')
        gen_sub_crops( samples , os.path.join(args.temp_folder,'hp'), size = cropsize, um = centersize )
    
    num_samples = len(samples)
        
    ## Enseble 3 models predictions for Hyperplastic
    preds_hp = []
    
    for classifier_idx in [0,1,2]: 
        print('\n-----------------------------')
        print(f'Running classifier {classifier_idx+1} of 3...')
        # Loading model
        net, out = models.load_onnx_models(f'hp_classifier.800.{classifier_idx}')
        eddl.build(
            net, eddl.sgd(0.001, 0.9),["soft_cross_entropy"], ["categorical_accuracy"],
            eddl.CS_GPU(args.gpu,"low_mem") if args.gpu else eddl.CS_CPU(), init_weights=False
        )
        eddl.set_mode(net, 0)
        # Normalization done manually on the Tensor
        HP_mean = Tensor.fromarray(np.full((max_num, 3, size[0], size[1]), HP_mean), dev=net.dev)
        HP_std = Tensor.fromarray(np.full((max_num, 3, size[0], size[1]), HP_std), dev=net.dev)

        # inference 1 sample each pass: we need to process images full-resolution subcrops
        preds=[]
        print('Inference Hyperplastic')
        for j,sample in enumerate(tqdm(samples)):

            #load subcrops at 800 micron for each 7000 micron sample
            if args.use_temp:
                batchLoader = get_sub_crops_tensors( sample , os.path.join(args.temp_folder,'hp'), HP_mean, HP_std, batch_size = args.fullres_batch_size , px_size = image_size, um = centersize, gpu = net.dev)
            else:
                batchLoader = gen_sub_crops_tensors( sample, HP_mean, HP_std, batch_size = args.fullres_batch_size , size = cropsize, um = centersize, px_size = image_size, gpu = net.dev)
            
            s_preds = []
            for i,x in enumerate(batchLoader):

                eddl.forward(net, [x])
                output = eddl.getOutput(out)
                s_preds.append(output.getdata())
            
            # sample mean prediction
            if len(s_preds) > 0 :
                s_preds = np.stack(s_preds, axis=0)
                s_preds = np.mean(s_preds, axis=1)
            else: #no tissue subcrops 
                s_preds = np.zeros((1,6))
            preds.append(s_preds)

        preds_hp.append(np.stack(preds,axis=0))
        
    ## ensemble results 
    preds_hp_sfm = np.squeeze(np.mean(preds_hp, axis=0))
    preds_hp = np.argmax(preds_hp_sfm, axis=1)

    hp_df = pd.DataFrame(preds_hp)
    hp_df.to_csv('predictions_hp.csv', index=False)
    print("---Time to Inference HP - UnitoPatho: %s seconds ---" % (time.time() - start_time_hp))
    print("---Time to Slide Inference - UnitoPatho: %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_ds", help = 'path to UnitoPatho', metavar="INPUT_DATASET", default='/mnt/data/unitopath-public')
    parser.add_argument("--batch-size", help="batch-size for 224x224 resolution images", type=int, metavar="INT", default=248)
    parser.add_argument("--fullres-batch-size", help="batch-size for full resolution images", type=int, metavar="INT", default=6)
    parser.add_argument('--gpu', nargs='+', type=int, required=False, help='`--gpu 1 1` to use two GPUs', default=1)
    parser.add_argument("--temp_folder", help="temporary folder for inference speedup (slow down the first run, high storage demand )", default='')
    parser.add_argument('--wandb', action='store_true', help='enable wandb logs', default=True)
    parser.add_argument("--name", help='run name', type=str,  default='icip_inference_eddl')
    main(parser.parse_args())
