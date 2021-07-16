# UC2-UniToPatho

##### This repository contains the source code for getting started on [UniToPatho](https://ieee-dataport.org/open-access/unitopatho) dataset by using [pyEDDL](https://github.com/deephealthproject/pyeddl)/[pyECVL](https://github.com/deephealthproject/pyecvl)

##### KPI measurements: 
Preliminary KPI measurements are availabe at https://wandb.opendeephealth.di.unito.it/eidoslab/uc2-eddl/reports/DeepHealth-UC2--VmlldzoxNA?accessToken=gb26jguu33kr0pw1qvmee8pexg6z44aa1s7cj02cwqcdt6wv6mvy9lq6ahp9kquc

## Getting Ready

##### Step 1: Install dependencies
Install [pyEDDL](https://github.com/deephealthproject/pyeddl)/[pyECVL](https://github.com/deephealthproject/pyecvl)

and also other dependencies

`pip3 install pandas numpy opencv-python pyyaml scikit-learn matplotlib wandb tqdm openslide_python scikit-image gdown seaborn`


##### Step 2: Download the dataset
Download and extract [UniToPatho](https://ieee-dataport.org/open-access/unitopatho) 

##### Step 3: Run `gen_yaml.py`
This script will generate all files needed in the [DeepHealth Toolkit Dataset Format](https://github.com/deephealthproject/ecvl/wiki/DeepHealth-Toolkit-Dataset-Format)

Example:

`python3 gen_yaml.py --folder <dataset_path>/unitopath-public/ --val_set --balanced --gen_adenoma`

```
usage: gen_yaml.py [-h] --folder FOLDER [--trainlist TRAINLIST]
                   [--testlist TESTLIST] [--balance] [--val_set]
                   [--gen_800_224] [--seed SEED] [--bal_idx BAL_IDX]

optional arguments:
  -h, --help            show this help message and exit
  --folder FOLDER       Unitopatho folder
  --trainlist TRAINLIST specific wsi set for train (default empty)
  --testlist TESTLIST   specific wsi set for test (default test_wsi.txt)
  --balance             balance training set
  --val_set             create validation set
  --gen_800_224         create a 224px version of 800micrometer dataset (it takes some time)
  --gen_HG_LG           create yml for hg lg only, 2 classes, for 800
  --gen_adenoma         create yml for adenoma type only, 3 classes, for 7000
  --seed SEED           seed for data balancing
  --bal_idx BAL_IDX     less represented class index for dataset balancing (default 3)
```

##### Step 3_bis: (Optional) Run `model_pytorch2onnx.py`
Only if you want to generate 'onnx_models' directory by yourself.

## Inference Pipline proposed in [UniToPatho, a labeled histopathological dataset for colorectal polyps classification and adenoma dysplasia grading](https://arxiv.org/abs/2101.09991)

##### Step 1: Run Inference script `Unitopatho_ensemble_inference.py`
It will generate 3 different `.csv` files by follwing the inference pipeline in the paper.

Example:

`python3 -u Unitopatho_ensemble_inference.py <dataset_path>/unitopath-public/ --gpu 1 --temp_folder ''`

```
usage: Unitopatho_ensemble_inference.py [-h] [--batch-size INT]
                                        [--fullres-batch-size INT]
                                        [--gpu GPU [GPU ...]]
                                        [--temp_folder TEMP_FOLDER]
                                        INPUT_DATASET

colorectal polyp classification inference for UniToPatho.

positional arguments:
  INPUT_DATASET         path to UnitoPatho

optional arguments:
  -h, --help            show this help message and exit
  --batch-size INT      batch-size for 224x224 resolution images
  --fullres-batch-size  batch-size for full resolution images
  --gpu GPU [GPU ...]   `--gpu 1 1` to use two GPUs
  --temp_folder         temporary folder for inference speedup (slow down the first run, high storage demand ), default none
```

##### Step 2: Run Inference script `Unitopatho_ensemble_results.py`
It will generate the resulting confusion matrix image (`.pdf`) and the metric results from `.csv` files.

Example:

`python3 -u Unitopatho_ensemble_inference.py <dataset_path>/unitopath-public/`

```
usage: Unitopatho_ensemble_results.py [-h] [--threshold THRESHOLD]
                                      INPUT_DATASET

positional arguments:
  INPUT_DATASET         path to UnitoPatho

optional arguments:
  -h, --help            show this help message and exit
  --threshold THRESHOLD
                        threshold for high-grade dysplasia inference
                        (default=0.2)
```

## Train and test a ResNet model

##### Step 1: Train with `colorectal_polyp_classification_training.py`

Example:

`python3 -u colorectal_polyp_classification_training.py --gpu 1 --name 'my_first_run' --pretrain 18 <dataset_path>/unitopath-public/7000_224`

```
usage: colorectal_polyp_classification_training.py [-h] [--epochs INT]
                                                   [--batch-size INT]
                                                   [--lr LR]
                                                   [--weight-decay WEIGHT_DECAY]
                                                   [--val_epochs INT]
                                                   [--gpu GPU [GPU ...]]
                                                   [--checkpoints DIR]
                                                   [--name NAME]
                                                   [--pretrain PRETRAIN]
                                                   [--input-size INPUT_SIZE]
                                                   [--seed SEED]
                                                   [--yml-name YML_NAME]
                                                   [--ckpts RESUME_PATH]
                                                   [--wandb]
                                                   INPUT_DATASET

colorectal polyp classification training example.

positional arguments:
  INPUT_DATASET         path to the dataset

optional arguments:
  -h, --help            show this help message and exit
  --epochs INT          number of training epochs
  --batch-size INT      batch-size
  --lr LR               learning rate
  --weight-decay        weight-decay
  --val_epochs INT      validation set inference each (default=1) epochs
  --gpu GPU [GPU ...]   `--gpu 1 1` to use two GPUs
  --checkpoints DIR     if set, save checkpoints in this directory
  --name NAME           run name
  --pretrain PRETRAIN   use pretrained resnet network: default=18, allows 50 and -1 (resnet 18 not pretrained)
  --input-size          224 px or original size 
  --seed SEED           training seed
  --yml-name YML_NAME   yml name (default=deephealth-uc2-7000_balanced_adenoma.yml )
  --ckpts RESUME_PATH   resume trining from a checkpoint
  --wandb               enable wandb logs
  --mem                 allows full_mem, mid_mem, low_mem
```

##### Step 2: Train with `colorectal_polyp_classification_inference.py`

Example:

`python3 -u colorectal_polyp_classification_inference.py --gpu 1 --ckpts checkpoints/<checkpoint_name>.bin --pretrain 18 <dataset_path>/unitopath-public/7000_224`

```
usage: colorectal_polyp_classification_inference.py [-h]
                                                    [--ckpts CHECKPOINTS_PATH]
                                                    [--batch-size INT]
                                                    [--gpu GPU [GPU ...]]
                                                    [--pretrain PRETRAIN]
                                                    [--yml-name YML_NAME]
                                                    [--input-size INPUT_SIZE]
                                                    INPUT_DATASET

colorectal polyp classification inference example.

positional arguments:
  INPUT_DATASET         path to the dataset

optional arguments:
  -h, --help            show this help message and exit
  --ckpts               checkpoint path
  --batch-size INT      batch-size
  --gpu GPU [GPU ...]   `--gpu 1 1` to use two GPUs
  --pretrain PRETRAIN   use pretrained resnet network: default=18, allows 50 and -1 (resnet 18 not pretrained)
  --yml-name YML_NAME   yml name (default=deephealth-uc2-7000_balanced_adenoma.yml )
  --input-size          224 px or original size 
```

## Generate your own dataset from `.ndpi` Whole Slides and `.ndpa` annotation files
Annotation files are associated to the slide only if they are named `<slide_name>.ndpi.ndpa`

Example:

`python3 -u cropify_ecvl.py --jobs 4 --size 4000 --pxsize 224 --extension png --ROIs <annotation_path> --ROIs  <slides_path> <output_path>`

Note: `--ROIs` argument is optional: if not provided, slides will be clipped where tissue is detected

```
usage: cropify_ecvl.py [-h] [--ROIs ROIS] [--extension EXTENSION]
                       [--jobs JOBS] [--size SIZE] [--pxsize PXSIZE]
                       [--subset SUBSET]
                       data output

positional arguments:
  data                  dataset path
  output                Output path

optional arguments:
  -h, --help            show this help message and exit
  --ROIs ROIS           Path for meatadata folder
  --extension           output image format (default='png')
  --jobs JOBS           Number of parallel jobs
  --size SIZE           Crop size (in μm)
  --pxsize PXSIZE       Crop size (in px) ( fullres -1)
  --subset SUBSET       subset of the slide (text file descriptor, default test_wsi.txt )
```






