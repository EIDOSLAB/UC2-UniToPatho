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
  --weight-decay WEIGHT_DECAY
                        weight-decay
  --val_epochs INT      validation set inference each (default=1) epochs
  --gpu GPU [GPU ...]   `--gpu 1 1` to use two GPUs
  --checkpoints DIR     if set, save checkpoints in this directory
  --name NAME           run name
  --pretrain PRETRAIN   use pretrained resnet network: default=18, allows 50
                        and -1 (resnet 18 not pretrained)
  --input-size INPUT_SIZE
                        224 px or original size (1812 at 800um)
  --seed SEED           training seed
  --yml-name YML_NAME   yml name (default=deephealth-uc2-800_224_balanced.yml
                        )
  --ckpts RESUME_PATH   resume trining from a checkpoint
  --wandb               enable wandb logs
```

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
  --ckpts CHECKPOINTS_PATH
                        checkpoint path
  --batch-size INT      batch-size
  --gpu GPU [GPU ...]   `--gpu 1 1` to use two GPUs
  --pretrain PRETRAIN   use pretrained resnet network: default=18, allows 50
                        and -1 (resnet 18 not pretrained)
  --yml-name YML_NAME   yml name (default=deephealth-uc2-800_224_balanced.yml
                        )
  --input-size INPUT_SIZE
                        224 px or original size (1812 at 800um)
```

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
  --extension EXTENSION
                        output image format (default='png')
  --jobs JOBS           Number of parallel jobs
  --size SIZE           Crop size (in μm)
  --pxsize PXSIZE       Crop size (in px) ( fullres -1)
  --subset SUBSET       subset of the slide (text file descriptor, default
                        test_wsi.txt )
```

```
usage: gen_yaml.py [-h] --folder FOLDER [--trainlist TRAINLIST]
                   [--testlist TESTLIST] [--balance] [--val_set]
                   [--gen_800_224] [--seed SEED] [--bal_idx BAL_IDX]

optional arguments:
  -h, --help            show this help message and exit
  --folder FOLDER       Unitopatho folder
  --trainlist TRAINLIST
                        specific wsi set for train (default empty)
  --testlist TESTLIST   specific wsi set for test (default test_wsi.txt)
  --balance             balance test set
  --val_set             create validation set
  --gen_800_224         create a 224px version of 800micrometer dataset
  --seed SEED           seed for data balancing
  --bal_idx BAL_IDX     less represented class for dataset balancing (default
                        3 => 4th)
```

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
  --fullres-batch-size INT
                        batch-size for full resolution images
  --gpu GPU [GPU ...]   `--gpu 1 1` to use two GPUs
  --temp_folder TEMP_FOLDER
                        temporary folder for inference speedup (slow down the
                        first run, high storage demand )
```

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


