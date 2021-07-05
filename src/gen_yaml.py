import pandas as pd
import glob
import os
import re
import argparse
import numpy as np
import yaml
from sklearn.model_selection import GroupShuffleSplit,ShuffleSplit
from utils import gen_res_folder
        

def gen_df(basepath):
    data = dict(
        image_id=[],
        top_label_name=[],
        top_label=[],
        type_name=[],
        type=[],
        grade_name=[],
        grade=[],
        wsi=[],
        roi=[],
        mpp=[],
        x=[],
        y=[],
        w=[],
        h=[],
        path=[]
    )

    for path in glob.glob(os.path.join(basepath, '*/*.png')):

        
        filename = os.path.basename(path)
        image_id = filename
        data['image_id'].append(image_id)

        # Retrieve polyp tipe (norm, hp, ta, tva)
        type_name = 'NORM'
        if 'HP' in path:
            type_name = 'HP'
        elif 'TA' in path:
            type_name = 'TA'
        elif 'TVA' in path:
            type_name = 'TVA'

        type_mapping = {type: i for i, type in enumerate(np.sort(['NORM', 'HP', 'TA', 'TVA']))}
        type = type_mapping[type_name]
        data['type'].append(type)
        data['type_name'].append(type_name)

        # retrieve polyp grading (-1 if norm or hp, 0 if LG, 1 if HG)
        grade_name = ''
        if 'LG' in image_id:
            grade_name = 'LG'
        elif 'HG' in image_id:
            grade_name = 'HG'

        grade_mapping = {
            '': -1,
            'LG': 0,
            'HG': 1
        }
        grade = grade_mapping[grade_name]
        data['grade'].append(grade)
        data['grade_name'].append(grade_name)

        # retrieve 6-class label norm, hp, ta.lg, ta.hg, tva.hg, tva.lg
        top_labels_mapping = {topl: i for i, topl in enumerate(np.sort(['NORM', 'HP', 'TA.LG', 'TA.HG', 'TVA.LG', 'TVA.HG']))}
        top_label_name = f'{type_name}.{grade_name}' if grade_name != '' else type_name
        top_label = top_labels_mapping[top_label_name]
        data['top_label'].append(top_label)
        data['top_label_name'].append(top_label_name)
        data['path'].append(os.path.join(top_label_name, image_id))

        # retrieve wsi name
        wsi = image_id.split('.ndpi')[0]
        data['wsi'].append(wsi)

        # retrieve roi id
        roi = int(re.findall(r'reg[0-9]+', image_id)[0].replace('reg', ''))
        data['roi'].append(roi)

        # retrieve mpp value
        mpp = float(re.findall(r'mpp0.[0-9]+', image_id)[0].replace('mpp', ''))
        data['mpp'].append(mpp)

        # retrieve patch coordinates in original WSI
        x,y,w,h = [int(n) for n in re.findall(r"([0-9]+,[0-9]+,[0-9]+,[0-9]+)", image_id)[0].split(',')]
        data['x'].append(x)
        data['y'].append(y)
        data['w'].append(w)
        data['h'].append(h)
 
    df = pd.DataFrame(data)
    df = df.sort_values(by=['top_label_name'])
    return df


def df_to_DH_yaml(df, name, splits, target_name = 'top_label_name'):
    data = dict(
        name='',
        description='UC2 is an annotated dataset for a total of hematoxylin and eosin stained patches extracted from whole-slide images, meant for training deep neural networks for colorectal polyps classification and adenomas grading. The slides are acquired through a Hamamatsu Nanozoomer S210 scanner at 20x magnification (0.4415 um/px). Each slide belongs to a different patient and is annotated by expert pathologists, according to six classes as follows:\nNORM - Normal tissue;\nHP - Hyperplastic Polyp;\nTA.HG - Tubular Adenoma, High-Grade dysplasia;\nTA.LG - Tubular Adenoma, Low-Grade dysplasia;\nTVA.HG - Tubulo-Villous Adenoma, High-Grade dysplasia;\nTVA.LG - Tubulo-Villous Adenoma, Low-Grade dysplasia.\nHere, We provide a yaml description in according with the DeepHealth Toolkit Dataset format for each um/px resolution.',
        classes=[],
        images=[],
        split = []
    )
    data['name'] = name
    data['classes']= sorted(df[target_name].unique())

    data['split'] = splits

    df = df[['path',target_name]]
    data['images'] = [ { 'location': t[0], 'label':t[1] } for t in df.loc[range(len(df))].values ] 

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='Unitopatho folder', required=True)
    parser.add_argument('--trainlist', type=str, help='specific wsi set for train (default empty)', default='')
    parser.add_argument('--testlist', type=str, help='specific wsi set for test (default test_wsi.txt)', default='test_wsi.txt')
    parser.add_argument('--balance', action='store_true', help='balance test set', default=False)
    parser.add_argument('--val_set', action='store_true', help='create validation set', default=False)
    parser.add_argument('--gen_800_224', action='store_true', help='create a 224px version of 800micrometer dataset', default=False)
    parser.add_argument('--seed', help='seed for data balancing', type=int, default=42)
    parser.add_argument('--gen_HG_LG', action='store_true', help='create yml for hg lg only, 2 classes, for 800', default=False)
    parser.add_argument('--gen_adenoma', action='store_true', help='create yml for adenoma type only, 3 classes, for 7000', default=False)
    parser.add_argument('--bal_idx', type=int, help='less represented class index for dataset balancing (default 3)', default=3)

    args = parser.parse_args()
    list_dir = [os.path.join(args.folder, f) for f in os.listdir(args.folder) if os.path.isdir(os.path.join(args.folder, f)) ]

    if args.gen_800_224:
        list_dir.append(os.path.join(args.folder, '800_224'))

    for basepath in list_dir:
        #basepath = str(args.folder)
        print('Yaml Generation in', basepath )

        folder_name = basepath.split('/')[-1]

        if args.gen_800_224 and '800_224' in basepath and not os.path.isdir(basepath):
            print('Create 800_224 folder')
            basepath_800 = basepath.replace('800_224','800')
            df = gen_df(basepath_800)
            gen_res_folder(df,basepath_800,basepath,224)
        else:
            df = gen_df(basepath)

        print(f'=> Got {len(df)} images')
        if len(df) == 0:
            continue
     
        target_name = 'top_label_name'
        suffix = ''
        bal_idx = args.bal_idx
        if args.gen_HG_LG and ('800' == folder_name or '800_224' == folder_name):
            print('Create LG/HG yaml for 800 folder')
            df = df.loc[df['grade'].isin([0,1])].reset_index(drop=True)
            bal_idx = 0
            target_name = 'grade_name'
            suffix = '_grade'

        if args.gen_adenoma and ('7000' == folder_name or '7000_224' == folder_name):
            print('Create Adenoma Type yaml for 7000 folder')
            df['type'] = np.where(df['type']==1,0,df['type'])
            df['type'] = np.where(df['type']==2,1,df['type'])
            df['type'] = np.where(df['type']==3,2,df['type'])
            df['type_name'] = np.where(df['type_name'].isin(['HP','NORM']),'NOT_ADN',df['type_name'])
            #df = df.loc[df['grade'].isin([0,1])].reset_index(drop=True)
            bal_idx = 1
            target_name = 'type_name'
            suffix = '_adenoma'
       
        train_idx,val_idx,test_idx = [], [],[]

        ## Given testlist
        if args.testlist != '' and args.testlist is not None:
          with open(os.path.join(args.folder,args.testlist), 'r') as f:
            test_wsi = [l.strip() for l in f]
            test_df = df[df.wsi.isin(test_wsi)].copy()
            test_df['yml_idx'] = test_df.index.tolist()
            test_df.to_csv(os.path.join(basepath, 'test.csv'), index=False)
            test_idx = test_df.index.tolist()
            print('Test Images',len(test_idx))
            
            train_df = df[~df.wsi.isin(test_wsi)].copy()
            train_df['yml_idx'] = train_df.index.tolist()
            train_df.to_csv(os.path.join(basepath, 'train.csv'), index=False)
            train_idx = train_df.index.tolist()
            print('Train Images',len(train_idx))
            
        ## Given trainlist
        if args.trainlist != '' and args.trainlist is not None:
          
          with open(os.path.join(args.folder,args.trainlist), 'r') as f:
            train_wsi = [l.strip() for l in f]
            train_df = df[df.wsi.isin(train_wsi)].copy()
            train_idx = train_df.index.tolist()
            train_df['yml_idx'] = train_df.index.tolist()
            train_df.to_csv(os.path.join(basepath, 'train.csv'), index=False)
            print('Train Images',len(train_idx))
            if len(test_idx) == 0:
              test_df = df[~df.wsi.isin(train_wsi)].copy()
              test_df['yml_idx'] = test_df.index.tolist()
              test_df.to_csv(os.path.join(basepath, 'test.csv'), index=False)
              test_idx = test_df.index.tolist()
              print('Test Images',len(test_idx))
                
        df.to_csv(os.path.join(basepath, folder_name + '.csv'), index=False)

        ## Random split without testlist or trainlist
        if args.trainlist == '' and args.testlist == '':
          splitter = GroupShuffleSplit(test_size=0.1, random_state=args.seed, n_splits=1).split(X = df, y = df['top_label_name'], groups=df['wsi'])

          train_idx, test_idx = next(splitter)
          train_df = df.iloc[train_idx]
          test_df = df.iloc[test_idx]
          train_idx,test_idx = train_df.index.tolist(),test_df.index.tolist()
          train_df.to_csv(os.path.join(basepath, 'train.csv'), index=False)
          test_df.to_csv(os.path.join(basepath, 'test.csv'), index=False)
          print('Test Images',len(test_idx))
          print('Train Images',len(train_idx))
        
        
        ## generate validation set
        if args.val_set:
          splitter = GroupShuffleSplit(test_size=0.05, random_state=args.seed, n_splits=1).split(X = train_df, y = train_df['top_label_name'], groups=train_df['wsi'])
          train_idx, val_idx = next(splitter)
          val_df = train_df.iloc[val_idx]
          train_df = train_df.iloc[train_idx]

          train_idx,val_idx = train_df.index.tolist(),val_df.index.tolist()
          print('Validation Images (from train set)',len(val_idx))

        splits = dict(training = train_idx, validation = val_idx, test = test_idx)

        yml_name = 'deephealth-uc2-'+folder_name+suffix
        yaml_dict = df_to_DH_yaml(df, yml_name , splits, target_name) 
        

        with open( os.path.join(basepath, yml_name + '.yml'), 'w') as f:
          yaml.safe_dump(yaml_dict, f, default_flow_style=False)

        

        # Balance Training Set
        
        if args.balance:
          train_df.index.names = ['index_orig']
          train_df = train_df.reset_index()
          min_size = np.sort(train_df.groupby(target_name).count()['image_id'])[bal_idx]
          train_df = train_df.groupby(target_name).apply(lambda group: group.sample(min_size, replace=len(group) < min_size, random_state=args.seed)).reset_index(drop=True)
          train_idx = train_df['index_orig'].values.tolist()

          print('Train Images (balanced)',len(train_idx))
          splits = dict(training = train_idx, validation = val_idx, test = test_idx)

          yml_name = 'deephealth-uc2-'+folder_name+'_balanced'+suffix
          yaml_dict = df_to_DH_yaml(df, yml_name , splits, target_name)

          with open( os.path.join(basepath, yml_name + '.yml'), 'w') as f:
            yaml.safe_dump(yaml_dict, f, default_flow_style=False)
    

