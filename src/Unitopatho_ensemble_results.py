import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from utils import comp_stats
from functools import partial
from scipy import stats
from tqdm import tqdm
import pyecvl.ecvl as ecvl

HP, NORM, TAHG, TALG, TVAHG, TVALG = 0, 1, 2, 3, 4, 5
NOT_ADEN, TA, TVA = 0, 2, 3

def make_top_label(row):
    if row.predicted_hp == 1:
        row.predicted_top_label = HP
        row.predicted_type = HP

    elif row.predicted_adenoma == 1:
        if row.predicted_type == TA:
            row.predicted_top_label = TAHG

        elif row.predicted_type == TVA:
            row.predicted_top_label = TVAHG

        if row.predicted_grade == 0:
            row.predicted_top_label += 1

    else:
        row.predicted_top_label = NORM
        row.predicted_type = NORM

    return row

def infer_grade(threshold, row):
    values = row.values[~np.isnan(row.values)]
    total = len(values)
    if total == 0:
        return 0
    values = (values > 0.5).astype(np.int)
    hg = (values == 1).sum().astype(np.float)
    return 1 if hg/total > threshold else 0


def analyze(gt, adenoma_df, grade_df, hp_df, grade_threshold):
    df = adenoma_df.copy()

    df['predicted_hp'] = (hp_df.iloc[:, 0].values == 0).astype(np.int)

    get_grade = partial(infer_grade, grade_threshold)
    grade_predictions = grade_df.apply(get_grade, axis=1)
    df['predicted_grade'] = grade_predictions.values

    df['predicted_top_label'] = np.nan
    df = df.apply(make_top_label, axis=1)

    cm = confusion_matrix(gt, df.predicted_top_label.values)

    print(f'6-class cm:\n', cm / cm.sum(axis=1)[:, None])

    return cm

def main(config):

    ds_file = os.path.join(config.in_ds,'7000','deephealth-uc2-7000.yml')
    if not os.path.isfile(ds_file):
        raise Exception('missing Dataset yaml file')
    d = ecvl.Dataset(ds_file)

    samples = [ s.location_[0] for s in np.array(d.samples_)[d.split_.test_] ]
    gt = [ s.label_[0] for s in np.array(d.samples_)[d.split_.test_] ]

    adenoma_df = pd.read_csv('predictions_adenoma.csv')
    grade_df = pd.read_csv('predictions_grade.csv')
    hp_df = pd.read_csv('predictions_hp.csv')

    assert len(gt) == len(adenoma_df)
    assert len(gt) == len(grade_df)
    assert len(gt) == len(hp_df)


    cm = analyze(gt, adenoma_df, grade_df, hp_df, config.threshold)

    labels = np.sort(d.classes_).tolist()
    plt.figure(figsize=(5, 5))

    cmap = sns.light_palette("seagreen", as_cmap=True)
    cmap = sns.color_palette("Reds")

    sns.heatmap(cm / np.sum(cm, axis=1)[:, None], annot=True,
                fmt='.0%', cmap=cmap, cbar=False, vmin=0., vmax=1.,
                xticklabels=labels, yticklabels=labels)
    plt.tight_layout()
    plt.savefig('multires-6c.pdf', format='pdf', dpi=200)

    scores = comp_stats( cm )
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("in_ds", help = 'path to UnitoPatho' ,metavar="INPUT_DATASET")
    parser.add_argument('--threshold', help = 'threshold for high-grade dysplasia inference (default=0.2)' , type=float, default=0.2)
    config = parser.parse_args()

    main(config)
