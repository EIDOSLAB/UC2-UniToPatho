import numpy as np
from pyecvl import ecvl 
import os
from cropify_core_ecvl import is_histopath
import pandas as pd
from pyeddl.tensor import Tensor, DEV_CPU
from tqdm import tqdm
import cv2

IMG_EXT = 'png' 
SLIDE_RES = 0.4415

# !RGB (note ECVL works in BGR, so we have to change the input to feed network trained on pytorch)
adenoma_mean = np.array([0.8153967261314392, 0.7560872435569763, 0.7853971719741821])*255
adenoma_std = np.array([0.0768752247095108, 0.13408100605010986, 0.06851005554199219])*255
HP_mean, HP_std = np.array([0.485, 0.456, 0.406])*255, np.array([0.229, 0.224, 0.225])*255
grade_mean, grade_std = np.array([0.485, 0.456, 0.406])*255, np.array([0.229, 0.224, 0.225])*255



def gen_res_folder(df,folder_in,folder_out,pxsize):
   os.makedirs(folder_out, exist_ok = True)
   if folder_in == folder_out:
       print('Skip: same folder')
       return
   if len(os.listdir(folder_out)) == len(df):
       return
   print("Resizing {} images to {}".format(len(df),pxsize), end="", flush=True)
   for idx in tqdm(df.index):
       image_id,gt = df.iloc[idx]['image_id'],df.iloc[idx]['top_label_name']
       img = ecvl.ImRead(os.path.join(folder_in,gt,image_id))
       dim = img.dims_[1]

       ## avoid artifacts for high dimension images
       while int(dim * 0.5) > pxsize:
           ecvl.ResizeDim(img,img,[int(dim * 0.5), int(dim * 0.5)], ecvl.InterpolationType.lanczos4)
           dim = int(dim * 0.5)
       ecvl.ResizeDim(img,img,[pxsize,pxsize], ecvl.InterpolationType.lanczos4)
       os.makedirs(os.path.join(folder_out,gt), exist_ok = True)
       ecvl.ImWrite(os.path.join(folder_out,gt,image_id),img)


def gen_sub_crops( samples , temp_folder, size = 800, um = 2000 , step = 2):

    os.makedirs(temp_folder, exist_ok = True)

    if len(os.listdir(temp_folder)) == len(samples):
        print('Found temp image folder')
        return

    px_size = int(size / SLIDE_RES)
    c_size = int(um / SLIDE_RES) + 1
    px_step = px_size//step

    for sample in tqdm(samples):
        ss = sample.split('/')
        image_id = ss[-1] 
        patch = ecvl.ImRead(sample)
        ecvl.ChangeColorSpace(patch, patch, ecvl.ColorType.RGB)
        ecvl.CenterCrop(patch, patch, [c_size,c_size])
        image_folder = os.path.join(temp_folder,image_id.replace('.'+IMG_EXT,''))
        os.makedirs(image_folder, exist_ok = True)
        patch_data = np.array(patch , copy=False)

        for x in range(0, c_size-px_size, px_step):
            for y in range(0, c_size-px_size, px_step):
                sub_crop = patch_data[x:x+px_size,y:y+px_size,:]
                if is_histopath(sub_crop):
                    sub_crop = ecvl.Image.fromarray(sub_crop,'xyc',ecvl.ColorType.RGB)
                    ecvl.ImWrite(os.path.join(image_folder,'centercrop{}-x{}-y{}.{}'.format(um,x,y,IMG_EXT)),sub_crop)

    print('Temp image folder ready')



def get_sub_crops_tensors( image_id, temp_folder, mean_t,std_t, batch_size = 4 , px_size = 224, gpu = DEV_CPU):

    slide_name = image_id.replace('.'+IMG_EXT,'').split('/')[-1]
    fpath = os.path.join(temp_folder,slide_name)
    sub_crops = [c for c in os.listdir(fpath) if os.path.isfile(os.path.join(fpath,c)) and f'.{IMG_EXT}' in c]
    n_crops = len(sub_crops)
    sub_crops = ( c for c in sub_crops)

    while n_crops > 0:
        tens_size = min(batch_size,n_crops)
        n_crops -= tens_size

        x = np.zeros((tens_size, 3 , px_size, px_size))
        #y = np.zeros((tens_size, 1 )) + gt

        for i in range(tens_size):
            sub_crop = ecvl.ImRead(os.path.join(fpath,next(sub_crops)))
            ecvl.ChangeColorSpace(patch, patch, ecvl.ColorType.RGB)
            #Normalization done manually on gpu
            #ecvl.ConvertTo(sub_crop, sub_crop, ecvl.DataType.float32)
            #ecvl.Normalize(sub_crop,sub_crop,mean,std)
            sub_crop = np.array(sub_crop, copy=False)

            if sub_crop.shape[-1] == 3:
                sub_crop = np.moveaxis(sub_crop, [0,1,2], [2,1,0])
            x[i,:,:,:] = sub_crop

        mean = mean_t.select([f":{tens_size}"])
        std = std_t.select([f":{tens_size}"])

        x = Tensor.fromarray(x, dev=gpu)
        x.sub_(mean)
        x.div_(std)
        #y = Tensor.fromarray(y, dev=gpu)

        yield(x)#,y)


def gen_sub_crops_tensors( sample, mean_t,std_t, batch_size = 4 , px_size = 224, size = 800, um = 2000, step = 2, gpu = DEV_CPU):


    max_num = (um//size) ** 2

    px_size = int(size / SLIDE_RES)
    c_size = int(um / SLIDE_RES) + 1
    px_step = px_size//step

    hist = []
    patch = ecvl.ImRead(sample)

    ecvl.CenterCrop(patch, patch, [c_size,c_size])
    ## its BGR here!
    ecvl.ChangeColorSpace(patch, patch, ecvl.ColorType.RGB)
    patch_data = np.array(patch , copy=False)

    for x in range(0, c_size-px_size, px_step):
        for y in range(0, c_size-px_size, px_step):
            sub_crop = patch_data[x:x+px_size,y:y+px_size,:]

            if is_histopath(sub_crop):
                #Normalization done manually on gpu
                #sub_crop = ecvl.Image.fromarray(sub_crop,'xyc',ecvl.ColorType.RGB)
                #ecvl.ConvertTo(sub_crop, sub_crop, ecvl.DataType.float32)
                #ecvl.Normalize(sub_crop,sub_crop,mean,std)
                #sub_crop = np.array(sub_crop, copy=False)

                if sub_crop.shape[-1] == 3:
                    #channels first
                    sub_crop = np.moveaxis(sub_crop, [0,1,2], [2,1,0])

                hist.append(sub_crop)

    n_crops = len(hist)
    b = 0

    while n_crops > 0:
        tens_size = min(batch_size,n_crops)
        n_crops -= tens_size
 
        mean = mean_t.select([f":{tens_size}"])
        std = std_t.select([f":{tens_size}"])

        t = hist[b*batch_size:(b*batch_size)+tens_size]
        x = Tensor.fromarray(t, dev=gpu)
        x.sub_(mean)
        x.div_(std)
        #print(np.max(x.getdata()))
        #print(np.min(x.getdata()))
        b += 1 
        yield(x)




def comp_stats(confusion_matrix):
    np.seterr(divide='ignore', invalid='ignore')
    ret = {}
    
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    TPR[np.isinf(TPR)] = 0
    TPR[np.isnan(TPR)] = 0
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    TNR[np.isinf(TNR)] = 0
    TNR[np.isnan(TNR)] = 0
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    PPV[np.isinf(PPV)] = 0
    PPV[np.isnan(PPV)] = 0
    # Negative predictive value
    NPV = TN/(TN+FN)
    NPV[np.isinf(NPV)] = 0
    NPV[np.isnan(NPV)] = 0
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    FPR[np.isinf(FPR)] = 0
    FPR[np.isnan(FPR)] = 0
    # False negative rate
    FNR = FN/(TP+FN)
    FNR[np.isinf(FNR)] = 0
    FNR[np.isnan(FNR)] = 0
    # False discovery rate
    FDR = FP/(TP+FP)
    FDR[np.isinf(FDR)] = 0
    FDR[np.isnan(FDR)] = 0

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    ACC[np.isinf(ACC)] = 0
    ACC[np.isnan(ACC)] = 0

    BA = (TPR + TNR )/2
    BA[np.isinf(BA)] = 0
    BA[np.isnan(BA)] = 0

    F1 = ( 2 * TP )/(2 * TP + FP +FN)
    F1[np.isinf(F1)] = 0
    F1[np.isnan(F1)] = 0
    
    ret.update({'FP':FP})
    ret.update({'FN':FN})
    ret.update({'TP':TP})
    ret.update({'TN':TN})
    ret.update({'TPR':TPR})
    ret.update({'TNR':TNR})
    ret.update({'PPV':PPV})
    ret.update({'NPV':NPV})
    ret.update({'FPR':FPR})
    ret.update({'FNR':FNR})
    ret.update({'FDR':FDR})
    ret.update({'ACC':ACC})
    ret.update({'BA':BA})
    ret.update({'F1':F1})

    return ret

def standardize(I, percentile=95):
    """
    Transform image I to standard brightness.
    Modifies the luminosity channel such that a fixed percentile is saturated.
    :param I: Image uint8 BGR.
    :param percentile: Percentile for luminosity saturation. At least (100 - percentile)% of pixels should be fully luminous (white).
    :return: Image uint8 BGR with standardized brightness.
    """
    assert is_uint8_image(I), "Image should be BGR uint8."
    I_LAB = cv2.cvtColor(I, cv2.COLOR_BGR2LAB)
    L_float = I_LAB[:, :, 0].astype(float)
    p = np.percentile(L_float, percentile)
    I_LAB[:, :, 0] = np.clip(255 * L_float / p, 0, 255).astype(np.uint8)
    I = cv2.cvtColor(I_LAB, cv2.COLOR_LAB2BGR)
    return I


