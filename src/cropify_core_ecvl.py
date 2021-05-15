import openslide
import numpy as np
import gc
from skimage import filters
from skimage import measure
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage import exposure
from PIL import Image
import matplotlib.pyplot as plt
import math
import logging

from annotation_ecvl import *
from skimage.measure import block_reduce
from pyecvl import ecvl
import cv2
# pyecvl             0.9.1

def expand2square(img, background_color, pxsize):
    
    # cxy format
    c, width, height = img.dims_
    if width == height:
        return img
    result = numpy.full((c, pxsize, pxsize), background_color, dtype=numpy.uint8)
    y,x = (pxsize - height) // 2, (pxsize - width) // 2
    result[:, x:x+width, y:y+height ] = np.array(img)
    result = ecvl.Image.fromarray(result,'cxy',ecvl.ColorType.RGB)
    return result


def is_histopath(img, purple_threshold = 100, purple_scale_size = 15):

    dim = min(img.shape[0],img.shape[1])

    while int(dim * 0.5) > 224:
        img = cv2.resize(img,(int(dim * 0.5), int(dim * 0.5)),interpolation=cv2.INTER_LINEAR)
        dim = int(dim * 0.5)
    img = cv2.resize(img,(224, 224),interpolation=cv2.INTER_LINEAR)
    crop = np.asarray(img)

    block_size = (max(1,crop.shape[0] // purple_scale_size),
                  max(1,crop.shape[1] // purple_scale_size), 1)
    pooled = block_reduce(image=crop, block_size=block_size, func=np.average)

    # Calculate boolean arrays for determining if portion is purple.
    r, g, b = pooled[..., 0], pooled[..., 1], pooled[..., 2]
    cond1 = r > g - 10
    cond2 = b > g - 10
    cond3 = ((r + b) / 2) > g + 20

    # Find the indexes of pooled satisfying all 3 conditions.
    pooled = pooled[cond1 & cond2 & cond3]
    num_purple = pooled.shape[0]

    return num_purple > purple_threshold

def main_regions(boxes, threshold=0.5):

  def contained(a, b):  # returns 0 if rectangles don't intersect
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    aa = (a[3]-a[1]) * (a[2] - a[0])
    if (dx>=0) and (dy>=0):
      return (dx*dy)/aa
    return 0

  aus = boxes.copy()
  for b in boxes:
    for a in aus:
      if b!=a and b in aus and contained(b, a) >= threshold:
        aus.remove(b)
  return aus

def make_crops(slide, size_m=300, stride=1, annotation_list=None ,pxsize=224, mincropxregion=1, usemask=False, crop_center=False ):

  thumb_max_size = 2000
  slide,slidename = slide

  levels = ecvl.OpenSlideGetLevels(slidename)
  width_s, height_s = levels[0]
 
  slide_ratio = width_s / height_s
  
  for i,l in enumerate(levels):
    if l[0]<thumb_max_size and l[0]<thumb_max_size:
      level_thumb = i
      break

  width_t, height_t = levels[level_thumb] 

  ratio_w = width_s / width_t 
  ratio_h = height_s / height_t 
 
  mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X]) #squared crops
  size = math.ceil(size_m / mpp)

  ## each patch of resulting image (pxsize,pxsize) will be 'size_m' microns
  compression_factor = size_m / (pxsize * mpp)
  pxsize = math.ceil(pxsize)
  level = 0
  while compression_factor/2 > 1 :
     compression_factor /= 2
     level += 1
  
  if annotation_list is None:

    thumb = ecvl.OpenSlideRead(slidename, level_thumb, (0, 0, width_t, height_t))
    ecvl.ChangeColorSpace(box, box, ecvl.ColorType.RGB)

    grayscale = ecvl.Image.empty()
    ecvl.ChangeColorSpace(thumb, grayscale, ecvl.ColorType.GRAY)

    threshold = ecvl.OtsuThreshold(grayscale)
    #from cxy to xyc
    ecvl.RearrangeChannels(grayscale, grayscale, 'xyc', new_type=ecvl.DataType.uint8)

    bw = (np.array(grayscale) < threshold).astype('uint8') * 255

    bw = ecvl.Image.fromarray(bw,'xyc',ecvl.ColorType.GRAY)

    label_image = ecvl.Image.empty()
    ecvl.ConnectedComponentsLabeling(bw,label_image)

    #from xyc to yxc
    label_image = np.squeeze(numpy.moveaxis(numpy.array(label_image), [0, 1, 2], [1,0,2]))

    # from roi statistics, searching for tissue areas min 2500 micron
    roi_meanarea = math.ceil( (1000 / mpp) / ratio_w ) **2
    region_minarea = mincropxregion * int( size / ratio_w)**2

    region_minarea = min( region_minarea, roi_meanarea)

    proposed_regions = regionprops(label_image)

    #skip wider region: avoid scans with burden edges
    proposed_regions = [r for r in proposed_regions if r.bbox_area < label_image.size * 0.95]

    #skip region from artifacts: avoid scan-wide flaps
    def is_flap(region):
      minr, minc, maxr, maxc = region.bbox
      rt = (maxc-minc)/(maxr-minr)
      return rt < 0.2 or rt > 5
    proposed_regions = [r for r in proposed_regions if not is_flap(r)]

    boxes = []

    thumb = numpy.moveaxis(numpy.array(thumb), [0, 1, 2], [2,1,0])

    for region in proposed_regions:
      minr, minc, maxr, maxc = region.bbox
      if region.bbox_area >= region_minarea and is_histopath(thumb[minr:maxr,minc:maxc],100,15):
        boxes.append((minc, minr, maxc, maxr))

    if len(boxes) == 0:
       proposed_regions.sort(key=lambda x: x.bbox_area, reverse=True)
       for r in proposed_regions:
         minr, minc, maxr, maxc = r.bbox
         if is_histopath(thumb[minr:maxr,minc:maxc],100,15):
           boxes.append((minc, minr, maxc, maxr))
           break

    
    #discard boxes with more than threshold overlap
    boxes = main_regions(boxes, threshold=0.5)

    for i in range(len(boxes)):
      (x, y, xx, yy) = boxes[i]
      w = int((xx-x) * ratio_w)
      h = int((yy-y) * ratio_h)
      x = int(x * ratio_w)
      y = int(y * ratio_h)
      boxes[i] = (x,y,w,h,i)

    #region minimum size
    
    for b in range(len(boxes)):
      x,y,width,height,i = boxes[b]

      if width < size:
        x = max(0,x - (size - width)//2)
        width = min(size,width_s - x)

      if height < size:
        y = max(0,y - (size - height)//2)
        height = min(size,height_s - y)      

      boxes[b] = x,y,width,height,i
    
    num_crops = len(boxes)
 
    yield max(1,num_crops)
    

  else:
 
    boxes = []
    num_crops = 0
    for i,a in enumerate(annotation_list):
      x,y,width,height = a.get_enclosing_rectangle_pixels()

      #avoid annotation slide overflows (yep that can happen)
      x = max(x,0)
      y = max(y,0)
      width = min(width,width_s-x)
      height = min(height,height_s-y)

      ## minsize for each ROI

      if width < size:
        x = max(0,x - (size - width)//2)
        width = min(size,width_s - x)

      if height < size:
        y = max(0,y - (size - height)//2)
        height = min(size,height_s - y)
 
      boxes.append((x,y,width,height,i))
      
      num_crops += 1
    yield max(1,num_crops)

  logging.info("Got {} boxes".format(len(boxes)))
  logging.info("Got {} crops".format(num_crops))
  logging.info("Level 0 size: {}".format(slide.dimensions)) 

  thumb = None
  grayscale = None
  bw = None
  label_image = None 
  gc.collect()

  for (start_x, start_y, width, height, numbox) in boxes:        

    box = None
    gc.collect()
    try:
        

        side_x,side_y = min(width_s-start_x,size),min(height_s-start_y,size)
        step = size * stride

        end_y = max(start_y + 1, start_y + height - side_y + 1)
        end_x = max(start_x + 1, start_x + width - side_x + 1)

        n_w = width/side_x
        n_h = height/side_y
        e_num_crops = int(n_w) * int(n_h)

        #if i can take one patch, center the crop
        if crop_center and stride == 1 and n_w < 2:
            start_x += max(0,(width-side_x)//2)
            side_x = min(width_s-start_x,size)
        if crop_center and stride == 1 and n_h < 2:
            start_y += max(0,(height-side_y)//2)
            side_y = min(height_s-start_y,size)
  
        find_one = False
        is_good = False

        for cy in range(start_y, end_y , step):
          for cx in range(start_x, end_x , step):

            if annotation_list is None:
              scale = 2**level
              box = ecvl.OpenSlideRead( slidename , level, (cx, cy, side_x//scale, side_y//scale))
            elif usemask:
              box = annotation_list[numbox].get_image_cropped(boxcut=(cx,cy,side_x,side_y),level=level)
            else:
              box = annotation_list[numbox].get_enclosing_image(boxcut=(cx,cy,side_x,side_y),level=level)

            
            # compress the image
            if compression_factor > 1:
              # cxy format !! ecvl.ResizeDim changes channells to xyc
              ecvl.ResizeDim(box, box, [math.ceil(box.dims_[1] / compression_factor), math.ceil(box.dims_[2] / compression_factor)], interp=ecvl.InterpolationType.lanczos4)
              ecvl.RearrangeChannels(box, box, 'cxy')
            
            ecvl.ChangeColorSpace(box, box, ecvl.ColorType.RGB)
            img_res = numpy.moveaxis(numpy.array(box), [0, 1, 2], [2,1,0])
            is_good = e_num_crops < 2 or is_histopath(img_res,100,15)
            find_one = find_one or is_good

            if is_good:
              box = expand2square(box, 0, pxsize)
              yield (cx, cy, side_x, side_y, box, numbox)

        if not find_one:

          #middle crop, in case of 0 crop from RoI
          middle_cy = start_y + max(0,(height - side_y)//2)
          middle_cx = start_x + max(0,(width - side_x)//2)
          side_x,side_y = min(width_s - middle_cx, side_x),min(height_s - middle_cy, side_y)

          if annotation_list is None:
              scale = 2**level
              box = ecvl.OpenSlideRead( slidename , level, (middle_cx, middle_cy, side_x//scale, side_y//scale))
          elif usemask:
            box = annotation_list[numbox].get_image_cropped(boxcut=(middle_cx,middle_cy,side_x,side_y),level=level)
          else:
            box = annotation_list[numbox].get_enclosing_image(boxcut=(middle_cx,middle_cy,side_x,side_y),level=level)

          if compression_factor > 1:
            # cxy format
            ecvl.ResizeDim(box, box, [math.ceil(box.dims_[1] / compression_factor), math.ceil(box.dims_[2] / compression_factor)], interp=ecvl.InterpolationType.lanczos4)
            ecvl.RearrangeChannels(box, box, 'cxy')

          ecvl.ChangeColorSpace(box, box, ecvl.ColorType.RGB)
          box = expand2square(box, 0, pxsize)
          yield (middle_cx, middle_cy, side_x, side_y, box, numbox)
        continue

    except Exception as e:
      print(e)
      logging.error(e, exc_info=1)
      continue
        
