import os
import argparse
from glob import glob
import openslide

import numpy as np
import multiprocessing
import time
import logging
import gc
logging.basicConfig(filename='crop_ecvl.log', level=logging.INFO)
import annotation_ecvl as annotation

from pyecvl import ecvl
import cropify_core_ecvl as cg
import time
# pyecvl             0.9.1

"""
Wholeslide preprocessing with pyecvl.

"""
                                                                                    

root = None 
output_root = None 
classes = list() 
dataset = {}

parallel_jobs = 8
parallel_data = list()
jobs = list()
jobs_progress = None


def make_crops_generator(job_id, label, path, size=500 , annotation_list = None, root_roi = '', pxsize=224, extension='jpg'):
  global jobs_progress
  try:
    slide = openslide.OpenSlide(path)
  except Exception as e:
    logging.error("\tJob {} Could not open".format(job_id), path, e)
    return

  if not os.path.isdir(output_root + label):
    try:
      os.mkdir(output_root + label)
    except Exception as e:
      logging.error(e, exc_info=1)
 
  try: 
    gc.collect()
    name = path.split('/')[-1]
    rois = None

    print("Processing: " + name)

    if (name + ".ndpa") in annotation_list:
        rois = annotation.get_annotation_list(root_roi + name + ".ndpa", path)

    logging.info("from {} got metadata: {}".format(name,str(rois is not None)))
    crops_g = cg.make_crops((slide,path), size , annotation_list=rois, pxsize=pxsize, usemask=False) 
    num_crops = next(crops_g)


  except Exception as e:
    print(path)
    logging.error(path, exc_info=1)
    print(e)
    logging.error(e, exc_info=1)
    crops_g = list()
    rois = None

  
  curr_progress = jobs_progress[job_id]
  try:
    i = 0
    for (cx, cy, w,h, crop, reg) in crops_g:
      output_base = output_root + label + '/' + name
      output = output_base

      if rois is not None:
        output += '_ROI_'

      output += '_mpp{6:.2f}_reg{0:03d}_crop_sk{1:05d}_({2:d},{3:d},{4:d},{5:d}).{7}'.format(reg,i, cx, cy, w, h, float(slide.properties[openslide.PROPERTY_NAME_MPP_X]), extension)
      
      ecvl.ImWrite(output, crop)

      crop=None
      hold=None
      gc.collect()
      crop_progress = (reg * 100) / num_crops
      crop_progress = crop_progress / len(parallel_data[job_id])
      jobs_progress[job_id] = curr_progress + crop_progress
      i += 1
  except Exception as e:
     print(e)
     logging.error(output_base, exc_info=1)
     logging.error(e, exc_info=1)
  jobs_progress[job_id] = curr_progress + 100/len(parallel_data[job_id])

def chunkify(lst, n):
  return [lst[i::n] for i in range(n)]

def run_job(id, data, size,  rois, root_roi, pxsize, extension):
  global jobs_progress
  
  j = 0
  for (label, path) in data:
    jobs_progress[id] = (j*100) / len(data)
    make_crops_generator(id, label, path, size,   rois, root_roi,  pxsize, extension)
    j += 1
  jobs_progress[id] = 100

start_time = time.time()

def print_progress():
  global jobs_progress
  global parallel_data
  global start_time

  #os.system("clear")
  print("Running {} jobs".format(parallel_jobs))
  print("Overall progress: {}%\t\tTime: {}".format(
      int(sum(jobs_progress) / parallel_jobs), 
      time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
  ))
  print("\n\n\n")

  for id, progr in enumerate(jobs_progress):
    print("Job {} [".format(id) + '#'*int(progr) + "] ({}) {}%".format(len(parallel_data[id]), int(progr)))
  time.sleep(5)


parser = argparse.ArgumentParser()
parser.add_argument("data", help="dataset path", type=str)
parser.add_argument("output", help="Output path", type=str)
parser.add_argument("--ROIs", help="Path for meatadata folder", type=str, default='.')
parser.add_argument("--extension", help="output image format (default='png')" ,type=str, default='png')
parser.add_argument("--jobs", help="Number of parallel jobs", type=int, default=4)
parser.add_argument("--size", help="Crop size (in μm)", type=int, default=500)
parser.add_argument("--pxsize", help="Crop size (in px) ( fullres -1)", type=int, default = -1)
parser.add_argument("--subset", help="subset of the slide (text file descriptor, default test_wsi.txt )", type=str, default='test_wsi.txt')

if __name__ == '__main__':
  args = parser.parse_args()
 
  subset = None
  if args.subset != '':
      with open(args.subset, 'r') as f:
          names = list(f.readlines())
          subset = [line.rstrip() for line in names] #['100-B2-TVALG']
  
  size = args.size
  default_mpp = 0.4415

  print("Crop size:", size)

  pxsize = args.pxsize
  if pxsize == -1:
      pxsize = int(size / default_mpp)

  root = args.data
  print("Loading data from", root)
  root_roi = args.ROIs
  classes = os.listdir(root)
  if 'crops' in classes:
    classes.remove('crops')

  dataset = {}
  output_root = args.output

  if not os.path.isdir(output_root):
    os.mkdir(output_root)

  # Data Collection
  for c in classes:
    files = glob(root + c + "/*.ndpi")
    if subset:
      files = [ f for f in files if (f.split('/')[-1]).split('.ndpi')[0] in subset ]
    dataset[c] = files

  data = list()
  for label in classes:
    for path in dataset[label]:
      data.append((label, path))
  
  # Annotation Collection
  rois = list()
  if os.path.isdir(root_roi):
    for roi in glob(root_roi + "/*.ndpa"):
      rois.append(roi.split('/')[-1])
  else:
    rois = []

  print("Total data length", len(data))
  print("Total annotation files", len(rois))

  parallel_jobs = args.jobs
  parallel_data = chunkify(data, parallel_jobs)
  jobs_progress = multiprocessing.Array('d', parallel_jobs) 

  print("Starting parallel computation ({} jobs)".format(parallel_jobs))
  start_time = time.time()

  for i in range(parallel_jobs):
    p = multiprocessing.Process(target=run_job, args=(i, parallel_data[i], size ,  rois, root_roi, pxsize, args.extension))
    p.daemon = False
    jobs.append(p)
    p.start()
    print(i)

  while not (all(x == 100 for x in jobs_progress)):
    print_progress()
  print_progress()

  for j in jobs:
    j.join()

  print("---Time to Preprocess: %s seconds ---" % (time.time() - start_time))
    
