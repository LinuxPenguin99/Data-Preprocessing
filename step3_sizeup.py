import os
from datetime import datetime as T
import numpy as np
import cv2
import glob
# import pandas as pd
import shutil
from joblib import Parallel, delayed

S140 = 140
S256 = 256
S299 = 299
S512 = 512
DIM  = S299

base_path  = '/home/vcfdregg3/바탕화면/CNV'
step2_path = os.path.join(base_path, "Rotated_CNV_Image")
step3_path = os.path.join(base_path, 'Sized_Up_Image')

JOBCODE   = ['O', 'U']

######################
def make_subdir(path):
  if os.path.exists(path):
    try:
      shutil.rmtree(path)
    except OSError as ex:
      print(ex)
      exit(0)

  try:
    os.mkdir(path)
  except OSError:
    exit(0)
######################
def resize_image(pImg, SIZE): # 해당 이미지를 파마리터만큼 리사이즈
  imgRS = cv2.resize(pImg, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
  return imgRS
######################
def save_image(pImg, path, srcFName):
  fileName = os.path.join(path, srcFName)
  cv2.imwrite(fileName, pImg)
  print('Saved:', fileName)
######################
def get_square_image(pImg):
  (h, w) = pImg.shape[:2]
  r = int(w * 0.15)
  y0 = r
  y1 = h - r
  x0 = r
  x1 = w - r

  pImgSQ = pImg[y0:y1, x0:x1, :]

  return pImgSQ
######################
def do_job_BR(pImg, SIZE): # 해당 이미지 리사이즈
  pImgNew = resize_image(pImg, SIZE)
  return pImgNew
######################
def do_job_BU(pImg, SIZE): # 정해진 값에 의해 확대 과정을 거친 후 해당 이미지 리사이즈
  pImgSQ = get_square_image(pImg)
  pImgNew = resize_image(pImgSQ, SIZE)
  return pImgNew
######################
def do_job(pImg, pJobCode, SIZE):
  if   pJobCode == JOBCODE[0]:
    pImgNew = do_job_BR(pImg, SIZE)
  elif pJobCode == JOBCODE[1]:
    pImgNew = do_job_BU(pImg, SIZE)
  return pImgNew
######################
def do_rescale(pFileName):
  _, fileName = os.path.split(pFileName)
  prefix, ext = fileName.split('.') # 파일명, 확장자명 분리

  imgOrg = cv2.imread(pFileName)

  for jobCode in JOBCODE: # 해당 이미지를 리스케일하여 새로운 이름으로 저장
    newFileName = prefix + '_' + jobCode + '.' + ext
    imgNew = do_job(imgOrg, jobCode, DIM)
    save_image(imgNew, step3_path, newFileName)
######################
def run_main():
  query    = os.path.join(step2_path, '*.JPG')
  fileList = glob.glob(query)
  fileList.sort()

  make_subdir(step3_path)

  Parallel(n_jobs=-1)(delayed(do_rescale)
                      (fileName) for fileName in fileList)

######################
if __name__ == '__main__':
  ST = T.now()
  run_main()
  ET = T.now()
  print('Elapsed Time =', ET-ST)
