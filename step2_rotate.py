import os
from datetime import datetime as T
import numpy as np
import cv2
import glob
import shutil
from joblib import Parallel, delayed

base_path = "/home/vcfdregg3/바탕화면/CNV"
step3_path = os.path.join(base_path, "Preprocessed_CNV_image")
step4_path = os.path.join(base_path, "Rotated_CNV_Image")

######################
def make_subdir(path):  # 새로운 서브 폴더 생성 *존재시 삭제 후 재성성
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
def rotate_image(pImg, angle):  #
    (h, w) = pImg.shape[:2]  # 해당 이미지의 높이 너비 값 저장
    center = (w / 2, h / 2)  # 높이/2, 너비/2 -> 중간 지점 저장
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)  # 물체를 평면상의 한 점을 중심으로 𝜃 만큼 회전
    pImgRT = cv2.warpAffine(pImg, rot_mat, (h, w), flags=cv2.INTER_LINEAR)
    return pImgRT


######################
def save_image(pImg, path, srcFName):  # 이미지를 인자로 받아오는 해당 폴더에 저장
    fileName = os.path.join(path, srcFName)
    cv2.imwrite(fileName, pImg)
    print("Saved:", fileName)


######################
def do_rotate(pFileName): # 60도 만큼 회전(0 ~ 300)한 이미지 저장
    _, fileName = os.path.split(pFileName)
    prefix, ext = fileName.split(".")

    imgOrg = cv2.imread(pFileName)

    for i in range(0, 360, 60):
        imgNew = rotate_image(imgOrg, i)
        newFileName = prefix + "_" + "%03d.JPG" % (i)
        save_image(imgNew, step4_path, newFileName)


######################
def run_main():
    query = os.path.join(step3_path, "*.JPG")
    fileList = glob.glob(query)
    fileList.sort()

    make_subdir(step4_path)

    Parallel(n_jobs=-1)(delayed(do_rotate)(fileName) for fileName in fileList)


######################
if __name__ == "__main__":
    ST = T.now()
    run_main()
    ET = T.now()
    print("Elapsed Time =", ET - ST)
