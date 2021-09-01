import os
from datetime import datetime as T
import numpy as np
import cv2
import glob

# import pandas as pd
import shutil
from joblib import Parallel, delayed

base_path = "/home/vcfdregg3/바탕화면/CNV"
step1_path = os.path.join(base_path, "CNV_Image")  # 경로를 병합하여 새 경로 생성 base_path/"dir"
step2_path = os.path.join(base_path, "Preprocessed_CNV_image")

######################
def make_subdir(path):
    if os.path.exists(path):  # 해당 경로 디렉토리 참조 True or False
        try:
            shutil.rmtree(path)  # 해당 경로 디렉토리 삭제
        except OSError as ex:  # 시스템 관련 에러
            print(ex)
            exit(0)

    try:
        os.mkdir(path)  # 해당 경로 디렉토리 생성
    except OSError:
        exit(0)


######################
def save_image(pImg, pFileName):
    cv2.imwrite(pFileName, pImg)  # 첫 번째 인자를 이름, 두 번째 인자를 가져온 사진으로 받아 현재 디렉토리에 파일로 출력
    print("Saved:", pFileName)


######################
def do_filtering(pFileName):

    img = cv2.imread(pFileName)  # 이미지 파일 불러 변수에 저장, 절대 경로 필요

    fileName = os.path.split(pFileName)[1]  # 폴더 명 0번 째, 파일 명 1번 째 배열에 저장
    dst_name = fileName.replace(".jpg", "_O.JPG")  # 해당 파일 이름 변경
    dst_path = os.path.join(step2_path, dst_name)  # 경로 + 파일 이름 (절대경로) 저장
    save_image(img, dst_path)  # 해당 경로에 save_image 함수 호출

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 해당 이미지를 그레이 컬러로 변경
    img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # 수직선 방향의 에지를 검출
    img_sobel_x = cv2.convertScaleAbs(
        img_sobel_x
    )  # 결과에 절대값을 적용하고 값 범위를 8비트 unsigned int로 변경
    img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y)
    img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)  # 두 이미지 블랜딩
    dst_name = fileName.replace(".jpeg", "_SO.JPG")
    dst_path = os.path.join(step2_path, dst_name)
    save_image(img_sobel, dst_path)

    Scharr = cv2.Sobel(img_gray, -1, 0, 1, ksize=-1)  # scharr 필터를 이용하여 에지 검출
    dst_name = fileName.replace(".jpeg", "_SC.JPG")
    dst_path = os.path.join(step2_path, dst_name)
    save_image(Scharr, dst_path)

    gi = cv2.GaussianBlur(img, (5, 5), 0)  # 중심에 있는 픽셀에 높은 가중치 부여하여 엣지 검출
    dst_name = fileName.replace(".jpeg", "_G.JPG")
    dst_path = os.path.join(step2_path, dst_name)
    save_image(gi, dst_path)

    mi = cv2.medianBlur(img, 5)  # 무작위 노이즈 제거
    dst_name = fileName.replace(".jpeg", "_M.JPG")
    dst_path = os.path.join(step2_path, dst_name)
    save_image(mi, dst_path)

    bi = cv2.bilateralFilter(img, 9, 75, 75)  # 에지를 보존하면서 노이즈를 감소
    dst_name = fileName.replace(".jpeg", "_B.JPG")
    dst_path = os.path.join(step2_path, dst_name)
    save_image(bi, dst_path)

    ker = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    si = cv2.filter2D(img, -1, ker)
    dst_name = fileName.replace(".jpeg", "_S.JPG")  # 날카로운 에지 무뎌짐, 잡음의 영향 줄어듬
    dst_path = os.path.join(step2_path, dst_name)
    save_image(si, dst_path)

    imgcvted = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lc, ac, bc = cv2.split(imgcvted)
    clahe = cv2.createCLAHE(
        clipLimit=1.5, tileGridSize=(9, 9)
    )  # 일정한 영역을 분리하여 해당 영역에 대한 히스토그램 균등화(밝기 값의 평균화)
    cl = clahe.apply(lc)
    merged = cv2.merge((cl, ac, bc))
    hi = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    dst_name = fileName.replace(".jpeg", "_H.JPG")
    dst_path = os.path.join(step2_path, dst_name)
    save_image(hi, dst_path)


######################
def run_main():
    query = os.path.join(step1_path, "*.jpeg")
    fileList = glob.glob(query) # 
    fileList.sort()

    make_subdir(step2_path)

    Parallel(n_jobs=-1)(delayed(do_filtering)(fileName) for fileName in fileList)


######################
if __name__ == "__main__":
    ST = T.now()
    run_main()
    ET = T.now()
    print("Elapsed Time =", ET - ST)
