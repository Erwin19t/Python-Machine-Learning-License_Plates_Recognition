import torch
import time
import re
import easyocr
import cv2
import os
import matplotlib.pyplot as plot
import numpy as np

##------------------MODELO ENTRENADO-----------------------##
#Model Path
M_path = r"/home/erwin19t/Documents/Tesis/Codes/YOLO/yolov5/"
 #Weights_Path
W_Path = r"/home/erwin19t/Documents/Tesis/Codes/YOLO/yolov5/runs/train/exp/weights/last.pt"
model =  torch.hub.load(M_path, 'custom', source = 'local', path = W_Path, force_reload = True) ### The repo is stored locally

classes = model.names ### class names in string format

##------------------RECONOCIMIENTO DE PLACA------------##
img_path = r"/home/erwin19t/Documents/Tesis/Codes/YOLO/Dataset/Test/P13.jpg"
IMG = cv2.imread(img_path, 1)
Dim = (900, 1200)
IMG = cv2.resize(IMG, Dim, interpolation = cv2.INTER_AREA)
#cv2.imshow('img', IMG)
#cv2.waitKey(0)

def detectx (IMG, model):
    IMG = [IMG]
    print(f"[INFO] Detecting. . . ")
    results = model(IMG)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates

results = detectx(IMG, model = model) ### DETECTION HAPPENING HERE    
labels, cord = results
row = cord[0]

x_shape, y_shape = IMG.shape[1], IMG.shape[0]
x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
IMG = IMG[y1-3:y2-3,x1-3:x2-3]
#cv2.imshow('img', IMG)
#cv2.waitKey(0)

##---------------------------EASYOCR----------------------------------##
reader = easyocr.Reader(['en'], gpu = True) 
RST = reader.readtext(IMG, 'greedy', 5, 1, 0, allowlist = '0123456789ABCDEFGHJKLMNPRSTUVWXYZ-')
RST
#, height_ths = 5
font = cv2.FONT_HERSHEY_SIMPLEX

for res in RST:
    print("res:\n", res)
    pt0 = res[0][0]
    pt1 = res[0][1]
    pt2 = res[0][2]
    pt3 = res[0][3]
    if(len(res[1][:])==9 | (len(res[1][:])==10)):
        # assume pt1 is a tuple of (x, y) coordinates
        if isinstance(pt1, tuple) and len(pt1) == 2:
        # convert pt1 to a tuple with 4 elements
         pt1 = (pt1[0], pt1[1] - 20, pt1[0] + 20, pt1[1])
         round(pt1[0], 5)
         round(pt1[1], 5)
         round(pt0[0], 5)
         round(pt0[0], 5)
        cv2.rectangle(IMG, pt0, pt1, (0, 100, 255), -1)
        cv2.putText(IMG, res[1], (pt0[0], pt0[1]-3), 1, 1.4, (0, 255, 0), 1)
        cv2.rectangle(IMG, pt0, pt2, (166, 56, 242), 2)
        cv2.circle(IMG, pt0, 2, (255, 0, 0), 2)
        cv2.circle(IMG, pt1, 2, (0, 255, 0), 2)
        cv2.circle(IMG, pt2, 2, (0, 0, 255), 2)
        cv2.circle(IMG, pt3, 2, (0, 255, 255), 2)
cv2.imshow('img', IMG)
cv2.waitKey(0)
