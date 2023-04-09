import torch
import time ##
import re
import easyocr
import cv2
import os
import matplotlib.pyplot as plot
import numpy as np

##------------------MODELO ENTRENADO-----------------------##
#Model Path
M_path = r"/home/erwin19t/Documents/Tesis/Codes/YOLO/yolov5/"
New_path = r"/home/erwin19t/Documents/Tesis/Codes/YOLO/video/V03.mp4"
 #Weights_Path
W_Path = r"/home/erwin19t/Documents/Tesis/Codes/YOLO/yolov5/runs/train/exp/weights/last.pt"
model =  torch.hub.load(M_path, 'custom', source = 'local', path = W_Path, force_reload = True) ### The repo is stored locally

classes = model.names ### class names in string format

##------------------RECONOCIMIENTO DE PLACA------------##
def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates



##---------------------------EASYOCR----------------------------------##
reader = easyocr.Reader(['en'], gpu = True) 
cap = cv2.VideoCapture('V02.mp4')

if cap.isOpened():
    #-----CREACIÒN DE ESP DE MEMORIA PARA VIDEO A GENERAR-----##
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v') ##(*'XVID')
    out = cv2.VideoWriter(New_path, codec, fps, (width, height))

    while True:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break   
        results = detectx(frame, model = model) ### DETECTION HAPPENING HERE  
        labels, cord = results
        print("labels", labels[0])
        row = cord[0]
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        #frame = frame[y1-3:y2-3,x1-3:x2-3]
        
        RST = reader.readtext(frame, 'greedy', 5, 1, 0, allowlist = '0123456789ABCDEFGHJKLMNPRSTUVWXYZ-')
        font = cv2.FONT_HERSHEY_SIMPLEX
        for res in RST:
            print("res:\n", res)
            pt0 = res[0][0]
            pt1 = res[0][1]
            pt2 = res[0][2]
            pt3 = res[0][3]
            if((len(res[1][:])==9 | (len(res[1][:])==10)) & ('-' in res[1][:])):         
                cv2.rectangle(frame,(x1, y1), (x2, y2), (166, 56, 242), 2)
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (166, 56, 242), -1)
                cv2.putText(frame, res[1], (x1, y1), 2, 1.2, (0, 0, 0), 1)
            #-------------GUARDAR NUEVO VIDEO------------------#
            if New_path:
                print(f"[INFO] Saving output video. . . ")
                out.write(frame)
out.release()
cv2.destroyAllWindows()
