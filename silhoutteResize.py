import cv2
import numpy as np
import os


pos_file = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/090/'

pic_num = 0

#img = cv2.imread(pos_file , cv2.IMREAD_GRAYSCALE)

for i in os.listdir(pos_file):
        try:

            print(i)

            img = cv2.imread(pos_file+i , cv2.IMREAD_GRAYSCALE)

            _,contour,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contour[0]
            x, y, w, h = cv2.boundingRect(cnt)
           #print(contour)
            print(x, y, w, h)
            crop = img[y - 10:y + h + 10, x - 10:x + w + 10]
            crop = cv2.resize(crop, (95, 166))
            cv2.imwrite('D:/SHIVANSH/Machine Learning/Soft Biometrics Project/GEI/'+str(pic_num)+'.png', crop)

            h, w = crop.shape
            print("pic_Num:", pic_num, "height:", h, "width:", w)

            pic_num += 1
            print(pic_num)

        except Exception as e:
            print(str(e))




