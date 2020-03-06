import cv2
import numpy
import os
import PIL
from PIL import Image

parent_dir = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/casia a/silhouettes/'

pic_num = 0
img_arr = 0
flag = 1
for gender in os.listdir(parent_dir):
    for folder in os.listdir(parent_dir+gender):
        if(folder == "GEI"):
            continue
        for img_folder in os.listdir(parent_dir+gender+'/'+folder):
            pic_num = 0
            for img in os.listdir(parent_dir+gender+'/'+folder+'/'+img_folder):

                try:


                #print(i)
                #print(parent_dir+angle+"/"+i)
                   image = cv2.imread(parent_dir+gender+'/'+folder+'/'+img_folder+'/'+img, cv2.IMREAD_GRAYSCALE)
                   #print("asfsdf")
                   #print(image)
                   _, contour, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                   #print("success")
                   #print(contour)
                   areas = [cv2.contourArea(c) for c in contour]
                   max_index = numpy.argmax(areas)
                   cnt = contour[max_index]


                   x, y, w, h = cv2.boundingRect(cnt)
                   # print(contour)
                   print(x, y, w, h)
                   crop = image[y - 10:y + h + 10, x - 10:x + w + 10]
                   crop = cv2.resize(crop, (95, 166))
                   cv2.imwrite(parent_dir+gender+'/'+folder+'/'+img_folder+'/'+img, crop)
                   pic_num += 1
                   h, w = crop.shape
                   print("pic_Num:", pic_num, "height:", h, "width:", w)


                except Exception as e:
                   print(str(e))
            for img in os.listdir(parent_dir+gender+'/'+folder+'/'+img_folder):
                try:
                    image = cv2.imread(parent_dir+gender+'/'+folder+'/'+img_folder+'/'+img, cv2.IMREAD_GRAYSCALE)
                    arr = numpy.asarray(image, dtype=int)
                    img_arr = img_arr + arr


                except Exception as e:
                    print(str(e))

            gei = img_arr / pic_num
            img_arr = 0
            cv2.imwrite(parent_dir+gender+'/GEI/'+str(flag)+".png", gei)
            #cv2.imwrite('D:/SHIVANSH/Machine Learning/Soft Biometrics Project/test/'+walk_angle+'/GEI/' + str(flag) + ".png", gei)
            flag+=1
