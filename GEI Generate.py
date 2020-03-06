import cv2
import numpy
import os
import PIL
from PIL import Image

walk_angle = '018'

#parent_dir = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/test/'+walk_angle+'/img/'
#print(parent_dir)
parent_dir =  'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/dataset/'+walk_angle+'/boy/'
#print(parent_dir)
#parent_dir =  'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/dataset/'+walk_angle+'/girl/'
print(parent_dir)


pic_num = 0
img_arr = 0
flag = 1

for angle in os.listdir(parent_dir):
    if(angle == "GEI"):
        continue
    for i in os.listdir(parent_dir+angle):
        try:


            #print(i)
            #print(parent_dir+angle+"/"+i)
            image = cv2.imread(parent_dir+angle+"/"+i, cv2.IMREAD_GRAYSCALE)
            #print("asfsdf")
            #print(image)
            _, contour, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #print("success")
            #print(contour)
            cnt = contour[0]

            x, y, w, h = cv2.boundingRect(cnt)
            # print(contour)
            print(x, y, w, h)
            crop = image[y - 10:y + h + 10, x - 10:x + w + 10]
            crop = cv2.resize(crop, (95, 166))
            cv2.imwrite(parent_dir+angle+"/"+i, crop)
            pic_num += 1
            h, w = crop.shape
            print("pic_Num:", pic_num, "height:", h, "width:", w)


        except Exception as e:
            print(str(e))
    for i in os.listdir(parent_dir+angle):
        try:
            image = cv2.imread(parent_dir+angle+"/"+i, cv2.IMREAD_GRAYSCALE)
            arr = numpy.asarray(image, dtype=int)
            img_arr = img_arr + arr


        except Exception as e:
            print(str(e))

    gei = img_arr / pic_num

    cv2.imwrite(parent_dir+'GEI/'+str(flag)+".png", gei)
    #cv2.imwrite('D:/SHIVANSH/Machine Learning/Soft Biometrics Project/test/'+walk_angle+'/GEI/' + str(flag) + ".png", gei)
    flag+=1



''' while True:
        cv2.imshow('img', gei)
        k = cv2.waitKey(30) & 0xFF
        if k == 32:
            break
# cleanup the camera and close any open windows
cv2.destroyAllWindows()'''

