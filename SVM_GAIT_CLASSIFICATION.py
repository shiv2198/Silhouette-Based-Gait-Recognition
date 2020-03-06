import cv2
import numpy as np
import os
import PIL
from PIL import Image

walk_angle = '090'
#train_file = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/train/'+walk_angle+'/'
#test_file = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/test/'+walk_angle+'/GEI/'
#test_image_loc = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/test/GEI/img.png'

#                                                        train_file = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/half GEI/train/'+str(walk_angle)+'/'
#                                                        test_file = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/half GEI/test/'+str(walk_angle)+'/'
'''CASIA A'''
#train_file = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/casia a/train/'
#test_file = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/casia a/test/'

'''SINGLE FRAME'''
train_file = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/single frame/train/'
test_file = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/single frame/test/'

dir_arr = [name for name in os.listdir(train_file)]
image_arr = []
image_arr.append([])
image_arr.append([])

###################Test Image Array
test_arr = []
test_arr.append([])
test_arr.append([])

#test_image = img = cv2.imread('D:/SHIVANSH/Machine Learning/Soft Biometrics Project/test/GEI/img.png', cv2.IMREAD_GRAYSCALE)
#print(test_image)




for file in dir_arr:

    for i in os.listdir(train_file+file):
            try:

                print('train->'+file+':'+i)

                img = cv2.imread(train_file+file+'/'+ i, cv2.IMREAD_GRAYSCALE)
                #print(img)

                image_arr[0].append(img)
                image_arr[1].append(file)


            except Exception as e:
                print(str(e))
    print("loop 1 end")

    for i in os.listdir(test_file+file):
            try:

                img = cv2.imread(test_file+file+'/'+i, cv2.IMREAD_GRAYSCALE)
                #print(img)

                test_arr[0].append(img)
                test_arr[1].append(file)


            except Exception as e:
                print(str(e))

print(image_arr[0])
#RESIZE TRAIN AND TEST IMAGES
X_train = np.array(image_arr[0]).astype(dtype=int)
nsamples , nx , ny = X_train.shape
X_train = X_train.reshape((nsamples,nx*ny))
y_train = np.array(image_arr[1])

X_test = np.array(test_arr[0]).astype(dtype=int)
nsamples , nx , ny = X_test.shape
print(nsamples, nx, ny)
X_test = X_test.reshape((nsamples,nx*ny))
y_test = np.array(test_arr[1])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print(X_test)

# Fitting Kernel SVM to the Training set

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 1, degree = 3)
classifier.fit(X_train, y_train)

predict = classifier.predict(X_test)
print(predict)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predict)
print(cm)

#accuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,predict)
print(accuracy)