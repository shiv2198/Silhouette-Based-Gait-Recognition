import cv2
import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
import pandas as pd

dataset = pd.read_csv('D:/SHIVANSH/Machine Learning/Machine_Learning/facial expression recognition/fer2013/fer2013.csv')


cap = cv2.VideoCapture(0)

ret, img = cap.read()
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x = np.array(gray_image).astype(float)
a,b = x.shape
x = x.reshape((1,a*b))
print(a , b)
print(x)
while 1:
    cv2.imshow('img', gray_image)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
# '''Image Dilation for disturbance in gait'''
#
# parent_dir = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/casia a/silhouettes/'
#
# for gender in os.listdir(parent_dir):
#     for folder in os.listdir(parent_dir+gender):
#         if(folder == "GEI"):
#             continue
#         for img_folder in os.listdir(parent_dir+gender+'/'+folder):
#             pic_num = 0
#             for img in os.listdir(parent_dir+gender+'/'+folder+'/'+img_folder):
#
#                 try:
#                     print(gender+'=>'+img)
#                     image = cv2.imread(parent_dir+gender+'/'+folder+'/'+img_folder+'/'+img)
#                     kernel = np.ones((10 ,1), np.uint64)
#                     dilation = cv2.dilate(image, kernel, iterations=1)
#                     #kernel = np.ones((5, 5), np.uint64)
#                     #dilation = cv2.dilate(image, kernel, iterations=1)
#                     #cv2.imwrite('C:/Users/admin/Desktop/delete/'+img, dilation)
#                     cv2.imwrite(parent_dir+gender+'/'+folder+'/'+img_folder+'/'+img, dilation)
#
#
#
#
#                 except Exception as e:
#                     print(str(e))
#





# walk_angle = '180'
# path = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/half GEI/'
# print(walk_angle)
# for type in os.listdir(path):
#     for gender in os.listdir(path+type+'/'+str(walk_angle)):
#         for i in os.listdir(path+'/'+type+'/'+str(walk_angle)+'/'+gender):
#             img = cv2.imread(path+'/'+type+'/'+str(walk_angle)+'/'+gender+'/'+i,cv2.IMREAD_GRAYSCALE)
#
#             crop = img[73:166, 0:95]
#
#             cv2.imwrite(path+'/'+type+'/'+str(walk_angle)+'/'+gender+'/'+i,crop)
#
#






# count = 31

# parent_dir = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/test/'+str(walk_angle)+'/'+'GEI/'
#
# for gender in os.listdir(parent_dir):
#
#     for i in os.listdir(parent_dir+gender):
#         try:
#
#             img = cv2.imread(parent_dir+gender+'/'+i, cv2.IMREAD_GRAYSCALE)
#             print(img)
#             flip = np.fliplr(img)
#             image = Image.fromarray(flip)
#             image.save(parent_dir+gender+'/'+str(count)+'.png')
#             count +=1
#
#
#
#
#
#         except Exception as e:
#             print(str(e))





















# path = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/test/072/GEI/girl/10.png'
# img = cv2.imread(path , cv2.IMREAD_GRAYSCALE)
# print(img)
# flip = np.fliplr(img)
# image = Image.fromarray(flip)
# image.save('C:/Users/admin/Desktop/delete/test2.png')

# count = 1
# path = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/test/072/GEI/girl/'
# for i in os.listdir(path):
#
#     img = cv2.imread(path+i , cv2.IMREAD_GRAYSCALE)
#     print(img)
#     flip = np.fliplr(img)
#     image = Image.fromarray(flip)
#     image.save('C:/Users/admin/Desktop/delete/'+str(count)+'.png')
#   count +=1

#cv2.imshow('flip',flip)

#shape = [166 , 95 , 1]
#
#x = tf.placeholder(dtype=tf.float32 , shape=shape)
#
#flip2 = tf.image.flip_up_down(x)
#
#flip3 = tf.image.flip_left_right(x)
#
#flip4 = tf.image.random_flip_left_right(x)
#
#cv2.imwrite("C:/Users/admin/Desktop/delete",flip)
#cv2.imwrite('C:/Users/admin/Desktop/delete',flip2)
#cv2.imwrite('C:/Users/admin/Desktop/delete',flip3)
#cv2.imwrite('C:/Users/admin/Desktop/delete',flip4)
#
#


















































# # Importing the Keras libraries and packages
# from keras.models import Sequential
# from keras.layers import Convolution2D
# from keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense
# import matplotlib.pyplot as plt
#
#
# # Initialising the CNN
# classifier = Sequential()
#
# # Step 1 - Convolution
# classifier.add(Convolution2D(128, 3, 3, input_shape = (256, 256, 3), activation = 'relu'))
#
# # Step 2 - Pooling
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
#
# # Adding a second convolutional layer
# classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
#
#
# # Adding a third convolutional layer
# classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
#
#
# # Adding a forth convolutional layer
# classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
#
# # Step 3 - Flattening
# classifier.add(Flatten())
#
# # Step 4 - Full connection
# classifier.add(Dense(output_dim = 256, activation = 'relu'))
# classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
#
# # Compiling the CNN
# classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
# # Part 2 - Fitting the CNN to the images
#
# from keras.preprocessing.image import ImageDataGenerator
#
# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    horizontal_flip = True)
#
# test_datagen = ImageDataGenerator(rescale = 1./255)
#
# training_set = train_datagen.flow_from_directory('D:/SHIVANSH/Machine Learning/Soft Biometrics Project/train',
#                                                  target_size = (256, 256),
#                                                  batch_size = 32,
#                                                  class_mode = 'binary')
#
# test_set = test_datagen.flow_from_directory('D:/SHIVANSH/Machine Learning/Soft Biometrics Project/test/GEI',
#                                             target_size = (256, 256),
#                                             batch_size = 32,
#                                             class_mode = 'binary')
#
# classifier.fit_generator(training_set,
#                          samples_per_epoch = 40,
#                          nb_epoch = 50,
#                          validation_data = test_set,
#                          nb_val_samples = 38)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# pos_file = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/ALLINONE/train/'
# test_file = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/ALLINONE/test/'
# #test_image_loc = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/test/GEI/img.png'
#
# dir_arr = [name for name in os.listdir(pos_file)]
# image_arr = []
# image_arr.append([])
# image_arr.append([])
#
# ###################Test Image Array
# test_arr = []
# test_arr.append([])
# test_arr.append([])
#
# #test_image = img = cv2.imread('D:/SHIVANSH/Machine Learning/Soft Biometrics Project/test/GEI/img.png', cv2.IMREAD_GRAYSCALE)
# #print(test_image)
#
#
#
#
# for file in dir_arr:
#
#     for i in os.listdir(pos_file+file):
#
#         for image in os.listdir(pos_file+file+'/'+i):
#
#             try:
#
#                 print('train->'+file+':'+i)
#
#                 img = cv2.imread(pos_file+file+'/'+i+'/'+image, cv2.IMREAD_GRAYSCALE)
#                 #print(img)
#
#                 image_arr[0].append(img)
#                 image_arr[1].append(file)
#
#
#             except Exception as e:
#                 print(str(e))
#     print("loop 1 end")
#
#     for i in os.listdir(test_file+file):
#
#         for image in os.listdir(test_file + file + '/' + i):
#
#             try:
#
#                 print('train->' + file + ':' + i)
#
#                 img = cv2.imread(test_file + file + '/' + i + '/' + image, cv2.IMREAD_GRAYSCALE)
#                 # print(img)
#
#                 test_arr[0].append(img)
#                 test_arr[1].append(file)
#
#
#             except Exception as e:
#                 print(str(e))
#
#
# #RESIZE TRAIN AND TEST IMAGES
# X_train = np.array(image_arr[0]).astype(dtype=int)
# nsamples , nx , ny = X_train.shape
# X_train = X_train.reshape((nsamples,nx*ny))
# y_train = np.array(image_arr[1])
#
# X_test = np.array(test_arr[0]).astype(dtype=int)
# nsamples, nx , ny = X_test.shape
# X_test = X_test.reshape((nsamples,nx*ny))
# y_test = np.array(test_arr[1])
#
# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# # Fitting Kernel SVM to the Training set
#
# from sklearn.svm import SVC
# classifier = SVC(kernel = 'rbf', random_state = 1, degree = 3)
# classifier.fit(X_train, y_train)
#
# predict = classifier.predict(X_test)
# print(predict)
#
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, predict)
#
# #accuracy score
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(y_test,predict)
# print(cm)
# print(accuracy)


'''SINGLE FRAME SVM TEST WITH GEI TRAINED MODEL'''
# pos_file = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/single frame/train/'
# test_file = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/single frame/test/'
# #test_image_loc = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/test/GEI/img.png'
#
# dir_arr = [name for name in os.listdir(pos_file)]
# image_arr = []
# image_arr.append([])
# image_arr.append([])
#
# ###################Test Image Array
# test_arr = []
# test_arr.append([])
# test_arr.append([])
#
# #test_image = img = cv2.imread('D:/SHIVANSH/Machine Learning/Soft Biometrics Project/test/GEI/img.png', cv2.IMREAD_GRAYSCALE)
# #print(test_image)
#
#
#
#
# for file in dir_arr:
#
#     for i in os.listdir(pos_file+file):
#
#
#             try:
#
#                 print('train->'+file+':'+i)
#
#                 img = cv2.imread(pos_file+file+'/'+i, cv2.IMREAD_GRAYSCALE)
#                 #print(img)
#
#                 image_arr[0].append(img)
#                 image_arr[1].append(file)
#
#
#             except Exception as e:
#                 print(str(e))
#     print("loop 1 end")
#
#     for i in os.listdir(test_file+file):
#
#
#
#             try:
#
#                 print('test->' + file + ':' + i)
#
#                 img = cv2.imread(test_file + file + '/' + i, cv2.IMREAD_GRAYSCALE)
#                 # print(img)
#
#                 test_arr[0].append(img)
#                 test_arr[1].append(file)
#
#
#             except Exception as e:
#                 print(str(e))
#
# print(image_arr)
# #RESIZE TRAIN AND TEST IMAGES
# X_train = np.array(image_arr[0][0]).astype(dtype= int)
# nsamples , nx , ny = X_train.shape
# X_train = X_train.reshape((nsamples,nx*ny))
# y_train = np.array(image_arr[1])
#
# X_test = np.array(test_arr[0][0]).astype(dtype=int)
# nsamples, nx , ny = X_test.shape
# X_test = X_test.reshape((nsamples,nx*ny))
# y_test = np.array(test_arr[1])
#
# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# # Fitting Kernel SVM to the Training set
#
# from sklearn.svm import SVC
# classifier = SVC(kernel = 'rbf', random_state = 1, degree = 3)
# classifier.fit(X_train, y_train)
#
# predict = classifier.predict(X_test)
# print(predict)
#
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, predict)
#
# #accuracy score
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(y_test,predict)
# print(cm)
# print(accuracy)

