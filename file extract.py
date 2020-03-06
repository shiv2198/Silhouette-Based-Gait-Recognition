import os
import shutil
import numpy

walk_angle = '018'
src = 100
child = 1

girl_arr = ['008','012','016','021','022','031','036','043','044','058','060','082','097','100','104','107','108','121','121']

boy_arr = ['001','002','003','004','010','006','007','008','009','011','012','013','014','015','016','017','018','019','020']
test_arr = ['066','067','068','069','070','071','072','073','074','075','076','077','078','079','080','081','082','084','085','086','060']
par_path = 'D:/SHIVANSH/Papers/Soft Biometrics/gait/Dataset/GaitDatasetB-silh/DatasetB/silhouettes/'

#child_path = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/dataset/'+walk_angle+'/girl/'
child_path = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/dataset/'+walk_angle+'/boy/'
#child_path = 'D:/SHIVANSH/Machine Learning/Soft Biometrics Project/test/'+walk_angle+'/img/'



while(child!=21):

    #par = par_path+girl_arr[child-1]+'/nm-01/'+walk_angle+'/'
    #print(par)
    par = par_path + boy_arr[child-1] + '/nm-01/'+walk_angle+'/'
    #par = par_path + test_arr[child - 1] + '/nm-01/'+walk_angle+'/'
    dest = child_path+str(child)+'/'


    for file in os.listdir(par):
    #for file in36 os.listdir(dest):


        src_file = os.path.join(par, file)
        dst_file = os.path.join(dest, file)
        shutil.copy(src_file,dst_file)
        #shutil.move(dst_file,'C:/Users/admin/Desktop/delete')

    src+=1
    child+=1
#'C:/Users/admin/Desktop/delete'




