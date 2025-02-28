from PIL import Image
import os
from glob import glob
#path='./Bangla-Money-Dataset-master/All_0_10_20_50_100/'
path='G:/Coding/Python/UGV Projects/Musuam_Phone_Detection/Dataset_Phone/All Data/Mobile Back Image With Hand'
filename_list = glob(os.path.join(path, "*.jp*"))
savePath = 'G:/Coding/Python/UGV Projects/Musuam_Phone_Detection/Dataset_Phone/Process_Data'
countSaveFile = glob(os.path.join(savePath, "*.jpg"))
countSaveFile=int(len(countSaveFile)/3)
print(countSaveFile)
count=1001+countSaveFile
for filename in filename_list:
    filename=filename[len(path):len(filename)]
    print(filename)
    im = Image.open(path+filename)
    #angle45 = 45
    #angle_45 = -45
    angle90 = 90
    angle_90 = -90
    #im45 = im.rotate(angle45)
    #imMin45 = im.rotate(angle_45)
    im90 = im.rotate(angle90)
    imMin90 = im.rotate(angle_90)
    im.save(savePath+'/phone_'+str(count)+'.jpg')
    #im45.save(savePath+'/'+str(count)+'_45.jpg')
    #imMin45.save(savePath+'/'+ str(count) + '__45.jpg')
    im90.save(savePath + '/phone_' + str(count) + '_90.jpg')
    imMin90.save(savePath + '/phone_' + str(count) + '__90.jpg')
    #im.save('New_Money_Data/' + filename + '.jpg')
    #out.save("New_Money_Data/" + filename + '_180.jpg')
    print(count)
    count=count+1