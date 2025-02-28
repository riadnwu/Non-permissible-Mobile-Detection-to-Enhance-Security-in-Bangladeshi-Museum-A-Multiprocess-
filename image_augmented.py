import cv2
import albumentations as alb
import os
import shutil

# Define the path to the input directory containing JPG files
images_input_directory = 'G:/Coding/Python/UGV Projects/Musuam_Phone_Detection/data/images/train'
labels_input_directory = 'G:/Coding/Python/UGV Projects/Musuam_Phone_Detection/data/labels/train'

# Define the output directory to save augmented images
output_directory = 'G:/Coding/Python/UGV Projects/Musuam_Phone_Detection/augmented_images'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)
os.makedirs(os.path.join(output_directory, 'images/'), exist_ok=True)
os.makedirs(os.path.join(output_directory,'labels/'), exist_ok=True)

# Define augmentation transformations
augmentations = [
    ("Blur", alb.Blur((5, 10), p=0.7)),
    ("HueSaturationValue", alb.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.9)),
    ("RandomRain", alb.RandomRain(p=0.5)),
    ("RandomBrightnessContrast", alb.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=(-0.3, 0.3), p=0.7)),
    ("GaussNoise", alb.GaussNoise(var_limit=(500, 1000), p=0.7))
]

# Get a list of all JPG files in the input directory
jpg_files = [f for f in os.listdir(images_input_directory) if f.endswith(".jpg")]



# Loop through each JPG file, apply augmentations, and save the resulting images
countAll=1
jpg_files.sort()
for jpg_file in jpg_files:
    image_path = os.path.join(images_input_directory, jpg_file)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640,384))
    ##image = cv2.resize(image, (640, 640))
    #print(input_directory+'images/'+jpg_file)
    #print(output_directory+'images/'+jpg_file)

    label_old_path=labels_input_directory+'/'+jpg_file[0:len(jpg_file)-4]+'.txt'
    #print(label_old_path)
    #print(output_directory+'labels/'+jpg_file[0:len(jpg_file)-4]+'.txt')
    #shutil.copy(input_directory+'images/'+jpg_file, output_directory+'images/'+jpg_file)
    shutil.copy(label_old_path,output_directory+'/labels/'+jpg_file[0:len(jpg_file)-4]+'.txt')
    cv2.imwrite( output_directory+'/images/'+jpg_file, image)

    for augmentation_name, transform in augmentations:
        augmented_image = transform(image=image)['image']
        temp_file_name= jpg_file[0:len(jpg_file)-4]+'_'+augmentation_name
        output_file_path = os.path.join(output_directory+'/images',temp_file_name+'.jpg')
        #print(output_directory+'labels/'+temp_file_name+'.txt')
        #print(output_file_path)
        shutil.copy(label_old_path, output_directory+'/labels/'+temp_file_name+'.txt')
        cv2.imwrite(output_file_path, augmented_image)
    print(countAll)
    countAll+=1


    #break

print("Augmented images saved successfully.")