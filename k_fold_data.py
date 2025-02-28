import os
import shutil
from sklearn.model_selection import KFold
import threading

def fold_all(fold, train_index, test_index):
    train_fold_dir = os.path.join(destination_directory, f"fold_{fold}/images/train")
    test_fold_dir = os.path.join(destination_directory, f"fold_{fold}/images/test")

    lable_train_fold_dir = os.path.join(label_destination_directory, f"fold_{fold}/labels/train")
    lable_test_fold_dir = os.path.join(label_destination_directory, f"fold_{fold}/labels/test")

    os.makedirs(train_fold_dir, exist_ok=True)
    os.makedirs(test_fold_dir, exist_ok=True)
    os.makedirs(lable_train_fold_dir, exist_ok=True)
    os.makedirs(lable_test_fold_dir, exist_ok=True)

    train_files = [file_names[i] for i in train_index]
    test_files = [file_names[i] for i in test_index]

    # print(len(train_files))

    for file in train_files:
        source_path = os.path.join(source_directory, file)
        destination_path = os.path.join(train_fold_dir, file)
        labelFile = file[0:len(file) - 4] + '.txt'
        label_source_path = os.path.join(label_source_directory, labelFile)
        label_destination_path = os.path.join(lable_train_fold_dir, labelFile)

        # print(destination_path)
        # print(label_destination_path)
        shutil.copy(source_path, destination_path)
        shutil.copy(label_source_path, label_destination_path)
        # break

    # print(len(test_files))
    for file in test_files:
        source_path = os.path.join(source_directory, file)
        destination_path = os.path.join(test_fold_dir, file)
        labelFile = file[0:len(file) - 4] + '.txt'
        label_source_path = os.path.join(label_source_directory, labelFile)
        label_destination_path = os.path.join(lable_test_fold_dir, labelFile)

        # print(destination_path)
        # print(label_destination_path)
        shutil.copy(source_path, destination_path)
        shutil.copy(label_source_path, label_destination_path)
        # break

threads = []

source_directory = "G:/Coding/Python/UGV Projects/Musuam_Phone_Detection/augmented_images/images"
destination_directory = "G:/Coding/Python/UGV Projects/Musuam_Phone_Detection/augmented_images"

label_source_directory = "G:/Coding/Python/UGV Projects/Musuam_Phone_Detection/augmented_images/labels"
label_destination_directory = "G:/Coding/Python/UGV Projects/Musuam_Phone_Detection/augmented_images"


# Create destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

file_names = [f for f in os.listdir(source_directory) if f.endswith(".jpg")]
file_names.sort()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
countFold=1
for fold, (train_index, test_index) in enumerate(kf.split(file_names)):
    thread = threading.Thread(target=fold_all, args=(fold, train_index, test_index))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()



    print(countFold)
    countFold+=1
    #break
print("Data split and saved successfully.")