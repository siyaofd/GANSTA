import os
import glob
import csv
import shutil
'''
Data Description:
https://github.com/JingchunCheng/All-Age-Faces-Dataset
Directory Structure:
Aligned_face
    + images...

gender description in train.txt and val.txt file. 1 is male, 0 is female

'''
GENDER_LIST = ['male', 'female']
GENDER = set(GENDER_LIST)

train_file = 'train.txt'
validation_file = 'val.txt'

images_dir = './Aligned_face'
base_dir2 = './bucket_root2'
description_file = 'description_all.csv'
bucket_description_file = 'description_bucket.csv'
age_bucket_size = 5
min_image_size = 8000

# Create CSV File Row
description_row = ['age', 'male', 'female']

# List all files, populate the map
files = os.listdir(images_dir)
age_gender_map = dict()
i = 0


# Parse filename for age
def parse_file(filename):
    p1, _ = filename.split(".")
    _ , age = p1.split('A')
    return int(age)

# Parse gender
def get_gender(w):
    if int(w) == 0:
        return 'female'
    elif int(w) == 1:
        return 'male'
    else:
        raise Exception("Neither male or female")

# Return map of file name to gender
def load_info(train_file, val_file):
    file_gender_map = dict()
    with open(train_file, 'r') as f1:
        for line in f1.readlines():
            elems = line.strip().split(' ')
            if elems and len(elems) > 1:
                file_gender_map[str(elems[0])] = get_gender(elems[1])
    with open(val_file, 'r') as f1:
        for line in f1.readlines():
            elems = line.strip().split(' ')
            if elems and len(elems) > 1:
                file_gender_map[str(elems[0])] = get_gender(elems[1])
    
    return file_gender_map

filename_gender_map = load_info(train_file, validation_file)

age_gender_map = dict()
# Populate age gender map
for filename in os.listdir(images_dir):
    if '.jpg' in filename:
        age = parse_file(filename)
        if age not in age_gender_map:
            age_gender_map[age] = dict()
            age_gender_map[age]['male'] = list()
            age_gender_map[age]['female'] = list()
        gen = filename_gender_map[filename]
        path = os.path.join(images_dir, filename)
        age_gender_map[age][gen].append(path)

## Remove Existing output directory, and make output directory
if os.path.exists(base_dir2):
    print("Removing existing folders " + base_dir2)
    g = input("Really existing folders ??? " + base_dir2)
    g = input("Really existing folders ?????? " + base_dir2)
    print('Removing !!! ' + base_dir2)
    shutil.rmtree(base_dir2, ignore_errors=False, onerror=None)
os.makedirs(base_dir2)


## This is Separate Gender Age bucket, contains both male and female
# Output age buckets. We group multiple ages into 1 bucket
bucket_first_row = ['age range', 'male count', 'female count']
bucket_output_list = list()
bucket_output_list.append(bucket_first_row)
with open(bucket_description_file, 'w') as f:
    writer = csv.writer(f)

    for x in range(20):
        print("Copying range: " + str(x))

        start = x * age_bucket_size + 1
        end = (x + 1) * age_bucket_size + 1
        key_list = [ i for i in range(start, end)]

        bucket_name = str(key_list[0]) + '-' + str(key_list[-1])
        bucket_path = os.path.join(base_dir2, bucket_name)
        bucket_male_path = os.path.join(bucket_path, 'male')
        bucket_female_path = os.path.join(bucket_path, 'female')

        # Create bucket folder
        os.makedirs(bucket_path)
        os.makedirs(bucket_male_path)
        os.makedirs(bucket_female_path)

        # Copy age in range to bucket
        male_count = 0
        female_count = 0
        for key in key_list:
            if key not in age_gender_map:
                continue
            # first element is path
            for f1 in age_gender_map[key]['male']: 
                shutil.copy(f1, bucket_male_path)
            for f2 in age_gender_map[key]['female']: 
                shutil.copy(f2, bucket_female_path)

            male_count += len(age_gender_map[key]['male'])
            female_count += len(age_gender_map[key]['female'])

        # Write to csv row:
        bucket_output_list.append([bucket_name, str(male_count), str(female_count)])
    
    writer.writerows(bucket_output_list)