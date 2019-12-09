import os
import glob
import csv
import shutil
'''
Data Description:
https://afad-dataset.github.io/
There are 164,432 well-labeled photos in the AFAD dataset. It consist of 63,680 photos for female as well as 100,752 photos for male, and the ages range from 15 to 40.

Directory Structure:
AFAD-Full
    + 15
        + 111(male)
        + 112(female)
    + 16
        +...
    + ...
    + 75
        +...

 Preprocess needed:
    * Contains a lot of thumb nails, need to filter out too small images.
    We use 8k as proxy for big.       

'''
GENDER_LIST = ['male', 'female']
GENDER = set(GENDER_LIST)



images_dir = './AFAD-Full'
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

# Populate age gender map
for dir1 in os.listdir(images_dir):
    # Directory must be digit
    if dir1.isdigit():
        age = int(dir1)
        if age not in age_gender_map:
            age_gender_map[age] = dict()
            age_gender_map[age]['male'] = list()
            age_gender_map[age]['female'] = list()
        gender_path = os.path.join(images_dir, dir1)
        for dir_gender in os.listdir(gender_path):
            person_path = os.path.join(gender_path, dir_gender)
            # 111 is male, 112 is female
            if dir_gender != '111' and dir_gender != '112':
                continue
            for file_name in os.listdir(person_path):
                # Only look at picture file
                if '.jpg' not in file_name:
                    continue
                final_path = os.path.join(person_path, file_name)
                stat = os.stat(final_path)
                size = int(stat[6]) # Size of image in bytes
                if size < min_image_size:
                    continue
                if dir_gender == '111':
                    age_gender_map[age]['male'].append(final_path)
                else:
                    age_gender_map[age]['female'].append(final_path)



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