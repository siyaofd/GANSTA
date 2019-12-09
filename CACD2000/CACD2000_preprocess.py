import os
import glob
import csv
import shutil
import random

'''
https://bcsiriuschen.github.io/CARC/

Directory Structure:
CACD2000-Full
    + image1
    + image2
    + ...
'''
# Seed random number because we want random number
random.seed(9033)
images_dir = './CACD2000-Full'
person_limit = 3 # randomly select 3 images max from a person

base_dir2 = './bucket_root2'
description_file = 'description_all.csv'
bucket_description_file = 'description_bucket.csv'
age_bucket_size = 5


def get_age_name(file_name):
    symbols = file_name.split("_")
    # expect age_name_digit.jpg so split length has to be >= 3
    if len(symbols) < 3:
        return None
    age = int(symbols[0])
    name = "_".join(symbols[1:-1])
    return (age, name)

files = os.listdir(images_dir)
age_name_map = dict()
i = 0

# List all files, put into age
for file in files:
    if i % 1000 == 0:
        print("processed " + str(i))
    i += 1
    file_path = os.path.join(images_dir, file)
    age, name = get_age_name(file)

    if age not in age_name_map:
        age_name_map[age] = dict()
    
    if name not in age_name_map[age]:
        age_name_map[age][name] = list()
    age_name_map[age][name].append(file_path)

final_age_name_map = dict()
for age in age_name_map.keys():
    final_age_name_map[age] = list()

    for name, person_list in age_name_map[age].items():
        if len(person_list) < person_limit:
            final_age_name_map[age] = final_age_name_map[age] + person_list
        else:
            tmp_list = random.sample(population = person_list, k = person_limit)
            final_age_name_map[age] = final_age_name_map[age] + tmp_list


## Remove Existing output directory, and make output directory
if os.path.exists(base_dir2):
    print("Removing existing folders " + base_dir2)
    g = input("Really existing folders ??? " + base_dir2)
    g = input("Really existing folders ?????? " + base_dir2)
    print('Removing !!! ' + base_dir2)
    shutil.rmtree(base_dir2, ignore_errors=False, onerror=None)
os.makedirs(base_dir2)

TO_COPY = True

## This is Separate Gender Age bucket, contains both male and female
# Output age buckets. We group multiple ages into 1 bucket
bucket_first_row = ['age range', 'count', 'max_per_celeb ' + str(person_limit)]
bucket_output_list = list()
bucket_output_list.append(bucket_first_row)
with open(bucket_description_file, 'w') as f:
    writer = csv.writer(f)

    total_count = 0
    for x in range(20):
        print("Copying range: " + str(x))

        start = x * age_bucket_size + 1
        end = (x + 1) * age_bucket_size + 1
        key_list = [ i for i in range(start, end)]

        bucket_name = str(key_list[0]) + '-' + str(key_list[-1])
        bucket_path = os.path.join(base_dir2, bucket_name)

        # Create bucket folder
        os.makedirs(bucket_path)

        range_count = 0
        # Copy age in range to bucket
        for key in key_list:
            if key not in final_age_name_map:
                continue

            # first element is path
            for fff in final_age_name_map[key]:
                if TO_COPY:
                    shutil.copy(fff, bucket_path)

            range_count += len(final_age_name_map[key])
            # Write to csv row:
        total_count += range_count
        bucket_output_list.append([bucket_name, str(range_count)])
    
    writer.writerows(bucket_output_list)

print('Total to get: ', total_count)