
import argparse as argparse
import csv
import os
import shutil
import glob
from distutils.dir_util import copy_tree


input_file = 'size_12k.csv'
gender_set = set(['male', 'female']) # Assume binary

description_file = 'description.csv'
bucket_description_file = 'bucket_description.csv'

base_dir = './bucket_root'
min_age = 0
max_age = 100
age_bucket_size = 5
age_gender_map = dict()

with open(input_file, 'r') as f:
    reader = csv.reader(f)
    mylist = list(reader)

    for row in mylist[1:]:
        age = int(row[0])
        gender = row[1]
        path = row[2]
        size = int(row[3])
        if age not in age_gender_map:
            age_gender_map[age] = dict()
            age_gender_map[age]['male'] = list()
            age_gender_map[age]['female'] = list()
        if gender not in gender_set:
            print("Unexpected gender " + gender)
        age_gender_map[age][gender].append((path, size))


## Remove Existing output directory, and make output directory
if os.path.exists(base_dir):
    print("Removing existing folders " + base_dir)
    g = input("Really existing folders ??? " + base_dir)
    g = input("Really existing folders ?????? " + base_dir)
    print('Removing !!! ' + base_dir)
    shutil.rmtree(base_dir, ignore_errors=False, onerror=None)
os.makedirs(base_dir)




# Write stats for each age/gender final output. Age is 1 year increment
first_row = ['age', 'male count', 'female count']
output_list = list()
output_list.append(first_row)
with open(description_file, 'w') as f:
    writer = csv.writer(f)
    age_array = sorted(age_gender_map.keys())
    for key in age_array:
        row = [str(key), str(len(age_gender_map[key]['male'])), str(len(age_gender_map[key]['female']))]
        output_list.append(row)
    writer.writerows(output_list)

#sorted_age_range = filter(lambda x : x >= min_age and x <= max_age, sorted(age_gender_map.keys()))


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
        bucket_path = os.path.join(base_dir, bucket_name)
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
            # first element is path
            for f1 in age_gender_map[key]['male']: 
                shutil.copy(f1[0], bucket_male_path)
            for f2 in age_gender_map[key]['female']: 
                shutil.copy(f2[0], bucket_female_path)

            male_count += len(age_gender_map[key]['male'])
            female_count += len(age_gender_map[key]['female'])

        # Write to csv row:
        bucket_output_list.append([bucket_name, str(male_count), str(female_count)])
    writer.writerows(bucket_output_list)











