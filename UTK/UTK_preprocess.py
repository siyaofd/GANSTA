import os
import glob
import csv
import shutil
'''
Data Description: 
https://susanqq.github.io/UTKFace/

[age]_[gender]_[race]_[date&time].jpg

[age] is an integer from 0 to 116, indicating the age
[gender] is either 0 (male) or 1 (female)
[race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
[date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace
'''
ETHNICITY = {
    0 : 'white',
    1 : 'black',
    2 : 'asian',
    3 : 'indian',
    4 : 'other'
}
ETHNICITY_LIST = ['white', 'black', 'asian', 'indian', 'other']
GENDER_LIST = ['male', 'female']
GENDER = set(GENDER_LIST)



images_dir = './images'
base_dir1 = './bucket_root1'
base_dir2 = './bucket_root2'
base_dir3 = './bucket_root3'

description_file = 'description_all.csv'
bucket_description_file = 'description_bucket.csv'
bucket_description_file2 = 'description_bucket2.csv'

age_bucket_size = 5

# Create CSV File Row
partial_row = list()
for e in ETHNICITY_LIST:
    for g in GENDER_LIST:
        partial_row.append(e + ' ' + g)

description_row = ['age', 'male', 'female'] + partial_row
bucket_description_row = list(description_row)
bucket_description_row[0] = 'age range'


def create_map():
    d = dict()
    d['male'] = dict()
    d['female'] = dict()
    for _, v in ETHNICITY.items():
        d['male'][v] = list()
        d['female'][v] = list()
    return d 

# Expect only filename
def extract_info(file_name):
    symbols = file_name.split("_")
    if (len(symbols) != 4):
        return None
    age = int(symbols[0])
    gender = 'male' if int(symbols[1]) == 0 else 'female'
    eth = int(symbols[2])
    if eth not in ETHNICITY.keys():
        return None
    return (age, gender, ETHNICITY[eth], eth)

# List all files, populate the map
files = os.listdir(images_dir)
age_gender_ethnicity = dict()
i = 0
for file in files:
    if i % 1000 == 0:
        print("processed " + str(i))
    i += 1

    info = extract_info(file)
    if info:
        age = info[0]
        gender = info[1]
        ethnicity = info[2]
        if age not in age_gender_ethnicity:
            age_gender_ethnicity[age] = create_map()
        age_gender_ethnicity[age][gender][ethnicity].append(file)


# Get stats for all ages
output_list = list()
output_list.append(description_row)
with open(description_file, 'w') as f:
    writer = csv.writer(f)

    for age in sorted(age_gender_ethnicity.keys()):
        total = 0
        part1 = list()
        part2 = list()
        part_row = list()
        m = dict()
        m['male'] = 0
        m['female'] = 0
        for e in ETHNICITY_LIST:
            for g in GENDER_LIST:
                num = len(age_gender_ethnicity[age][g][e])
                #print(g, e, num)
                part2.append(num)
                total += num
                m[g] += num
        output_list.append([age, m['male'], m['female']] + part2)
    writer.writerows(output_list)


## Remove Existing output directory, and make output directory

if os.path.exists(base_dir3):
    print("Removing existing folders " + base_dir3)
    g = input("Really existing folders ??? " + base_dir3)
    g = input("Really existing folders ?????? " + base_dir3)
    print('Removing !!! ' + base_dir3)
    shutil.rmtree(base_dir3, ignore_errors=False, onerror=None)
os.makedirs(base_dir3)


## This is Separate Gender Age bucket, contains both male and female
# Output age buckets. We group multiple ages into 1 bucket
bucket_first_row = ['age range', 'male count', 'female count']
bucket_output_list = list()
bucket_output_list.append(bucket_first_row)
with open(bucket_description_file2, 'w') as f:
    writer = csv.writer(f)

    for x in range(20):
        print("Copying range: " + str(x))

        start = x * age_bucket_size + 1
        end = (x + 1) * age_bucket_size + 1
        key_list = [ i for i in range(start, end)]

        bucket_name = str(key_list[0]) + '-' + str(key_list[-1])
        bucket_path = os.path.join(base_dir3, bucket_name)
        bucket_male_path = os.path.join(bucket_path, 'male')
        bucket_female_path = os.path.join(bucket_path, 'female')

        # Create bucket folder
        os.makedirs(bucket_path)
        os.makedirs(bucket_male_path)
        os.makedirs(bucket_female_path)


        male_cnt = 0
        female_cnt = 0
        # Copy age in range to bucket
        for key in key_list:
            if key not in age_gender_ethnicity:
                continue
            # first element is path
            for eth, eth_list in age_gender_ethnicity[key]['male'].items():
                if eth is not 'white':
                    continue
                for file_name in eth_list:
                    to_move_path = os.path.join(images_dir, file_name)
                    shutil.copy(to_move_path, bucket_male_path)
                
                male_cnt += len(eth_list)

            for eth, eth_list in age_gender_ethnicity[key]['female'].items():
                if eth is not 'white':
                    continue
                for file_name in eth_list:
                    to_move_path = os.path.join(images_dir, file_name)
                    shutil.copy(to_move_path, bucket_female_path)
                female_cnt += len(eth_list)
            

        # Write to csv row:
        bucket_output_list.append([bucket_name, str(male_cnt), str(female_cnt)])
    writer.writerows(bucket_output_list)