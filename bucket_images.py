import os
import re
import shutil

current_dir = './images'

def copy_file_folder(select, dictionary, dest_dir):
	for key in select:
		for file in dictionary[key]:
			shutil.copyfile(current_dir+'/'+file, dest_dir + '/' + file.replace(".JPG", ".jpg"))

# Images in this toy set is of the format: 081A04.JPG. 
# Where 081 is person ID, 04 is age. A is the delimiter.
# For now, we define 2 bucket: young_folder where age <=12 and old_folder where age >= 18
young_dict = dict()
old_dict = dict()
# List all input files
files = os.listdir(current_dir)
for file in files:
	res = re.match(r'([0-9]+)[^\W\d_]+([0-9]+).JPG', file)
	if res:
		person_id, age = int(res[1]), int(res[2])
		if person_id not in young_dict:
			young_dict[person_id] = list()
		if person_id not in old_dict:
			old_dict[person_id] = list()

		if age <= 12:
			young_dict[person_id].append(file)
		if age >= 18:
			old_dict[person_id].append(file)
	else:
		print("No Match for ", file)

young_persons = set(young_dict.keys())
old_persons = set(old_dict.keys())
# We only want pictures of person who is in both young and old bucket
common = set.intersection(young_persons, old_persons)

# Copy files to final bucket.
copy_file_folder(common, young_dict, './young_folder')
copy_file_folder(common, old_dict, './old_folder')
