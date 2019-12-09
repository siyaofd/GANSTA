import argparse as argparse

import csv
import os
metadata = 'meta.csv'

def cal_file_size(input_name, output_name):
	with open(input_name, 'r') as f:
		reader = csv.reader(f)
		mylist = list(reader)
		first = True
		output_list = list()
		i = 0
		for row in mylist:
			if first:
				first = False
				output_list.append(row.append('size(bytes)'))
				continue
			filename = row[2]
			stat = os.stat(filename)
			size = stat[6]
			row.append(str(size))
			output_list.append(row)
			if i % 1000:
				print("1000 x " + str(i/1000) + " processed")
			i += 1

		with open(output_name,'w') as outf:
			writer = csv.writer(outf)
			writer.writerows(output_list)


# Default take at least 8k image. We use 8k as proxy for images above 200pixels
def filter_by_file_size(input_name, output_name, size = 8000):
	with open(input_name, 'r') as f:
		reader = csv.reader(f)
		mylist = list(reader)
		first = True
		output_list = list()
		for row in mylist:
			if first:
				first = False
				output_list.append(row)
				continue
			ss = row[3]
			if int(ss) > size:
				output_list.append(row)

		with open(output_name,'w') as outf:
			writer = csv.writer(outf)
			writer.writerows(output_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-name', required=False)
    parser.add_argument('--existing', dest='existing', action='store_true')

    parser.add_argument('--output-name', required=True)
    parser.add_argument('--min-size', required=False, type=int, default=8000)
    parser.set_defaults(existing=True)

    args = parser.parse_args()
    if (args.input_name and 'csv' not in args.input_name) or 'csv' not in args.output_name:
    	exit('input [' + args.input_name if args.input_name else 'input' + '] or output [' + args.output_name + '] isnt csv')

    # Already has size file
    if args.existing:
    	filter_by_file_size(args.input_name, args.output_name, args.min_size)
    else:
    	cal_file_size(args.input_name, args.output_name)

    #main(args.db_path, photo_dir=args.photo_dir, output_dir=args.output_dir, min_score=args.min_score, img_size=args.img_size)