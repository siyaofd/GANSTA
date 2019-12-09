import argparse as argparse
import csv
import os
import shutil
import random
from PIL import Image
import numpy as np

random.seed(9001)

# Check is valid file and correct format
def validate_file(file_name):
    return os.path.isfile(file_name) and ('.jpg' in file_name or '.png' in file_name)

# Detect grayscale image. From stack overflow
def is_grey_scale(img_path):

    img = Image.open(img_path).convert('RGB')
    w,h = img.size
    ilist = random.sample(range(w), 50)
    jlist = random.sample(range(h), 50)
    for i, j in zip(ilist, jlist):
        r, g, b = img.getpixel((i,j))
        if r != g or g != b or r != b:
            return False
    return True

def iteration_progress(i, mmm):
    if i % mmm == 0:
        print("Seeing {} th file".format(i))
    return i + 1

# Move grayscale image to another folder
def remove_grayscale(input_dir, output_dir):
    file_list = list()
    i = 0
    for file_name in os.listdir(input_dir):
        i = iteration_progress(i, 1000)
        file_path = os.path.join(input_dir, file_name)
        if validate_file(file_path) and is_grey_scale(file_path):
            file_list.append(file_path)

    print("About to move {} grayscale images".format(len(file_list)))
    i = 0
    # Copy grayscale image to destination
    for fff in file_list:
        i = iteration_progress(i, 100)
        i += 1
        shutil.move(fff, output_dir)
    
    print("Moved {} grayscale images from {} to {}.".format(len(file_list), input_dir, output_dir))

def crop(input_dir, output_dir, xr, yr):
    i = 0
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if not validate_file(file_path):
            continue
        if i % 100 == 0:
            print("Cropped {} images.".format(i))
        i += 1
        im = Image.open(file_path)
        width, height = im.size
        left = int(width * xr)
        right = int(width * (1 - xr))
        top = int(height * xr)
        bottom = int(height * (1 - yr))
        im2 = im.crop((left, top, right, bottom))

        output_path = os.path.join(output_dir, file_name)
        im2.save(output_path, 'JPEG')
    print("Cropped {} images with xr: {}, yr: {} from {} saved to {}".format(i, xr, yr, input_dir, output_dir))
        
def rgb_mean_inrange(file_path, rgb_range):
    im = Image.open(file_path)
    im_arr = np.asarray(im)
    im_arr_mean = np.mean(im_arr)
    if rgb_range[0] < im_arr_mean and im_arr_mean < rgb_range[1]:
        return True
    return False

def select_files(input_dir, output_dir, number, rgb_range):
    tmp_list = os.listdir(input_dir)
    if len(tmp_list) > number:
        print("More Files than Requested, selecting {} out of {}.".format(number, len(tmp_list)))
        tmp_list = random.sample(population = tmp_list, k = number)
    i = 0  
    file_list = list()
    for file_name in tmp_list:
        i = iteration_progress(i, 1000)
        file_path = os.path.join(input_dir, file_name)
        if validate_file(file_path) and rgb_mean_inrange(file_path, rgb_range):
            file_list.append(file_path)

    i = 0
    # Copy file to destination
    for fff in file_list:
        i = iteration_progress(i, 100)
        shutil.copy(fff, output_dir)
    
    print("Copied {} files from {} to {}".format(len(file_list), input_dir, output_dir))

# Create ratio of test dataset. Move from original directory to new directory.
# if ratio = 0.2, will be 20% test
def test_split(input_dir, output_dir, ratio):
    file_list = list()
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        file_list.append(file_path)
    
    assert(ratio <= 1 and ratio >= 0)
    assert(file_list)

    number = int(len(file_list) * ratio)
    tmp_list = random.sample(population = file_list, k = number)
    # Copy file to destination
    for fff in tmp_list:
        shutil.move(fff, output_dir)
    
    print("Moved {} files from {} to {}, ratio: {}".format(len(tmp_list), input_dir, output_dir, ratio))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        '--input : input directory; --output : output directory', \
        '--cmd : crop, select, split, rmgray; --th : threshold, number or ratio', \
        '--xr : horizontal crop ratio; --yr : vertical crop ratio', \
        '--range : rgb mean range'
    )
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cmd', required=True)
    parser.add_argument('--th', required=True)
    parser.add_argument('--xr', required=False, type=float, default = 0.25)
    parser.add_argument('--yr', required=False, type=float, default = 0.25)
    parser.add_argument('--range', required=False, default = '10,240')
    args = parser.parse_args()
    command = args.cmd
    input_dir = args.input
    output_dir = args.output

    xr = args.xr
    yr = args.yr
    threshold = int(args.th) if args.th.isdigit() else float(args.th)

    rgb_lower, rgb_upper = args.range.strip().split(',')
    rgb_lower = int(rgb_lower)
    rgb_upper = int(rgb_upper)
    rgb_range = (rgb_lower, rgb_upper)
    print("brightness range: ", rgb_range)
    assert(rgb_lower <= rgb_upper)
    assert(0 <= rgb_lower)
    assert(rgb_upper <= 256)
    assert(threshold >= 0)
    assert(xr >= 0 and xr <= 1 and yr >= 0 and yr <= 1)
    assert(os.path.exists(input_dir))
    # Make sure directory is fine
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    if command == 'crop':
        crop(input_dir, output_dir, xr, yr)
    elif command == 'select':
        select_files(input_dir, output_dir, threshold, rgb_range)
    elif command == 'split':
        test_split(input_dir, output_dir, threshold)
    elif command == 'rmgray':
        remove_grayscale(input_dir, output_dir)
    else:
        exit("Invalid command " + args.cmd)
