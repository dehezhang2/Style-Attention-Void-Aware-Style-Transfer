import glob
import os
from PIL import Image
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='../../training_data/content_set/val2014/', type=str)
args = parser.parse_args()

img_paths = glob.glob(os.path.join(args.dataset, "*.*"))
total_count = len(img_paths)


size_list = []
for i, image_path in enumerate(img_paths):
    size_list.append(os.path.getsize(image_path))
    if i % 1000 == 0:
        print("First round processing: " + str(i) + "-th image")

size_list = sorted(size_list, reverse=True)
total_count = len(size_list)
threshold = size_list[1000]
print("Threshold: " + str(threshold))

illegal_count = 0
large_count = 0
corrupted_count = 0

for i, image_path in enumerate(img_paths):
    if os.path.getsize(image_path) > threshold:
        print("large image [{}/{}]".format(i, total_count))
        os.remove(image_path)
        large_count += 1
        continue
    try:
        img = Image.open(image_path)
        img = np.asarray(img)

        if len(np.shape(img)) != 3 or np.shape(img)[2] != 3:
            print(image_path + " image format illegal[{}/{}]".format(i, total_count))
            os.remove(image_path)
            illegal_count += 1
    except:
        print("large or corrupted image [{}/{}]".format(i, total_count))
        os.remove(image_path)
        corrupted_count += 1
    
print("Illegal Format: " + str(illegal_count))
print("Large: " + str(large_count))
print("Corrupted: " + str(corrupted_count))
print("Deleted " + str(illegal_count + large_count + corrupted_count))
print("Clean Done")
