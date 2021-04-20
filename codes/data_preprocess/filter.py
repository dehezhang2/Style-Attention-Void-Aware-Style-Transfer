import glob
import os
from PIL import Image
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", dest='../../training_data/content_set/val2014/', type=str)
args = parser.parse_args()
PIXEL_PER_BYTE = 5

if __name__ == "__main__":
    img_paths = glob.glob(os.path.join(args.dataset, "*.*"))

    total_count = len(img_paths)
    for i, image_path in enumerate(img_paths):
        if i > 29303:
            continue
        if i % 200 == 0:
            print("Processing image: [{}/{}]".format(i, total_count))
        # print("image " + str(i) + " size: " + str(int(os.path.getsize(image_path))) + " path: " + image_path)
        # if int(os.path.getsize(image_path)) > (89478485 / PIXEL_PER_BYTE):
        #     print(image_path + " image is too big[{}/{}]".format(i, total_count))
        #     os.remove(image_path)
        #     continue
        try:
            img = Image.open(image_path)
            img = np.asarray(img)

            if len(np.shape(img)) != 3 or np.shape(img)[2] != 3:
                print(image_path + " image format illegal[{}/{}]".format(i, total_count))
                os.remove(image_path)
        except:
            print("large image")
            os.remove(image_path)

    print("clean done")
