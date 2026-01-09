import cv2
import argparse
import numpy as np
# argparser = argparse.ArgumentParser()
# argparser.add_argument('--name', type=str)

# args = argparser.parse_args()

input_dir = 'input_ffhq'

def main(name):
    # name = args.name
    # print(name)
    mask = cv2.imread(f'input_ffhq/{name}/seg_masks/000000.png', cv2.IMREAD_UNCHANGED)
    print(f'loading mask from "input_ffhq/{name}/seg_masks/000000.png"')
    img = cv2.imread(f'input_ffhq/{name}/ori_imgs/000000.jpg', cv2.IMREAD_UNCHANGED)
    print(img.shape)
    print(f'loading image from "input_ffhq/{name}/ori_imgs/000000.jpg"')
    new_img = np.concatenate(
        [
            img,
            mask[..., None]
        ],
        axis=-1
    )
    cv2.imwrite(f'input_ffhq/{name}/ori_imgs/000000.png', new_img)


import os

if __name__ == '__main__':
    # main()
    paths = os.listdir('input_ffhq')
    print(paths)
    # try:
    for name in paths:
        try:
            main(name)
        except Exception as e:
            print(e)
        