import cv2
from glob import glob
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

# def worker(path):
def read_png(path, ext='.png', num_threads=8):
    # imgs = []
    def worker(p):
        # nonlocal imgs
        img = cv2.imread(p)
        return img
        # imgs.append(img)
    print(path)
    paths = sorted(glob(f'{path}/*{ext}'))
    with ThreadPoolExecutor(num_threads) as Pool:
        imgs = list(Pool.map(worker, paths))
    print(len(imgs))
    imgs = np.stack(imgs, axis=0)
    return imgs

def preprocess(name, path, img_path):
    img = cv2.imread(f'{path}/{name}/ori_imgs/{img_path}.png', cv2.IMREAD_UNCHANGED)
    print(f'{path}/{name}/ori_imgs/{img_path}.png')
    img, alpha = img[..., :3], img[..., -1:]
    print(img.shape)
    print(alpha.shape)
    parse = cv2.imread(f'{path}/{name}/parsing/{img_path}.png')

    ffm = ((parse == 2) | ((parse > 5) & (parse < 14)))
    face_mask = (ffm.astype(np.int8) * (parse != 0)).astype(np.int8)
    dim_factor = 0.8
    inverse_mask = 1 - face_mask
    dim_multiplier = face_mask + inverse_mask * dim_factor
    dimmed_img = img * dim_multiplier
    dimmed_img = np.concatenate(
        [
            dimmed_img, alpha
        ],
        axis=-1,
    )
    return Image.fromarray(dimmed_img.astype(np.uint8), mode='RGBA')

if __name__ == '__main__':
    x = preprocess('002_02')
    # print(x.shape)
    

    
