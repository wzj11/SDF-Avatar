import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

def worker(name):
    path = os.path.join('/public/home/wangzhijun/Nersemble', name, 'ori_imgs')
    ids = os.listdir(path)
    files = [os.path.join(path, id_) for id_ in ids if id_.endswith('.jpg')]
    files = sorted(files)
    a = ' '.join(files)
    # print(a)
    args_ = ['ffmpeg', '-framerate', '24', '-pattern_type', 'glob', '-i', f'{path}/*.jpg', '-b:v', '10M', '-pix_fmt', 'yuv420p', f'{path}/output.mp4']
    subprocess.run(args_)
    # print(files)

def main():
    path = '/public/home/wangzhijun/Nersemble'
    names = os.listdir(path)
    for name in names:
        print(f'processing {name}')
        worker(name)

if __name__ == '__main__':
    # worker('nersemble_vids_326.mp4')
    main()
    