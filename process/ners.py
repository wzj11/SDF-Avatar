import os


name_list = ['287', '290', '294', '285', '283', '282', '274', '262', '259', '253', '249', '248', '247', '240', '239', '238', '232', '227', '223', '220', '216', '212', '200', '199', '188', '179', '165', '149', '140', '139', '128', '115', '112', '108', '106', '104', '098', '083', '076', '075', '074', '071', '060', '055', '040', '036', '031', '030', '290', '294', '301', '306', '307', '313', '314', '315', '318', '319', '320', '326', '331', '371']

import subprocess

def main():
    paths = [f'/public/home/wangzhijun/Nersemble/nersemble_vids_{name}.mp4' for name in name_list]
    with open('nersemble_files', 'w') as f:
        for path in paths:
            f.write(path + '\n')
    # for path in paths:
    #     args_ = ['cp', '-r', f'{path}', '/public/home/wangzhijun/Ners/']
    #     subprocess.run(args_)
    print(paths)


if __name__ == '__main__':
    main()