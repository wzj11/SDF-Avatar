import cv2
import face_alignment
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image, height, width, datas=False, alpha_=False):
    image = np.array(image)
    alpha = image[..., 3] > 0
    H, W = alpha.shape
    # get the bounding box of alpha
    y, x = np.where(alpha)
    y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
    x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
    image_center = image[y0:y1, x0:x1]
    # resize the longer side to H * 0.9
    H, W, _ = image_center.shape
    if H > W:
        W = int(W * (height * 0.9) / H)
        H = int(height * 0.9)
    else:
        H = int(H * (width * 0.9) / W)
        W = int(width * 0.9)
    image_center = np.array(Image.fromarray(image_center).resize((W, H)))
    # pad to H, W
    start_h = (height - H) // 2
    start_w = (width - W) // 2
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[start_h : start_h + H, start_w : start_w + W] = image_center
    if alpha_:
        return image
    image = image.astype(np.float32) / 255.0
    new_alpha = image[:, :, 3]
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = (image * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)

    def save():
        # 原图大小
        H0, W0 = alpha.shape
        # bbox（含你那 ±1 的边界扩张）
        ys, xs = np.where(alpha)
        y0, y1 = max(ys.min()-1, 0), min(ys.max()+1, H0)
        x0, x1 = max(xs.min()-1, 0), min(xs.max()+1, W0)

        # 裁剪后大小
        Hc, Wc = y1 - y0, x1 - x0

        # 目标画布 (height, width)，等比缩放到“占 0.9 的最长边”
        s = min(0.9*height / Hc, 0.9*width / Wc)   # 这和你分支写法等价
        H1, W1 = int(round(Hc * s)), int(round(Wc * s))

        start_h = (height - H1) // 2
        start_w = (width  - W1) // 2
        # np.savez('/home/wzj/project/TRELLIS/src/tutils/trans.npz', s=s, x0=x0, y0=y0, w=start_w, h=start_h)
        print('save!!!!!!')
    save()

    if not datas:
        return image, new_alpha
    else:
        return start_h, H, start_w, W, y0, y1, x0, x1


def png_(name):
    output_path = '/public/home/wangzhijun/SDF-Avatar/outputs_ners'
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    # path = f'{output_path}/{name}/images/output_reference.png'
    x = Image.open(f'/public/home/wangzhijun/Nersemble/{name}/ori_imgs/000000.png')
    i = preprocess_image(x, 768, 768, True, alpha_=True)
    m = i[..., -1]
    m[m<250] = 0
    i[..., -1] = m
    
    print(i)
    cv2.imwrite(f'/public/home/wangzhijun/SDF-Avatar/outputs_ners/{name}/images/alpha.png', i[..., :-1])

def main(name):
    output_path = '/public/home/wangzhijun/SDF-Avatar/outputs_ners'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    path = f'{output_path}/{name}/images/output_reference.png'
    x = Image.open(f'/public/home/wangzhijun/Nersemble/{name}/ori_imgs/000000.png')
    start_h, H, start_w, W, y0, y1, x0, x1 = preprocess_image(x, 768, 768, True)
    # img_n = np.array(img_n)

    image = cv2.imread(f'{output_path}/{name}/normals/normal_0_wzj.png', cv2.IMREAD_UNCHANGED)
    image = image[..., :3] * (image[..., 3:] / 255.) + (1 - image[:, :, 3:4] / 255.) * 127
    image = image.astype(np.uint8)
    # print(image[0, 0])
    # exit()
    image_center = image[y0:y1, x0:x1]
    # print(image_center.shape)
    image_center = np.array(Image.fromarray(image_center).resize((W, H)))


    img_n = np.ones((768, 768, 3), dtype=np.uint8) * 127
    img_n[start_h : start_h + H, start_w : start_w + W] = image_center

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    preds = fa.get_landmarks(img)[0]
    preds = preds.astype(np.int32)
    # print(preds)
    

    # radius = 10                # 半径 (决定点的大小)
    # color = (0, 0, 255)        # 颜色 (B, G, R) -> 这里是红色
    # thickness = -1

    # landmks = np.zeros((768, 768, 3), np.uint8)
    # for x, y in preds:
    #     cv2.circle(img, (x, y), radius, color, thickness)
    # cv2.imwrite('test_land.png', img)
    y, x = preds[27]
    # y -= 20
    x -= 20
    length = min(x, 256)
    length1 = min(y, 256)
    length = min(length, length1)
    new_img = img[x - length:x + length, y - length:y + length]
    new_img_n = img_n[x - length:x + length, y - length:y + length]

    if length < 256:
        new_img = cv2.resize(new_img, (512, 512))
        new_img_n = cv2.resize(new_img_n, (512, 512))

    cv2.imwrite(f'{output_path}/{name}/images/croped.png', new_img[..., ::-1])
    cv2.imwrite(f'{output_path}/{name}/images/croped_normal.png', new_img_n[..., ::-1])
    cv2.imwrite(f'{output_path}/{name}/images/croped_all.png', 0.4 * new_img_n[..., ::-1] + 0.6 * new_img[..., ::-1])
    cv2.imwrite(f'{output_path}/{name}/images/croped_all_n.png', 0.6 * new_img_n[..., ::-1] + 0.4 * new_img[..., ::-1])


name_list = ['287', '290', '294', '285', '283', '282', '274', '262', '259', '253', '249', '248', '247', '240', '239', '238', '232', '227', '223', '220', '216', '212', '200', '199', '188', '179', '165', '149', '140', '139', '128', '115', '112', '108', '106', '104', '098', '083', '076', '075', '074', '071', '060', '055', '040', '036', '031', '030', '290', '294', '301', '306', '307', '313', '314', '315', '318', '319', '320', '326', '331', '371']


if __name__ == '__main__':
    # name = 'nersemble_vids_326.mp4'
    # main(name)
    # exit()
    paths = ['nersemble_vids_' + name + '.mp4' for name in name_list]
    for name in paths:
        print(f'processing {name}')
        png_(name)
        # try:
        #     print(f'processing {name}')
        #     png_(name)
        # except Exception as e:
        #     print(e)
