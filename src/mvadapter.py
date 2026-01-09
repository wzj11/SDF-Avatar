import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from models.smplx.smplx import SMPLX
import trimesh
from trellis.representations import MeshExtractResult
import cubvh
import cv2
from utils.utils_render import get_ndc_proj_matrix, render
import argparse
import yaml

device = 'cuda:0'
num_betas = 300
num_expression_coeffs = 100
model_path = 'models/smplx/SMPLX2020'
smplx = SMPLX(num_betas=num_betas, num_expression_coeffs=num_expression_coeffs, model_path=model_path)
smplx.to(device)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)

args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

root_path = os.path.dirname(
    os.path.dirname(__file__)
)

print(cfg.keys())
OUTPUT_PATH = cfg['output_dir']

def to_device(*args, **kwargs):
    func = lambda x: (x.to(device) if isinstance(x, torch.Tensor) else torch.from_numpy(x).float().to(device) if isinstance(x, np.ndarray) else x)
    # print(kwargs)
    # exit()
    if kwargs:
        return list(map(func, kwargs.values()))

def process(name):
    # name = 'pipe'
    track_path = f'inputs/{name}/body_track/smplx_track.pth'
    smplx_params = torch.load(track_path)
    cam_para, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, shape, expr, global_orient, transl = to_device(**smplx_params)
    batch_size = body_pose.shape[0]
    lhand_pose = smplx.left_hand_pose.expand(batch_size, -1)
    rhand_pose = smplx.right_hand_pose.expand(batch_size, -1)
    leye_pose = smplx.leye_pose.expand(batch_size, -1)
    reye_pose = smplx.reye_pose.expand(batch_size, -1)

    shape1 = shape[0:1].clone()
    jaw_pose1 = torch.tensor(
        [
            [0.4, 0., 0.]
        ]
    ).to(shape)
    lhand_pose1 = lhand_pose[0:1].clone()
    rhand_pose1 = rhand_pose[0:1].clone()
    leye_pose1 = leye_pose[0:1].clone()
    reye_pose1 = reye_pose[0:1].clone()
    expr1 = expr[0:1].clone()
    # body_pose1 = body_pose[0:1].clone()
    body_pose1 = torch.zeros_like(body_pose[0:1])
    # global_orient1 = global_orient[0:1].clone()
    global_orient1 = torch.zeros_like(global_orient[0:1])
    transl1 = transl[0:1].clone()

    shape = torch.cat([shape, shape1], dim=0)
    jaw_pose = torch.cat([jaw_pose, jaw_pose1], dim=0)
    lhand_pose = torch.cat([lhand_pose, lhand_pose1], dim=0)
    rhand_pose = torch.cat([rhand_pose, rhand_pose1], dim=0)
    leye_pose = torch.cat([leye_pose, leye_pose1], dim=0)
    reye_pose = torch.cat([reye_pose, reye_pose1], dim=0)
    expr = torch.cat([expr, expr1], dim=0)
    body_pose = torch.cat([body_pose, body_pose1], dim=0)
    global_orient = torch.cat([global_orient, global_orient1], dim=0)
    transl = torch.cat([transl, transl1], dim=0)



    output = smplx.forward(
            betas=shape,
            jaw_pose=jaw_pose,
            left_hand_pose=lhand_pose,
            right_hand_pose=rhand_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            expression=expr,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            with_rott_return=True
        )
    smplx_v = output['vertices'].detach().cpu().numpy()
    vv = smplx_v[-1]
    m = trimesh.Trimesh(vertices=vv, faces=smplx.faces)
    sub = np.load('params/mouth.npy')
    sub_1 = np.load('params/smplx_faces_no_eye.npy')
    sub_m = m.submesh([sub])[0]
    new_sub_m = m.submesh([sub_1])[0]
    # sub_m.export('src/filter/mouth.obj')

    
    # trellis = trimesh.load(f'exp/{name}/objects/mouth_22.obj')
    # trellis_v = torch.from_numpy(trellis.vertices).to(device)
    # trellis_f = torch.from_numpy(trellis.faces).to(device)
    
    import utils3d
    from utils3d.torch import matrix_to_quaternion, quaternion_to_matrix

    # utils3d.io.write_ply('src/filter/mouth.ply', points.detach().cpu().numpy())
    # l = trellis_mask.nonzero().flatten().cpu().numpy()
    
    # trellis.submesh([l])[0].export('src/filter/filtered.obj')

    motion = np.load(f'outputs/{name}/slats/motion_sdf.npz')
    face_id = motion['face_id']
    uvw = motion['uvw']
    index = output.vertices.shape[1] - 1092
    mask = ~(smplx.faces > index).any(axis=-1)
    ff = smplx.faces[mask]
    motion_id = ff[face_id]
    deformation = (smplx_v - smplx_v[0:1])[:, :-1092]
    deformation = deformation[:, motion_id, :]
    trans = np.load(f'outputs/{name}/params/trans.npz')
    kt = np.load(f'outputs/{name}/params/kt.npz')
    k, t = list(kt.values())
    s, R, T = list(trans.values())
    s = torch.from_numpy(s).to(device)
    R = quaternion_to_matrix(torch.from_numpy(R).to(device))
    T = torch.from_numpy(T).to(device)
    k = torch.from_numpy(k).to(device)
    t = torch.from_numpy(t).to(device)
    deformation = np.einsum('bkij, bki->bkj', deformation, np.tile(uvw, (deformation.shape[0], 1, 1)))
    deformation = s.cpu().numpy() * k.cpu().numpy() * deformation @ R[0].transpose(-2, -1).cpu().numpy()
    print(deformation[-1].shape)
    path = f'outputs/{name}/params/v_pos_i.pt'
    p = torch.load(path)
    new_p = (p).detach().cpu().numpy() + deformation[-1]
    # utils3d.io.write_ply('src/filter/cubes.ply', (p).detach().cpu().numpy() + deformation[15])
    # print(p.shape)
    # exit()
    indices = torch.load(f'outputs/{name}/params/marching.pt')
    my_p = new_p[indices.detach().cpu().numpy()].reshape(-1, 8, 3).mean(axis=1)
    my_p = (my_p - T.cpu().numpy()) @ R[0].cpu().numpy() / s[0].cpu().numpy()
    my_p = (my_p - t.cpu().numpy()) / k.cpu().numpy()
    # utils3d.io.write_ply('src/filter/coords.ply', my_p)

    BVH1 = cubvh.cuBVH(torch.from_numpy(sub_m.vertices).to(device), torch.from_numpy(sub_m.faces).to(device))
    BVH2 = cubvh.cuBVH(torch.from_numpy(vv).to(device), torch.from_numpy(smplx.faces).to(device))

    sub_mm = new_sub_m.vertices.copy()
    sub_mm[..., 2] = 0
    BVH3 = cubvh.cuBVH(torch.from_numpy(sub_mm).to(device), torch.from_numpy(new_sub_m.faces).to(device))


    x_min = sub_m.vertices[..., 0].min()
    x_max = sub_m.vertices[..., 0].max()
    y_min = sub_m.vertices[..., 1].min()
    y_max = sub_m.vertices[..., 1].max()

    my_p = torch.from_numpy(my_p).to(device)

    my_pp = my_p.clone()
    my_pp[..., 2] = 0

    dis, face_id, uvw = BVH1.unsigned_distance(my_p, True)
    dis1, face_id1, uvw1 = BVH2.unsigned_distance(my_p, True)
    dis2, face_id2, uvw2 = BVH3.unsigned_distance(my_pp, True)

    mask1 = (dis1 > 0.001)
    mask = (dis < 0.05)
    mask2 = (dis2 > 0.001)
    mask_x = ((my_p[..., 0] > x_min) & (my_p[..., 0] < x_max))
    mask_y = ((my_p[..., 1] > y_min) & (my_p[..., 1] < y_max))
    mm = (mask & mask1 & mask_x & mask_y & mask2)
    points = my_p[~mm]

    torch.save(mm, f'outputs/{name}/params/marching_mask.pt')
    # utils3d.io.write_ply('src/filter/coords_new.ply', points.detach().cpu().numpy())

def mvadapter(name):
    # name = 'pipe'
    print('doing mvadapter')
    from PIL import Image
    inputdir = cfg['input_dir']


    # process image
    path = f'{inputdir}/{name}/ori_imgs/{cfg["img_path"]}.png'
    image = Image.open(path)
    assert image.mode == 'RGBA'
    image = np.array(image)
    alpha = image[..., 3] > 0
    height = width = 768
    H0, W0 = alpha.shape
        # bbox（含你那 ±1 的边界扩张）
    ys, xs = np.where(alpha)
    y0, y1 = max(ys.min()-1, 0), min(ys.max()+1, H0)
    x0, x1 = max(xs.min()-1, 0), min(xs.max()+1, W0)
    
    Hc, Wc = y1 - y0, x1 - x0
    s = min(0.9*height / Hc, 0.9*width / Wc)
    H1, W1 = int(round(Hc * s)), int(round(Wc * s))
    h = (height - H1) // 2
    w = (width  - W1) // 2
    # exit()

    np.savez(f'{OUTPUT_PATH}/{name}/params/imgtrans.npz', s=s, x0=x0, y0=y0, w=w, h=h)







    track_path = f'{inputdir}/{name}/body_track/smplx_track.pth'
    img_path = f'{inputdir}/{name}/ori_imgs/{cfg["img_path"]}.png'
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    mesh = trimesh.load(f'{OUTPUT_PATH}/{name}/objects/single.obj')
    
    smplx_params = torch.load(track_path)
    cam_para, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, shape, expr, global_orient, transl = to_device(**smplx_params)


    batch_size = body_pose.shape[0]
    lhand_pose = smplx.left_hand_pose.expand(batch_size, -1)
    rhand_pose = smplx.right_hand_pose.expand(batch_size, -1)
    leye_pose = smplx.leye_pose.expand(batch_size, -1)
    reye_pose = smplx.reye_pose.expand(batch_size, -1)
    output = smplx.forward(
            betas=shape,
            jaw_pose=jaw_pose,
            left_hand_pose=lhand_pose,
            right_hand_pose=rhand_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            expression=expr,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            with_rott_return=True
        )
    index = output.vertices.shape[1] - 1092
    mask = (smplx.faces > index).any(axis=-1)
    ff = smplx.faces[mask]
    vv = output.vertices[0].detach().cpu().numpy()
    new_v = vv[ff]
    ff -= ff.min()
    ff += mesh.faces.max() + 1
    new_v = np.concatenate([mesh.vertices, vv[-1092:]], axis=0)
    new_f = np.concatenate([mesh.faces, ff], axis=0)
    my_mesh = trimesh.Trimesh(vertices=new_v, faces=new_f)
    # my_mesh.export('src/mvadapter.obj')
    smplx_m = MeshExtractResult(
        vertices=torch.from_numpy(my_mesh.vertices).to(device).float(),
        faces=torch.from_numpy(my_mesh.faces).to(device).int()
    )

    # tt = np.load('src/tutils/trans.npz')
    # s, x0, y0, w, h = list(tt.values())

    # w = w.item()
    # h = h.item()
    new_cam = cam_para.new_zeros(cam_para.shape)
    print(cam_para)
    new_cam[..., 0] = s * cam_para[..., 0]
    new_cam[..., 1] = s * cam_para[..., 1]
    new_cam[..., 2] = s * (cam_para[..., 2] - x0) + w
    new_cam[..., 3] = s * (cam_para[..., 3] - y0) + h
    print(new_cam)
    proj = get_ndc_proj_matrix(new_cam[0:1], [768, 768])
    rota1 = torch.Tensor(
        [
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ]
    ).to(device)
    smplx_m.face_normal = smplx_m.face_normal @ rota1.T
    z_mean = smplx_m.vertices[..., -1].mean().item()

    smplx_m.vertices[..., -1] = smplx_m.vertices[..., -1] - z_mean
    extrinsic = np.load('params/exs.npy')
    rot_mvadapter = np.load('params/rot_mvadapter.npy')
    extrinsic = torch.from_numpy(extrinsic).to(device)
    rot_mvadapter = torch.from_numpy(rot_mvadapter).to(device)
    extrinsic[:, 2, 3] = z_mean
    smplx_dicts = render(smplx_m, extrinsic, proj[0], HW=[768, 768], return_types=["mask", "normal", "depth", "pos"], rot_mvadapter=rot_mvadapter)

    for key in smplx_dicts.keys():
        if hasattr(smplx_dicts[key], 'shape'):
            print('key: ', key, ' shape: ', smplx_dicts[key].shape)

    img1 = smplx_dicts['normal'].permute(1, 2, 0, 3).reshape(-1, 768, extrinsic.shape[0] * 768)
    img2 = smplx_dicts['pos'].permute(1, 2, 0, 3).reshape(-1, 768, extrinsic.shape[0] * 768)
    m = smplx_dicts['mask'][:, None].permute(1, 2, 0, 3).reshape(-1, 768, extrinsic.shape[0] * 768)
    
    img1 = img1.permute(1, 2, 0)
    img2 = img2.permute(1, 2, 0)
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/objects/debug_normal.png', img1.detach().cpu().numpy()[..., ::-1] * 255.)
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/objects/debug_pos.png', img2.detach().cpu().numpy()[..., ::-1] * 255.)

if __name__ == '__main__':
    name = '085.mp4'
    # process(name)
    mvadapter(name)