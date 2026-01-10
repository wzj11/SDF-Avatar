import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import argparse
import yaml
from utils.utils_geo import wzj_final, densify
from torch.optim import Adam
from lpips import LPIPS
from models.smplx.smplx import SMPLX
from src.networks import myNet
from utils.utils_io import read_png
from utils.utils_render import get_ndc_proj_matrix, render
from trellis.utils.grad_clip_utils import AdaptiveGradClipper
from trellis.trainers.utils import *
from utils.utils_train import *
import math
import trimesh
from trellis.representations import MeshExtractResult
from trellis.modules import sparse as sp
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes
import cv2
from scipy.ndimage import binary_dilation
from scipy.ndimage import generate_binary_structure
import cubvh
from trellis.representations.mesh import SparseFeatures2Mesh
from teeth.myteeth.verts_displacement.geometry import GeoModel
from Geo_render.render_utils import MeshRenderer
from utils3d.torch import matrix_to_quaternion, quaternion_to_matrix
from tqdm import tqdm
import subprocess
from datetime import datetime
import subprocess
from PIL import Image


device = 'cuda:0'
num_betas = 300
num_expression_coeffs = 100
model_path = 'models/smplx/SMPLX2020'
smplx = SMPLX(num_betas=num_betas, num_expression_coeffs=num_expression_coeffs, model_path=model_path)
smplx.to(device)
model = SparseFeatures2Mesh(res=256)

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
    

def color(name):
    # inputdir = cfg['input_dir']

    # i, seg_mask = preprocess_image(Image.open(f'{inputdir}/{name}/ori_imgs/000000.png'), 768, 768)
    # my_img = np.concatenate(
    #     [
    #         np.array(i)[..., ::-1],
    #         seg_mask[..., None] * 255.
    #     ],
    #     axis=-1
    # )
    # cv2.imwrite(f'/public/home/wangzhijun/Ners/{name}/ori_imgs/000000.png', my_img)
    # return

    inputdir = cfg['input_dir']
    os.makedirs(f'{OUTPUT_PATH}/{name}/color', exist_ok=True)
    os.makedirs(f'{OUTPUT_PATH}/{name}/color_new', exist_ok=True)
    _, seg_mask = preprocess_image(Image.open(f'{inputdir}/{name}/ori_imgs/000000.png'), 768, 768)
    parse = cv2.imread(f'{inputdir}/{name}/parsing/000000.png', cv2.IMREAD_UNCHANGED)
    H_origin, W_origin = parse.shape
    ffm = ((parse == 2) | ((parse > 5) & (parse < 14)))
    # face_mask = (ffm.astype(np.uint8) * (parse!=0)).astype(np.uint8)
    face_mask = ((parse!=0)).astype(np.uint8)

    # cv2.imwrite(f'face_mask.png', face_mask * 255.)
    # _, seg = preprocess_image(Image.open(f'{inputdir}/{name}/parsing/000000.png'), 768, 768)
    # print(seg.shape)
    start_h, H, start_w, W, y0, y1, x0, x1 = preprocess_image(Image.open(f'{inputdir}/{name}/ori_imgs/000000.png'), 768, 768, True)

    

    image_center = face_mask[y0:y1, x0:x1]
    image_center = np.array(Image.fromarray(image_center).resize((W, H)))
    face_mask = np.zeros((768, 768), dtype=np.uint8)
    face_mask[start_h:start_h+H, start_w:start_w+W] = image_center
    structure = generate_binary_structure(2, 1)
    x_mask = binary_dilation((1 - face_mask) * 255., structure=structure, iterations=4)
    
    face_mask = torch.from_numpy(face_mask).to(device)

    seg_mask = torch.from_numpy(seg_mask).to(device)
    from torch.optim import Adam
    # from PIL import Image
    from torch.nn.functional import huber_loss
    gts = Image.open(f'{OUTPUT_PATH}/{name}/images/output.png')
    gt_0 = Image.open(f'{OUTPUT_PATH}/{name}/images/output_reference.png')
    gt_0 = np.array(gt_0).astype(np.float32) / 255.
    gt_0 = torch.from_numpy(gt_0).to(device).clamp(0, 1)
    gt_0 = gt_0.permute(2, 0, 1)
    gts = np.array(gts).astype(np.float32) / 255.
    gts = torch.from_numpy(gts).to(device).clamp(0, 1)
    gts = gts.reshape(768, 6, 768, 3).permute(1 ,3, 0, 2)

    # gt_new = Image.open(f'{inputdir}/{name}/ori_imgs/000043.jpg')
    # gt_new = np.array(gt_new).astype(np.float32) / 255.
    # gt_new = torch.from_numpy(gt_new).to(device).clamp(0, 1)
    # gt_new = gt_new.permute(2, 0, 1)

    # parse_new = cv2.imread(f'{inputdir}/{name}/parsing/000043.png', cv2.IMREAD_UNCHANGED)
    # ffm = ((parse_new == 2) | ((parse_new > 5) & (parse_new < 14)))
    # face_mask_new = (ffm.astype(np.uint8) * (parse_new!=0)).astype(np.uint8)
    # face_mask_new = torch.from_numpy(face_mask_new).to(device)

    track_path = f'{inputdir}/{name}/body_track/smplx_track.pth'
    smplx_params = torch.load(track_path)
    cam_para, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, shape, expr, global_orient, transl = to_device(**smplx_params)

    mesh = trimesh.load(f'{OUTPUT_PATH}/{name}/objects/single.obj')

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
    smplx_v = output['vertices'].squeeze().detach().cpu().numpy()
    index = output.vertices.shape[1] - 1092
    mask_ = (smplx.faces > index).any(axis=-1)
    ff = smplx.faces[mask_]
    vv = output.vertices[0].detach().cpu().numpy()
    new_v = vv[ff]
    ff -= ff.min()
    ff += mesh.faces.max() + 1 
    new_v = np.concatenate([mesh.vertices, vv[-1092:]], axis=0)
    new_f = np.concatenate([mesh.faces, ff], axis=0)
    my_mesh = trimesh.Trimesh(vertices=new_v, faces=new_f)

    smplx_m = MeshExtractResult(
        vertices=torch.from_numpy(my_mesh.vertices).to(device).float(),
        faces=torch.from_numpy(my_mesh.faces).to(device).int()
    )
    # ff = smplx.faces[mask]
    # vv = output.vertices[0].detach().cpu().numpy()
    # new_v = vv[ff]
    # ff -= ff.min()
    # ff += mesh.faces.max() + 1


    feats_n = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_0_new.pt').to(device)

    feats = torch.load(f'{OUTPUT_PATH}/{name}/params/delta_geo_mouth_new.pt').to(device)
    coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_0_new.pt').to(device)
    model = SparseFeatures2Mesh(res=256, use_color=True)
    z_mean = smplx_m.vertices[..., -1].mean().item()
    
    extrinsics = np.load('params/exs.npy')
    extrinsics = torch.from_numpy(extrinsics).to(device)
    extrinsics[:, 2, 3] = z_mean

    tt = np.load(f'{OUTPUT_PATH}/{name}/params/imgtrans.npz')
    s, x0, y0, w, h = list(tt.values())
    s = s.item()
    x0 = x0.item()
    y0 = y0.item()
    w = w.item()
    h = h.item()
    new_cam = cam_para.new_zeros(cam_para.shape)
    new_cam[..., 0] = s * cam_para[..., 0]
    new_cam[..., 1] = s * cam_para[..., 1]
    new_cam[..., 2] = s * (cam_para[..., 2] - x0) + w
    new_cam[..., 3] = s * (cam_para[..., 3] - y0) + h
    proj = get_ndc_proj_matrix(new_cam[0:1], [768, 768])

    extrinsics1 = torch.tensor(
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ]
    ).to(device)
    proj1 = get_ndc_proj_matrix(cam_para[0:1], [H_origin, W_origin])

    
    with torch.no_grad():
        temp_d = model(None, dcoords=coords, dfeats=feats_n, debug=True)
        sdf_d = temp_d.reshape((257, 257, 257))
        sdf_d = sdf_d.detach().cpu().numpy()
        sdf_mask = wzj_final(
            sdf_d,
        )
        # exit()
        sdf_mask = sdf_mask.reshape(-1)
        sdf_mask = torch.from_numpy(sdf_mask).to(device)
    with torch.no_grad():
        temp_d = model(None, dcoords=coords, dfeats=feats_n, debug=True)

        temp_d[sdf_mask] *= -1
        sdf_d = temp_d.reshape((257, 257, 257))
        sdf_d = sdf_d.detach().cpu().numpy()
        sdf_mask1 = wzj_final(
            sdf_d,
        )
        sdf_mask1 = sdf_mask1.reshape(-1)
        sdf_mask1 = torch.from_numpy(sdf_mask1).to(device)
        print(sdf_mask1.sum())
    sdf_mask = sdf_mask | sdf_mask1
    m_mask = torch.load(f'{OUTPUT_PATH}/{name}/params/marching_mask.pt').to(device)
    indices = torch.load(f'{OUTPUT_PATH}/{name}/params/indices.pt')
    with torch.no_grad():
        mesh = model(None, dcoords=coords, dfeats=feats, training=True, mask=sdf_mask, marching_mask=m_mask, indices=indices)

        mesh2smplx(name, mesh)
        z_mean = mesh.vertices[..., -1].mean().item()
        print(mesh.vertex_attrs.shape)
        # z_mean = mesh.vertices[..., -1].mean().item()
        mesh.vertices[..., -1] = mesh.vertices[..., -1] - z_mean

        dicts = render(mesh, extrinsics, proj[0], HW=[768, 768], return_types=['color', 'mask'])

        mask = 1 - dicts['mask'].detach().cpu().numpy()
        masks = []
        structure = generate_binary_structure(2, 1)

        for m in mask:
            masks.append(binary_dilation(m, structure=structure, iterations=8))
        mask = np.stack(masks, axis=0)
        mask = 1 - torch.from_numpy(mask).to(dicts['mask'])
        # mesh.vertices[..., -1] = mesh.vertices[..., -1] - z_mean
    # delta = mesh.vertex_attrs.new_zeros((mesh.vertex_attrs.shape[0] + 1092, 6), requires_grad=True)
    delta_eye = mesh.vertex_attrs.new_zeros((1092, 6), requires_grad=True)
    # light_params = torch.tensor([1, 0., 0., 0.]).to(device).requires_grad_(True)

    light_params = torch.zeros((9, 3), device='cuda', requires_grad=True)

    with torch.no_grad():
        light_params[0, :] = 1.77 
        light_params[1:, :] = 0.0


    # delta = feats.new_zeros(feats.shape, requires_grad=True)
    delta = feats.new_zeros((*feats.shape[:-1], 48), requires_grad=True)
    test = feats.new_empty((*feats.shape[:-1], 48)).normal_()
    
    optimizer = Adam(
        [
            {'params': [delta], 'lr': 1e-2},
            {'params': [light_params], 'lr': 1e-2},
            {'params': [delta_eye], 'lr': 5e-2}
        ]
    )
    # optimizer = Adam([delta], lr=5e-2)
    
    for epoch in range(301):
        optimizer.zero_grad()
        s, R, T, k, t = mesh2smplx(name, return_=True)
        ff = torch.cat(
            [
                feats[..., :-48].detach(),
                test + delta
            ],
            dim=-1
        ).to(device)
        # part1 = feats[..., :-48].detach()
        # part2 = feats[..., -48:].detach() + delta
        # ff = torch.cat([part1, part2], dim=-1)
        # ff = feats.clone().detach().requires_grad_(True)
        # ff[..., -48:] = feats[..., -48:].clone() + delta
        # mesh = model(None, dcoords=coords, dfeats=feats, training=False, mask=sdf_mask, marching_mask=m_mask, indices=indices)
        mesh = model(None, dcoords=coords, dfeats=ff, training=True, mask=sdf_mask, marching_mask=m_mask, indices=indices)
        mesh2smplx(name, mesh)

        with torch.no_grad():
            # my_vertices = torch.cat(
            #     [
            #         mesh.vertices,
            #         output.vertices[0, -1092:],
            #     ],
            #     dim=0
            # ).to(mesh.vertices)
            # my_vertices = torch.cat(
            #     [
            #         mesh.vertices,
            #         output.vertices[0, -1092:],
            #     ],
            #     dim=0
            # ).to(mesh.vertices)
            my_vertices=mesh.vertices
            # mf = mesh.faces.new_tensor(smplx.faces[mask_])
            # mf -= mf.min()
            # mf += mesh.faces.max() + 1
            # my_faces = torch.cat(
            #     [
            #         mesh.faces,
            #         mf
            #     ],
            #     dim = 0
            # ).to(mesh.faces)
            # print(mesh.vertex_attrs.shape)
            my_faces = mesh.faces

        # my_attr = torch.cat(
        #     [
        #         mesh.vertex_attrs,
        #         torch.ones((1092, 6)).to(mesh.vertex_attrs) + delta_eye
        #     ]
        # )
        my_attr = mesh.vertex_attrs
        mesh = MeshExtractResult(
            vertices=my_vertices,
            faces=my_faces,
            vertex_attrs=my_attr
        )

        m_v = mesh.vertices * k + t
        m_v = (m_v * s[0]) @ R[0].T + T
        m_f = mesh.faces.detach().cpu().numpy()
        
        z_mean = mesh.vertices[..., -1].mean().item()
        mesh.vertices[..., -1] = mesh.vertices[..., -1] - z_mean
        # attr = mesh.vertex_attrs + delta
        # mesh.vertex_attrs = attr
        m_a = mesh.vertex_attrs.detach().cpu().numpy()[..., :3] * 255.

        m_a_clamped = np.clip(m_a, 0.0, 255.0)

        # 3. 转换为 np.uint8 (N, 3)
        m_a_uint8_rgb = m_a_clamped.astype(np.uint8)

        # 4. 添加 Alpha 通道 (N, 4)
        num_vertices = m_a_uint8_rgb.shape[0]
        alpha_channel = np.full((num_vertices, 1), 255, dtype=np.uint8)
        m_a_final = np.hstack([m_a_uint8_rgb, alpha_channel])

        # 5. 赋值给 Trimesh

        mesh1 = trimesh.Trimesh(vertices=m_v.detach().cpu().numpy(), faces=m_f, vertex_colors=m_a_final)

        # mesh1.visual.vertex_colors = m_a
        # print(m_a_final)
        # print(mesh1.visual.vertex_colors)

        dicts = render(mesh, extrinsics, proj[0], HW=[768, 768], return_types=['color', 'mask'], shading=light_params)
        # mask = dicts['mask'][:, None].detach()
        pred = dicts['color']
        loss = 0
        lambdal = [1.5, 0.75, 1.5, 0.75, 0.75, 0.5]
        for i in range(6):
            p = pred[i]
            if i == 0:
                l1_l = \
                ((p - gt_0).abs() * (mask[i] * face_mask)).sum() / ((mask[i] * face_mask) * p.shape[0] + 1e-6).sum() + \
                ((p - gts[i]).abs() * (mask[i] * seg_mask * (1 - face_mask))).sum() / ((mask[i] * seg_mask * (1 - face_mask)) * p.shape[0] + 1e-6).sum()
                # l1_l = ((p - gts[i]).abs() * (mask[i] * seg_mask)).sum() / ((mask[i] * seg_mask) * p.shape[0] + 1e-6).sum()
                # l1_l = l1_loss(pred[i] * (mask[i] * seg_mask), gts[i] * (mask[i] * seg_mask), type='color')
                ssim_l = 1 - ssim(pred[i] * (mask[i] * face_mask), gt_0 * (mask[i] * face_mask))
            else:
                l1_l = ((p - gts[i]).abs() * (mask[i])).sum() / ((mask[i]) * p.shape[0] + 1e-6).sum()
                # l1_l = l1_loss(pred[i] * mask[i], gts[i] * mask[i], type='color')
                ssim_l = 1 - ssim(pred[i] * mask[i], gts[i] * mask[i])
            # lpips_l = lpips(pred[i] * mask[i], gts[i] * mask[i])
            # loss += lambdal[i] * (l1_l + 0.2 * ssim_l + 0.2 * lpips_l)
            # if epoch > 100:
            # loss += lambdal[i] * (l1_l + 0.2 * ssim_l + 0.05 * lpips_l)
            # else:
            if i == 0:
                loss += lambdal[i] * (l1_l + 0.5 * ssim_l)
            else:
                loss += lambdal[i] * (0.2 * l1_l + 0.5 * ssim_l)
            
        l1 = light_params[1:4] # Linear
        l2 = light_params[4:9]
        reg_loss = torch.mean(l1 ** 2) * 0.01 + torch.mean(l2 ** 2) * 0.5
        loss += reg_loss
        # loss += 0.01 * (delta ** 2).mean()
        loss.backward()
        # print(f"Gradients: {light_params.grad}")
        # exit()
        optimizer.step()
        with torch.no_grad():
            # 限制环境光在 0 到 1 之间
            light_params[0].clamp_(0.0, 1.0)
            # 限制方向光项（b, c, d）的强度不要太离谱
            light_params[1:].clamp_(-1.0, 1.0)

        # if epoch == 70:
        #     mesh.vertices[..., -1:] += z_mean
        #     trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/objects/single_c.obj')    
        if epoch ==0 or epoch == 200:
            mesh1.export(f'{OUTPUT_PATH}/{name}/objects/color_{epoch}.ply')
        if epoch % 5 == 0:
            print('epoch: ', epoch, ' loss: ', loss.item())
            with torch.no_grad():
                print(delta.grad[..., :8].max())
                img = pred.permute(1, 2, 0, 3).reshape(-1, 768, extrinsics.shape[0] * 768)
                img = img.permute(1, 2, 0)
                cv2.imwrite(f'{OUTPUT_PATH}/{name}/color/color_{epoch}.png', img.detach().cpu().numpy()[..., ::-1] * 255.)
            
                if epoch == 300:
                    # feats[..., -48:] += delta
                    print(light_params)
                    torch.save(delta_eye.detach(), f'{OUTPUT_PATH}/{name}/params/delta_with_eye.pt')
                    mmmmask = (delta.abs().mean(dim=-1) < 1e-4)
                    ff[mmmmask, -48:] = feats[mmmmask, -48:]
                    torch.save(ff.detach(), f'{OUTPUT_PATH}/{name}/params/delta_with_color.pt')

                    torch.save(light_params.detach(), f'{OUTPUT_PATH}/{name}/params/light_params.pt')

    # motion = np.load(f'{OUTPUT_PATH}/{name}/slats/motion_sdf.npz')
    # face_id = motion['face_id']
    # uvw = motion['uvw']
    # ff = smplx.faces
    # motion_id = ff[face_id]
    # deformation = smplx_v[:150] - smplx_v[0:1]
    # deformation = deformation[:, motion_id, :]
    # deformation = np.einsum('bkij, bki->bkj', deformation, np.tile(uvw, (deformation.shape[0], 1, 1)))

    # trans = np.load(f'{OUTPUT_PATH}/{name}/params/trans.npz')
    # kt = np.load(f'{OUTPUT_PATH}/{name}/params/kt.npz')
    # k, t = list(kt.values())
    # s, R, T = list(trans.values())
    # s = torch.from_numpy(s).to(device)
    # R = quaternion_to_matrix(torch.from_numpy(R).to(device))
    # T = torch.from_numpy(T).to(device)
    # k = torch.from_numpy(k).to(device)
    # t = torch.from_numpy(t).to(device)
    # # R = quaternion_to_matrix(torch.from_numpy(R).to(device))
    # deformation = s.cpu().numpy() * k.cpu().numpy() * deformation @ R[0].transpose(-2, -1).cpu().numpy()
    # n_coords = torch.load(f'{OUTPUT_PATH}/{name}/params/v_pos_.pt').to(coords)


    # ff_new = torch.load(f'{OUTPUT_PATH}/{name}/params/delta_with_color.pt')

    # iters = 100
    # for epoch in range(iters + 1):
    #     optimizer.zero_grad()
    #     s, R, T, k, t = mesh2smplx(name, return_=True)
    #     # ff = torch.cat(
    #     #     [
    #     #         feats[..., :-48].detach(),
    #     #         test.detach() + delta
    #     #     ],
    #     #     dim=-1
    #     # ).to(device)
    #     ff = ff_new
    #     mesh = model(None, dcoords=coords, dfeats=ff.detach(), training=True, mask=sdf_mask, marching_mask=m_mask, indices=indices)
    #     mesh2smplx(name, mesh)

    #     mesh1 = model(None, dcoords=coords, dfeats=ff.detach(), v_a=torch.from_numpy(deformation[44]).to(dtype=torch.float32, device=device), training=True, mask=sdf_mask, n_coords=n_coords, marching_mask=m_mask, indices=indices)
    #     mesh2smplx(name, mesh1)

    #     with torch.no_grad():
    #         my_vertices=mesh.vertices
    #         my_faces = mesh.faces
    #     my_attr = mesh.vertex_attrs
    #     mesh = MeshExtractResult(
    #         vertices=my_vertices,
    #         faces=my_faces,
    #         vertex_attrs=my_attr
    #     )
        
    #     z_mean = mesh.vertices[..., -1].mean().item()
    #     mesh.vertices[..., -1] = mesh.vertices[..., -1] - z_mean

    #     dicts = render(mesh, extrinsics, proj[0], HW=[768, 768], return_types=['color', 'mask'], shading=light_params)
    #     dicts1 = render(mesh1, extrinsics1, proj1[0], HW=[H_origin, W_origin], return_types=['color', 'mask'], shading=light_params)
    #     pred = dicts['color']
    #     pred1 = dicts1['color'].squeeze()
    #     loss = 0
    #     lambdal = [1.1, 0.9, 1.5, 0.9, 0.9, 0.5]
    #     for i in range(6):
    #         p = pred[i]
            
    #         if i == 0:
    #             l1_l = \
    #             ((p - gt_0).abs() * (mask[i] * face_mask)).sum() / ((mask[i] * face_mask) * p.shape[0] + 1e-6).sum() + \
    #             ((p - gts[i]).abs() * (mask[i] * seg_mask * (1 - face_mask))).sum() / ((mask[i] * seg_mask * (1 - face_mask)) * p.shape[0] + 1e-6).sum()
    #             ssim_l = 1 - ssim(pred[i] * (mask[i] * face_mask), gt_0 * (mask[i] * face_mask))

    #             l1_l_new = ((pred1 - gt_new).abs() * (dicts1['mask'].squeeze() * face_mask_new)).sum() / ((dicts1['mask'].squeeze() * face_mask_new) * p.shape[0] + 1e-6).sum()
    #             ssim_l_new = 1 - ssim(pred1 * (dicts1['mask'].squeeze() * face_mask_new), gt_new * (dicts1['mask'].squeeze() * face_mask_new))

    #         else:
    #             l1_l = ((p - gts[i]).abs() * (mask[i])).sum() / ((mask[i]) * p.shape[0] + 1e-6).sum()
    #             # l1_l = l1_loss(pred[i] * mask[i], gts[i] * mask[i], type='color')
    #             ssim_l = 1 - ssim(pred[i] * mask[i], gts[i] * mask[i])
    #         # lpips_l = lpips(pred[i] * mask[i], gts[i] * mask[i])
    #         # loss += lambdal[i] * (l1_l + 0.2 * ssim_l + 0.2 * lpips_l)
    #         # if epoch > 100:
    #         # loss += lambdal[i] * (l1_l + 0.2 * ssim_l + 0.05 * lpips_l)
    #         # else:
    #         loss += lambdal[i] * (l1_l + 0.2 * ssim_l)
    #     new_loss = (l1_l_new + 0.2 * ssim_l_new)
    #     loss += 1.5 * new_loss
    #     # loss += 0.1 * (delta ** 2).mean()
    #     loss.backward()
    #     # print(f"Gradients: {light_params.grad}")
    #     # exit()
    #     optimizer.step()


    #     if epoch % 5 == 0:
    #         with torch.no_grad():
    #             # print(delta.grad[..., :8].max())
    #             img = pred.permute(1, 2, 0, 3).reshape(-1, 768, extrinsics.shape[0] * 768)
    #             img = img.permute(1, 2, 0)
    #             cv2.imwrite(f'{OUTPUT_PATH}/{name}/color_new/color_old_{epoch}.png', img.detach().cpu().numpy()[..., ::-1] * 255.)


    #         print('epoch: ', epoch, ' loss: ', loss.item(), 'new_loss: ', new_loss.item())
    #         with torch.no_grad():
    #             img = dicts1['color'].squeeze().permute(1, 2, 0)
    #             cv2.imwrite(f'{OUTPUT_PATH}/{name}/color_new/color_{epoch}.png', img.detach().cpu().numpy()[..., ::-1] * 255.)
    #     if epoch == iters:
    #         print(light_params)
    #         torch.save(light_params.detach(), f'{OUTPUT_PATH}/{name}/params/light_params.pt')
        


def preprocess_image(image: Image.Image, height, width, datas=False):
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

    args_ = ['python', '-m', 'modules.MV.scripts.inference_ig2mv_sdxl', '--name', f'{name}', '--adapter_path', 'modules/MV/models']
    subprocess.run(args_)

def motion(name):
    # print('wzj')
    # exit()
    gm = GeoModel('cuda:0')
    render_m = MeshRenderer()

    os.makedirs(f'{OUTPUT_PATH}/{name}/normals', exist_ok=True)
    # name = 'emotional_vids_confident-bearded-man-posing-on-white-background-SBV-346838037-4K.mp4'
    inputdir = cfg['input_dir']

    image = cv2.imread(f'{inputdir}/{name}/ori_imgs/{cfg["img_path"]}.png')
    HW = image.shape[:2]
    H, W = HW
    coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_0_new.pt').to(device)
    feats_n = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_0_new.pt').to(device)
    # feats = torch.load(f'src/motion_sdf/delta.pt').to(device)
    # feats = torch.load(f'src/tutils/delta_with_color.pt').to(device)
    feats = torch.load(f'{OUTPUT_PATH}/{name}/params/delta_geo_mouth_new.pt').to(device)
    feats_g = torch.load(f'{OUTPUT_PATH}/{name}/params/{pt_name}.pt').to(device)

    input = sp.SparseTensor(
        coords=coords,
        feats=feats
    )
    # 1012
    # motion = np.load(f'output/{name}/slats/motion_sdf.npz')
    motion = np.load(f'{OUTPUT_PATH}/{name}/slats/motion_sdf.npz')
    face_id = motion['face_id']
    uvw = motion['uvw']
    track_path = f'{inputdir}/{name}/body_track/smplx_track.pth'
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
    smplx_v = output['vertices'].squeeze().detach().cpu().numpy()
    index = output.vertices.shape[1] - 1092
    mask = ~(smplx.faces > index).any(axis=-1)
    ff = smplx.faces

    _, new_ff = densify(smplx_v[-1], smplx.faces)
    motion_id = new_ff[face_id]

    # v_0, _ = densify(smplx_v[0], smplx.faces) 
    # deformation = (v_last[None] - v_0[None])

    # motion_id = ff[face_id]
    # motion_id = smplx.faces[face_id]
    # pp = output.vertices[0, :-1092].detach().cpu().numpy()
    # trimesh.Trimesh(vertices=output.vertices[0].detach().cpu().numpy(), faces=smplx.faces).export('src/sm0.obj')
    # trimesh.Trimesh(vertices=output.vertices[141].detach().cpu().numpy(), faces=smplx.faces).export('src/sm1.obj')
    
    proj = get_ndc_proj_matrix(cam_para[0:1], [H, W])

    
    # smplx_v[-1092:, 2] -= 0.002

    # 1012
    # deformation = smplx_v - smplx_v[0:1] # cpu
    # deformation = (smplx_v[:150] - smplx_v[0:1])
    # # deformation = (smplx_v - smplx_v[0:1])
    # # print(deformation.shape)
    # # print(motion_id.shape)
    # deformation = deformation[:, motion_id, :] # (B ,F, 3, 3) # cpu
    # deformation = np.einsum('bkij, bki->bkj', deformation, np.tile(uvw, (deformation.shape[0], 1, 1)))
    print(motion_id.max(), 'wzj')
    # print(deformation.shape)
    # exit()
    

    trans = np.load(f'{OUTPUT_PATH}/{name}/params/trans.npz')
    kt = np.load(f'{OUTPUT_PATH}/{name}/params/kt.npz')
    k, t = list(kt.values())
    s, R, T = list(trans.values())
    s = torch.from_numpy(s).to(device)
    R = quaternion_to_matrix(torch.from_numpy(R).to(device))
    T = torch.from_numpy(T).to(device)
    k = torch.from_numpy(k).to(device)
    t = torch.from_numpy(t).to(device)
    # R = quaternion_to_matrix(torch.from_numpy(R).to(device))
    # deformation = s.cpu().numpy() * k.cpu().numpy() * deformation @ R[0].transpose(-2, -1).cpu().numpy()
    extrinsic = torch.tensor(
            [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]
            ]
        ).to(device)
    # with torch.no_grad():
    #     sdf_d = model(None, dcoords=coords, dfeats=feats_n, debug=True)
    #     sdf_d = sdf_d.reshape((257, 257, 257))
    #     sdf_d = sdf_d.detach().cpu().numpy()
    #     sdf_n = flood_fill(sdf_d)
    #     sdf_n = sdf_n.reshape(-1)
    #     sdf_n = torch.from_numpy(sdf_n).to(device)
    #     sdf_mask = (sdf_n == -1)
    with torch.no_grad():
        temp_d = model(None, dcoords=coords, dfeats=feats_n, debug=True)
        sdf_d = temp_d.reshape((257, 257, 257))
        sdf_d = sdf_d.detach().cpu().numpy()
        sdf_mask = wzj_final(
            sdf_d,
        )
        # exit()
        sdf_mask = sdf_mask.reshape(-1)
        sdf_mask = torch.from_numpy(sdf_mask).to(device)

    with torch.no_grad():
        temp_d = model(None, dcoords=coords, dfeats=feats_n, debug=True)

        temp_d[sdf_mask] *= -1
        sdf_d = temp_d.reshape((257, 257, 257))
        sdf_d = sdf_d.detach().cpu().numpy()
        sdf_mask1 = wzj_final(
            sdf_d,
        )
        sdf_mask1 = sdf_mask1.reshape(-1)
        sdf_mask1 = torch.from_numpy(sdf_mask1).to(device)
        print(sdf_mask1.sum())
    sdf_mask = sdf_mask | sdf_mask1
    n_coords = torch.load(f'{OUTPUT_PATH}/{name}/params/v_pos_.pt').to(coords)
    m_mask = torch.load(f'{OUTPUT_PATH}/{name}/params/marching_mask.pt').to(device)

    frame_dicts = {key : torch.from_numpy(value).to('cuda:0') for key, value in smplx_params.items()}

    # TODO
    gm.compute_canonical_teeth(shape[0:1])
    teeth, teeth_faces = gm.compute_com_verts(frame_dicts, motion=True)
    indices = torch.load(f'{OUTPUT_PATH}/{name}/params/indices.pt')
    v_0, _ = densify(smplx_v[0], smplx.faces) 

    for num in tqdm(range(150), desc='render'):
        # mesh = model(None, dcoords=coords, dfeats=feats_g, mask=sdf_mask)
        # mesh.vertices = (mesh.vertices - T) @ R[0] / s[0]
        # mesh.vertices = (mesh.vertices - t) / k
        # img, new_m = render_m.forward_visualization_geo(mesh.vertices[None], mesh.faces[None].int(), cam_para, [H, W])
        # img = np.concatenate(
        #     [img, new_m*255.],
        #     axis=-1
        # )
        #     # cv2.imwrite(f'Paper/{num}_m.png', img[0])
        # cv2.imwrite(f'{OUTPUT_PATH}/{name}/normals/normal_{num}_wzj.png', img[0])
        # return
        # if num != 1 and num != 70:
        #     continue
        # mesh = model(input, v_a=torch.from_numpy(deformation[num]).to(dtype=torch.float32, device=device))
        # 
        v_last, _ = densify(smplx_v[num], smplx.faces)
        # motion_id = new_ff[face_id]

        deformation = (v_last[None] - v_0[None])
        # deformation = (smplx_v[:150] - smplx_v[0:1])
        # deformation = (smplx_v - smplx_v[0:1])
        # print(deformation.shape)
        # print(motion_id.shape)
        deformation = deformation[:, motion_id, :] # (B ,F, 3, 3) # cpu
        deformation = np.einsum('bkij, bki->bkj', deformation, np.tile(uvw, (deformation.shape[0], 1, 1)))
        deformation = s.cpu().numpy() * k.cpu().numpy() * deformation @ R[0].transpose(-2, -1).cpu().numpy()

        mesh = model(None, dcoords=coords, dfeats=feats, v_a=torch.from_numpy(deformation[0]).to(dtype=torch.float32, device=device), mask=sdf_mask, n_coords=n_coords, marching_mask=m_mask, indices=indices)
        # mesh = model(None, dcoords=coords, dfeats=feats, v_a=torch.from_numpy(deformation[0]).to(dtype=torch.float32, device=device), mask=sdf_mask, n_coords=n_coords)
        mesh.vertices = (mesh.vertices - T) @ R[0] / s[0]
        mesh.vertices = (mesh.vertices - t) / k
        # if num != 20:
        #     continue
        # else:
        #     trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.detach().cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/objects/animation_mouth.obj')
        #     exit()
                # my_vertices = torch.cat(
                #     [
                #         mesh.vertices,
                #         output.vertices[num, -1092:],
                #         teeth[num],
                #     ],
                #     dim=0
                # ).to(device)
                # mf = mesh.faces.new_tensor(smplx.faces[~mask])
                # mf -= mf.min()
                # mf += mesh.faces.max() + 1
                # my_faces = torch.cat(
                #     [
                #         mesh.faces,
                #         mf
                #     ],
                #     dim = 0
                # )
                # mf = my_faces.new_tensor(teeth_faces[0])
                # mf -= mf.min()
                # mf += my_faces.max() + 1
                # my_faces = torch.cat(
                #     [
                #         my_faces,
                #         mf
                #     ],
                #     dim=0
                # )
                # mesh = MeshExtractResult(
                #     vertices=my_vertices,
                #     faces=my_faces,
                # )
        # if num == 1 or num == 70:
        #     img, new_m = render_m.forward_visualization_geo(mesh.vertices[None], mesh.faces[None].int(), cam_para, [H, W])
        #     img = np.concatenate(
        #         [img, new_m*255.],
        #         axis=-1
        #     )
        #     cv2.imwrite(f'Paper/{num}_m.png', img[0])
        # continue
        # if num == 0:
        #     num_v = mesh.vertices.shape[0]
        #     num_f = mesh.faces.shape[0]
        # oa = (num_v == mesh.vertices.shape[0]) and (num_f == mesh.faces.shape[0])
        # print(oa)


        # if (num == shape.shape[0] - 1):
        #     print('final: ', oa)
        # if num == 0:
        #     trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.detach().cpu().numpy()).export('src/tutils/hhh.obj')
        # mesh.face_normal = torch.matmul(mesh.face_normal, R)
        # if num == 22 or num == 0:
        #     trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.detach().cpu().numpy()).export(f'exp/{name}/objects/mouth_{num}.obj')
        # dicts = render(mesh, extrinsic, proj[0], HW=[H, W])
        img, new_m = render_m.forward_visualization_geo(mesh.vertices[None], mesh.faces[None].int(), cam_para, [H, W])
        img = np.concatenate(
            [img, new_m*255.],
            axis=-1
        )
            # cv2.imwrite(f'Paper/{num}_m.png', img[0])
        cv2.imwrite(f'{OUTPUT_PATH}/{name}/normals/normal_{num}.png', img[0])
    
    now = datetime.now()
    t = now.strftime('%Y%m%d%H%M%S')
    args_ = ['ffmpeg', '-framerate', '24', '-start_number', '0', '-i', f'{OUTPUT_PATH}/{name}/normals/normal_%d.png', '-b:v', '10M', '-pix_fmt', 'yuv420p', f'{OUTPUT_PATH}/{name}/normals/{t}.mp4']
    subprocess.run(args_)


def motion_color(name):
    # print('wzj')
    # exit()
    inputdir = cfg['input_dir']
    gm = GeoModel('cuda:0')
    os.makedirs(f'{OUTPUT_PATH}/{name}/colors', exist_ok=True)
    # name = 'emotional_vids_confident-bearded-man-posing-on-white-background-SBV-346838037-4K.mp4'

    image = cv2.imread(f'{inputdir}/{name}/ori_imgs/{cfg["img_path"]}.png')
    HW = image.shape[:2]
    H, W = HW
    coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_0_new.pt').to(device)
    feats_n = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_0_new.pt').to(device)
    # feats = torch.load(f'src/motion_sdf/delta.pt').to(device)
    feats_c = torch.load(f'{OUTPUT_PATH}/{name}/params/delta_with_color.pt').to(device)
    feats_e = torch.load(f'{OUTPUT_PATH}/{name}/params/delta_with_eye.pt').to(device)
    feats = torch.load(f'{OUTPUT_PATH}/{name}/params/{pt_name}.pt').to(device)
    input = sp.SparseTensor(
        coords=coords,
        feats=feats
    )
    # 1012
    # motion = np.load(f'output/{name}/slats/motion_sdf.npz')
    motion = np.load(f'{OUTPUT_PATH}/{name}/slats/motion_sdf.npz')
    face_id = motion['face_id']
    uvw = motion['uvw']
    track_path = f'{inputdir}/{name}/body_track/smplx_track.pth'
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
    mask = ~(smplx.faces > index).any(axis=-1)
    ff = smplx.faces
    motion_id = ff[face_id]
    # motion_id = smplx.faces[face_id]
    # pp = output.vertices[0, :-1092].detach().cpu().numpy()
    # trimesh.Trimesh(vertices=output.vertices[0].detach().cpu().numpy(), faces=smplx.faces).export('src/sm0.obj')
    # trimesh.Trimesh(vertices=output.vertices[141].detach().cpu().numpy(), faces=smplx.faces).export('src/sm1.obj')
    
    proj = get_ndc_proj_matrix(cam_para[0:1], [H, W])

    smplx_v = output['vertices'].squeeze().detach().cpu().numpy()
    # smplx_v[-1092:, 2] -= 0.002

    # 1012
    # deformation = smplx_v - smplx_v[0:1] # cpu
    deformation = (smplx_v[:150] - smplx_v[0:1])
    # deformation = (smplx_v - smplx_v[0:1])
    # print(deformation.shape)
    # print(motion_id.shape)
    deformation = deformation[:, motion_id, :] # (B ,F, 3, 3) # cpu
    print(motion_id.max(), 'wzj')
    # print(deformation.shape)
    # exit()
    deformation = np.einsum('bkij, bki->bkj', deformation, np.tile(uvw, (deformation.shape[0], 1, 1)))

    trans = np.load(f'{OUTPUT_PATH}/{name}/params/trans.npz')
    kt = np.load(f'{OUTPUT_PATH}/{name}/params/kt.npz')
    k, t = list(kt.values())
    s, R, T = list(trans.values())
    s = torch.from_numpy(s).to(device)
    R = quaternion_to_matrix(torch.from_numpy(R).to(device))
    T = torch.from_numpy(T).to(device)
    k = torch.from_numpy(k).to(device)
    t = torch.from_numpy(t).to(device)
    # R = quaternion_to_matrix(torch.from_numpy(R).to(device))
    deformation = s.cpu().numpy() * k.cpu().numpy() * deformation @ R[0].transpose(-2, -1).cpu().numpy()
    extrinsic = torch.tensor(
            [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]
            ]
        ).to(device)
    # with torch.no_grad():
    #     sdf_d = model(None, dcoords=coords, dfeats=feats_n, debug=True)
    #     sdf_d = sdf_d.reshape((257, 257, 257))
    #     sdf_d = sdf_d.detach().cpu().numpy()
    #     sdf_n = flood_fill(sdf_d)
    #     sdf_n = sdf_n.reshape(-1)
    #     sdf_n = torch.from_numpy(sdf_n).to(device)
    #     sdf_mask = (sdf_n == -1)
    with torch.no_grad():
        temp_d = model(None, dcoords=coords, dfeats=feats_n, debug=True)
        sdf_d = temp_d.reshape((257, 257, 257))
        sdf_d = sdf_d.detach().cpu().numpy()
        sdf_mask = wzj_final(
            sdf_d,
        )
        # exit()
        sdf_mask = sdf_mask.reshape(-1)
        sdf_mask = torch.from_numpy(sdf_mask).to(device)

    with torch.no_grad():
        temp_d = model(None, dcoords=coords, dfeats=feats_n, debug=True)

        temp_d[sdf_mask] *= -1
        sdf_d = temp_d.reshape((257, 257, 257))
        sdf_d = sdf_d.detach().cpu().numpy()
        sdf_mask1 = wzj_final(
            sdf_d,
        )
        sdf_mask1 = sdf_mask1.reshape(-1)
        sdf_mask1 = torch.from_numpy(sdf_mask1).to(device)
        print(sdf_mask1.sum())
    sdf_mask = sdf_mask | sdf_mask1
    n_coords = torch.load(f'{OUTPUT_PATH}/{name}/params/v_pos_.pt').to(coords)
    m_mask = torch.load(f'{OUTPUT_PATH}/{name}/params/marching_mask.pt').to(device)
    frame_dicts = {key : torch.from_numpy(value).to('cuda:0') for key, value in smplx_params.items()}

    # TODO
    gm.compute_canonical_teeth(shape[0:1])
    teeth, teeth_faces = gm.compute_com_verts(frame_dicts, motion=True)
    indices = torch.load(f'{OUTPUT_PATH}/{name}/params/indices.pt')
    upper_teeth = trimesh.load('assets/upper_teeth.obj')
    lower_teeth = trimesh.load('assets/lower_teeth.obj')
    teeth_gt = np.concatenate(
        [
            upper_teeth.visual.vertex_colors,
            lower_teeth.visual.vertex_colors,
        ],
        axis=0
    )
    teeth_colors = torch.from_numpy(teeth_gt).to(device=device, dtype=torch.float32)
    light_params = torch.load(f'{OUTPUT_PATH}/{name}/params/light_params.pt').to(device)
    
    for num in tqdm(range(deformation.shape[0]), desc='render'):
        # mesh = model(input, v_a=torch.from_numpy(deformation[num]).to(dtype=torch.float32, device=device))
        # mesh = model(None, dcoords=coords, dfeats=feats, v_a=torch.from_numpy(deformation[num]).to(dtype=torch.float32, device=device), mask=sdf_mask, n_coords=n_coords, marching_mask=m_mask)
        mesh = model(None, dcoords=coords, dfeats=feats_c, v_a=torch.from_numpy(deformation[num]).to(dtype=torch.float32, device=device), mask=sdf_mask, n_coords=n_coords, marching_mask=m_mask, indices=indices)
        mesh2smplx(name, mesh)
        # mesh.vertices = (mesh.vertices - T) @ R[0] / s[0]
        # mesh.vertices = (mesh.vertices - t) / k
        # my_vertices = torch.cat(
        #     [
        #         mesh.vertices,
        #         output.vertices[num, -1092:],
        #         teeth[num],
        #     ],
        #     dim=0
        # ).to(device)
        # my_vertices = torch.cat(
        #     [
        #         mesh.vertices,
        #         output.vertices[num, -1092:],
        #         teeth[num],
        #     ],
        #     dim=0
        # ).to(device)
        my_vertices = torch.cat(
            [
                mesh.vertices,
                # output.vertices[num, -1092:],
                teeth[num],
            ],
            dim=0
        ).to(device)
        # mf = mesh.faces.new_tensor(smplx.faces[~mask])
        # mf -= mf.min()
        # mf += mesh.faces.max() + 1
        # my_faces = torch.cat(
        #     [
        #         mesh.faces,
        #         mf
        #     ],
        #     dim = 0
        # )
        my_faces = mesh.faces
        mf = my_faces.new_tensor(teeth_faces[0])
        mf -= mf.min()
        mf += my_faces.max() + 1
        my_faces = torch.cat(
            [
                my_faces,
                mf
            ],
            dim=0
        )
        # my_attr = torch.cat(
        #     [
        #         mesh.vertex_attrs,
        #         torch.ones((1092, 6)).to(mesh.vertex_attrs) + feats_e,
        #     ],
        #     dim=0
        # )
        my_attr = mesh.vertex_attrs
        # my_attr += feats_c
        teeth_color = torch.zeros((teeth[num].shape[0], 6)).to(mesh.vertex_attrs)
        # # teeth_color[..., 0] *= 230. / 255.
        # # teeth_color[..., 1] *= 220. / 255.
        # # teeth_color[..., 2] *= 200. / 255.
        teeth_color[..., :3] = torch.pow(teeth_colors[..., :3] / 255., 1 / 2.2)
        # teeth_color[..., :3] = teeth_colors[..., :3] / 255.
        print(my_attr.shape)
        print(teeth_color.shape)
        my_attr = torch.cat(
            [
                my_attr,
                teeth_color,
            ],
            dim=0
        )
        mesh = MeshExtractResult(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_attrs=mesh.vertex_attrs
        )

        
        # mesh = MeshExtractResult(
        #     vertices=my_vertices,
        #     faces=my_faces,
        # )
        # if num == 0:
        #     num_v = mesh.vertices.shape[0]
        #     num_f = mesh.faces.shape[0]
        # oa = (num_v == mesh.vertices.shape[0]) and (num_f == mesh.faces.shape[0])
        # print(oa)


        # if (num == shape.shape[0] - 1):
        #     print('final: ', oa)
        # if num == 0:
        #     trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.detach().cpu().numpy()).export('src/tutils/hhh.obj')
        # mesh.face_normal = torch.matmul(mesh.face_normal, R)
        dicts = render(mesh, extrinsic, proj[0], HW=[H, W], return_types=["mask", "normal", "depth", "color"], shading=light_params)
        # cv2.imwrite(f'outputs/{name}/geo/normal_{num}.png', dicts['normal'].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255)
        cv2.imwrite(f'{OUTPUT_PATH}/{name}/colors/color_{num}.png', dicts['color'].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)

    now = datetime.now()
    t = now.strftime('%Y%m%d%H%M%S')
    args_ = ['ffmpeg', '-framerate', '24', '-start_number', '0', '-i', f'{OUTPUT_PATH}/{name}/colors/color_%d.png', '-b:v', '10M', '-pix_fmt', 'yuv420p', f'{OUTPUT_PATH}/{name}/colors/{t}.mp4']
    subprocess.run(args_)

def mask_crop(name):
    inputdir = cfg['input_dir']
    track_path = f'{inputdir}/{name}/body_track/smplx_track.pth'
    smplx_params = torch.load(track_path)
    cam_para, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, shape, expr, global_orient, transl = to_device(**smplx_params)
    extrinsic = torch.tensor(
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ]
    ).to(device)
    image = cv2.imread(f'{inputdir}/{name}/ori_imgs/{cfg["img_path"]}.png')
    parse = cv2.imread(f'{inputdir}/{name}/parsing/{cfg["img_path"]}.png', cv2.IMREAD_UNCHANGED)
    parse_mouth = ((parse == 11) | (parse == 12) | (parse == 13)).astype(np.uint8)
    parse_mouth = torch.from_numpy(parse_mouth).to(device)

    HW = image.shape[:2]
    H, W = HW
    proj = get_ndc_proj_matrix(cam_para[0:1], [H, W])

    model = SparseFeatures2Mesh(res=256, use_color=True)

    feats_n = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_0_new.pt').to(device)
    feats = torch.load(f'{OUTPUT_PATH}/{name}/params/delta_geo_mouth_new.pt').to(device)
    coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_0_new.pt').to(device)

    with torch.no_grad():
        temp_d = model(None, dcoords=coords, dfeats=feats_n, debug=True)
        sdf_d = temp_d.reshape((257, 257, 257))
        sdf_d = sdf_d.detach().cpu().numpy()
        sdf_mask = wzj_final(
            sdf_d,
        )
        # exit()
        sdf_mask = sdf_mask.reshape(-1)
        sdf_mask = torch.from_numpy(sdf_mask).to(device)
    with torch.no_grad():
        temp_d = model(None, dcoords=coords, dfeats=feats_n, debug=True)

        temp_d[sdf_mask] *= -1
        sdf_d = temp_d.reshape((257, 257, 257))
        sdf_d = sdf_d.detach().cpu().numpy()
        sdf_mask1 = wzj_final(
            sdf_d,
        )
        sdf_mask1 = sdf_mask1.reshape(-1)
        sdf_mask1 = torch.from_numpy(sdf_mask1).to(device)
        print(sdf_mask1.sum())
    sdf_mask = sdf_mask | sdf_mask1
    m_mask = torch.load(f'{OUTPUT_PATH}/{name}/params/marching_mask.pt').to(device)
    indices = torch.load(f'{OUTPUT_PATH}/{name}/params/indices.pt')
    feats_c = torch.load(f'{OUTPUT_PATH}/{name}/params/delta_with_color.pt').to(device)
    feats_g = torch.load(f'{OUTPUT_PATH}/{name}/params/delta_geo_new.pt').to(device)



    mesh = model(None, dcoords=coords, dfeats=feats_g, training=False, mask=sdf_mask)
    mesh2smplx(name, mesh)
    dicts = render(mesh, extrinsic, proj[0], HW=[H, W], return_types=["normal", "mask"])
    # my_pos = -dicts['posi']
    my_pos = torch.nn.functional.normalize(my_pos, dim=0, eps=1e-6)
    my_nor = dicts['normal'] * 2 - 1
    xz = (my_pos * my_nor).sum(dim=0)
    mm = xz < 0.
    print(parse_mouth.max())
    normal_mask = (mm).float() * parse_mouth
    k = (normal_mask.cpu().numpy() * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(k, (5, 5), 0)
    _, smooth_mask = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
    print(normal_mask.max())
    normal_mask = (1 - smooth_mask / 255.).astype(np.uint8)
    

    input = cv2.imread(f'{OUTPUT_PATH}/{name}/normals/normal_0.png', cv2.IMREAD_UNCHANGED)
    input[..., 3] *= normal_mask
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/normals/normal_0_1.png', input)
    # cv2.imwrite('test_mask_new.png', mm.float().detach().cpu().numpy() * 255.)
    # cv2.imwrite('test_normal_new.png', dicts['normal'].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)
    # cv2.imwrite('test_posi_new.png', (my_pos.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] + 1) / 2 * 255.)


name_list = ['287', '290', '294', '285', '283', '282', '274', '262', '259', '253', '249', '248', '247', '240', '239', '238', '232', '227', '223', '220', '216', '212', '200', '199', '188', '179', '165', '149', '140', '139', '128', '115', '112', '108', '106', '104', '098', '083', '076', '075', '074', '071', '060', '055', '040', '036', '031', '030', '290', '294', '301', '306', '307', '313', '314', '315', '318', '319', '320', '326', '331', '371']

new_list = ['030', '036', '055', '071', '074', '076', '083', '106', '112', '115', '139', '165', '188', '199', '290', '294', '307', '313', '315', '331']


def trellis(name):
    os.makedirs(f'{OUTPUT_PATH}/{name}/compare', exist_ok=True)
    gm = GeoModel('cuda:0')
    render_m = MeshRenderer()
    inputdir = cfg['input_dir']
    image = cv2.imread(f'{inputdir}/{name}/ori_imgs/{cfg["img_path"]}.png')
    HW = image.shape[:2]
    H, W = HW
    track_path = f'{inputdir}/{name}/body_track/smplx_track.pth'
    smplx_params = torch.load(track_path)
    cam_para, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, shape, expr, global_orient, transl = to_device(**smplx_params)
    mesh = trimesh.load(f'{OUTPUT_PATH}/{name}/objects/sample_outer_filtered.obj')
    mesh = MeshExtractResult(vertices=torch.from_numpy(mesh.vertices).to(device).float(), faces=torch.from_numpy(mesh.faces).to(device))
    mesh2smplx(name, mesh=mesh)
    img, new_m = render_m.forward_visualization_geo(mesh.vertices[None], mesh.faces[None].int(), cam_para, [H, W])
    img = np.concatenate(
        [img, new_m*255.],
        axis=-1
    )

    cv2.imwrite(f'{OUTPUT_PATH}/{name}/compare/trellis.png', img[0])
pt_name = 'delta_geo_show_ffhq'

if __name__ == '__main__':
    name = 'pipe'
    motion(name)
    exit()
    # color(name)
    # motion_color(name)
    # exit()
    # trellis(name)
    # exit()
    paths = ['nersemble_vids_' + name + '.mp4' for name in new_list]
    with open('nersemble_ids', 'w') as f:
        for name in paths:
            f.write(name)
            f.write('\n')
    exit()

    paths = ['nersemble_vids_' + name + '.mp4' for name in name_list]
    for name in paths:
    # name = 'nersemble_vids_326.mp4'
        try:
            print(f'processing {name}')
            # motion(name)
            trellis(name)
        except Exception as e:
            print(e)
    exit()
    color(name)
    motion_color(name)

    exit()
    # motion_color(name)
    # exit()
    # x, y = preprocess_image
    # inputs = f'{cfg["input_dir"]}/{name}/ori_imgs/000000.png'
    # x, y = preprocess_image(Image.open(inputs), 768, 768)
    # cv2.imwrite('y.png', y * 255.)
    # exit()
    # mvadapter(name)
    # exit()
    # names = os.listdir('/public/home/wangzhijun/Nersemble')
    paths = ['nersemble_vids_' + name + '.mp4' for name in name_list]
    for name in paths:
        if os.path.exists(f'/public/home/wangzhijun/SDF-Avatar/outputs_ners/{name}/params/delta_with_color.pt'):
            print(f'{name} has been processed')
            continue
        try:
            print(f'processing {name}')
            # motion(name)
            color(name)
            # motion_color(name)
        except Exception as e:
            print(e)
    # color(name)
    # motion_color(name)
    exit()
    # motion(name)
    # motion_color(name)


    paths = os.listdir('/public/home/wangzhijun/Nersemble')

    for path in paths:
        if os.path.exists(f'outputs_ners/{path}/images/output.png'):
            # logger.info(f'{path} has been processed')
            # exit()
            continue
        if os.path.exists(f'outputs_ners/{path}/objects/single.obj'):
            # print(path)
            try:
                print(f'processing {path}')
                mvadapter(path)
            except Exception as e:
                print(e)
    exit()
    