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
from utils.utils_io import read_png
from utils.utils_render import get_ndc_proj_matrix, render
from trellis.utils.grad_clip_utils import AdaptiveGradClipper
from trellis.trainers.utils import *
from utils.utils_train import *
import math
import trimesh
from trellis.representations import MeshExtractResult
from trellis.modules import sparse as sp
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.structures import Meshes
import cv2
from scipy.ndimage import binary_dilation,binary_fill_holes, binary_erosion
from scipy.ndimage import generate_binary_structure
import cubvh
from trellis.modules import sparse as sp
from trellis.representations.mesh import SparseFeatures2Mesh
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR, CosineAnnealingWarmRestarts
import logging
import torch.nn.functional as F

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

def clip_columns_grad(grad):
    # grad 形状和 param 一样 [out_features, in_features]
    g = grad.clone()
    max_norm = 2e-5
    # 对指定列做逐列 clip
    col_norms = g[:8].norm(dim=-1, keepdim=True)  # 每列的 L2 norm
    scale = (max_norm / (col_norms + 1e-6)).clamp(max=1.0)
    g[:8] = g[:8] * scale  # 按列缩放
    return g

# def make_master_params(model_params):
#     '''
#     用了trellis处理float16的方法
#     这里的理论是，网络参数里既有float16，又有float32
#     网络训练过程中，中间变量占了绝大部分的显存，所以尽可能让中间变量实用float16
#     此时可以正常执行到backward
#     但是有一个问题，网络参数即有flaot16也有32
#     对于adam来说float16很容易出现nan，所以它希望grad是32的，
#     但是计算出的grad有16
#     那么一个解决方案就是把网络参数原样拷贝一份，但是detach()并处理为float32
#     然后将backward得到的grad也拷贝一份并处理为float32
#     然后将优化作用在这个拷贝变量上，
#     但此时原本的参数还没有更新，那么就是将新参数的值拷贝回旧的变量即可
#     '''
#     from torch._utils import _flatten_dense_tensors
#     master_params = _flatten_dense_tensors(
#         [param.detach().float() for param in model_params]
#     )
#     master_params = torch.nn.Parameter(master_params)
#     master_params.requires_grad = True
#     return [master_params]

def main(name):
    # from utils.utils_geo import 
    # print(OUTPUT_PATH)
    # name = '001_01'
    inputdir = cfg['input_dir']
    os.makedirs(f'{OUTPUT_PATH}/{name}/geo', exist_ok=True)
    from src.networks_bak import myNet
    training_net = myNet()
    training_net.to(device)
    training_net.train()
    training_net.out_layer.weight.register_hook(clip_columns_grad)
    training_net.out_layer.bias.register_hook(clip_columns_grad)

    # david_normal = cv2.imread(f'{inputdir}/{name}/normals_david/{cfg["img_path"]}.png', cv2.IMREAD_UNCHANGED)
    # david_normal = cv2.cvtColor(david_normal, cv2.COLOR_BGR2RGB)
    # david_normal = torch.as_tensor(david_normal, device=device).float() / 255.
    # david_normal = david_normal.permute(2, 0, 1)
    # normal_david = torch.nn.functional.normalize(david_normal, dim=0, eps=1e-6)
    track_path = f'{inputdir}/{name}/body_track/smplx_track.pth'
    print(track_path)
    parse = read_png(f'{inputdir}/{name}/parsing')

    #TODO
    H, W = parse.shape[1:3]
    print(f'{inputdir}/{name}/parsing/{cfg["img_path"]}.png')
    parse = cv2.imread(f'{inputdir}/{name}/parsing/{cfg["img_path"]}.png')
    parse = parse[None]

    gt_mask = torch.from_numpy((parse!=0).astype(np.uint8))
    ffm = ((parse == 2) | ((parse > 5) & (parse < 14)))
    eye = ((parse == 8) | (parse == 9))
    eye_mask = (eye.astype(np.uint8) * (parse != 0)).astype(np.uint8)
    face_mask = (ffm.astype(np.uint8) * (parse!=0)).astype(np.uint8)
    smplx_params = torch.load(track_path)
    cam_para, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, shape, expr, global_orient, transl = to_device(**smplx_params)
    proj = get_ndc_proj_matrix(cam_para[0:1], [H, W])
    extrinsic = torch.tensor(
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ]
    ).to(device)
    batch_size = body_pose.shape[0]
    lhand_pose = smplx.left_hand_pose.expand(batch_size, -1)
    rhand_pose = smplx.right_hand_pose.expand(batch_size, -1)
    # leye_pose = smplx.leye_pose.expand(batch_size, -1)
    # reye_pose = smplx.reye_pose.expand(batch_size, -1)
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

    # print(smplx_v.shape)
    # exit()


    coarse_feats = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_coarse_back.pt').to(device)
    coarse_coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_coarse_back.pt').to(device)
    feats = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_0_new.pt').to(device)
    coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_0_new.pt').to(device)


    from trellis.representations.mesh import SparseFeatures2Mesh
    model = SparseFeatures2Mesh(res=256, use_color=True)

    fp16_scale_growth = 0.0001
    log_scale = 20
    grad_clip = AdaptiveGradClipper(max_norm=0.5, clip_percentile=95)
    model_params = [p for p in training_net.parameters() if p.requires_grad]
    master_params = make_master_params(model_params)
    optimizer = torch.optim.AdamW(master_params, lr=5e-4, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

    delta = torch.zeros((feats.shape[0], 53)).float().to(device).requires_grad_(True)
    optimizer_d = torch.optim.Adam([delta], lr=1.5e-3)
    scheduler_d = StepLR(optimizer_d, step_size=100, gamma=0.9)

    face_mask = torch.from_numpy(face_mask).to(device).permute(0, 3, 1, 2)
    eye_mask = torch.from_numpy(eye_mask).to(device).permute(0, 3, 1, 2)
    with torch.no_grad():
        temp_d = model(None, dcoords=coords, dfeats=feats, debug=True)
        sdf_d = temp_d.reshape((257, 257, 257))
        sdf_d = sdf_d.detach().cpu().numpy()
        sdf_mask = wzj_final(
            sdf_d,
        )
        sdf_mask = sdf_mask.reshape(-1)
        sdf_mask = torch.from_numpy(sdf_mask).to(device)
        print(sdf_mask.sum())
    with torch.no_grad():
        temp_d = model(None, dcoords=coords, dfeats=feats, debug=True)

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
    
    rotas_t = torch.Tensor([
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ],
        [
            [math.cos(math.pi / 4), 0, -math.sin(math.pi / 4), 0],
            [0, 1, 0, 0],
            [math.sin(math.pi / 4), 0, math.cos(math.pi / 4), 0],
            [0, 0, 0, 1]
        ],
        [
            [math.cos(-math.pi / 4), 0, -math.sin(-math.pi / 4), 0],
            [0, 1, 0, 0],
            [math.sin(-math.pi / 4), 0, math.cos(-math.pi / 4), 0],
            [0, 0, 0, 1]
        ],
        [
            [1, 0, 0, 0],
            [0, math.cos(-math.pi / 4), -math.sin(-math.pi / 4), 0],
            [0, math.sin(-math.pi / 4), math.cos(-math.pi / 4), 0],
            [0, 0, 0, 1]
        ],
        [
            [math.cos(-math.pi), 0, -math.sin(-math.pi), 0],
            [0, 1, 0, 0],
            [math.sin(-math.pi), 0, math.cos(-math.pi), 0],
            [0, 0, 0, 1]
        ],
        [
            [1, 0, 0, 0],
            [0, math.cos(math.pi / 2), -math.sin(math.pi / 2), 0],
            [0, math.sin(math.pi / 2), math.cos(math.pi / 2), 0],
            [0, 0, 0, 1]
        ]
    ]).to(device)
    with torch.no_grad():
        mesh_gt = model(None, dcoords=coords, dfeats=feats, training=False, mask=sdf_mask)
        mesh2smplx(name, mesh_gt, output_dir=OUTPUT_PATH)
        z_mean = mesh_gt.vertices[..., -1].mean().item()
        dicts_gt = render(mesh_gt, extrinsic@rotas_t, proj[0], HW=[H, W])
    

    rotas = torch.Tensor([
        [
            [math.cos(math.pi / 4), 0, -math.sin(math.pi / 4), 0],
            [0, 1, 0, 0],
            [math.sin(math.pi / 4), 0, math.cos(math.pi / 4), 0],
            [0, 0, 0, 1]
        ],
        [
            [math.cos(-math.pi / 4), 0, -math.sin(-math.pi / 4), 0],
            [0, 1, 0, 0],
            [math.sin(-math.pi / 4), 0, math.cos(-math.pi / 4), 0],
            [0, 0, 0, 1]
        ],
        [
            [1, 0, 0, 0],
            [0, math.cos(-math.pi / 4), -math.sin(-math.pi / 4), 0],
            [0, math.sin(-math.pi / 4), math.cos(-math.pi / 4), 0],
            [0, 0, 0, 1]
        ]
    ]).to(device)

    with torch.no_grad():
        index = smplx_v.shape[1] - 1092 + 1
        mask = ~(smplx.faces > index).any(axis=-1)
        ff = smplx.faces[mask]
        p = smplx_v[0, :-1092]
        p = smplx_v[0]
        ff = smplx.faces
        v, f = densify(p, ff)
        # v, f = densify(smplx_v[0], smplx.faces)

        v1, f1 = densify(smplx_v[0], smplx.faces)
        # region = trimesh.Trimesh(vertices=v1, faces=f1).submesh([face_region])[0]
        # bvh_v = torch.from_numpy(region.vertices).to(device).float()
        # bvh_f = torch.from_numpy(region.faces).to(device).int()
        # BVH = cubvh.cuBVH(bvh_v, bvh_f)
        # v, f = densify(p, ff)
        smplx_m = MeshExtractResult(
            vertices=torch.from_numpy(v).to(device).float(),
            faces=torch.from_numpy(f).to(device).int()
        )
        smplx_m1 = MeshExtractResult(
            vertices=torch.from_numpy(p).to(device).float(),
            faces=torch.from_numpy(ff).to(device).float()
        )
        smplx_rast = render(smplx_m1, extrinsic, proj[0], HW=[H, W], only_rast=True)
        m_d = (smplx_rast[..., -1] * face_mask[0])
        tris = m_d[m_d > 0].flatten().unique() - 1
        t_m = trimesh.Trimesh(vertices=smplx_v[0], faces=smplx.faces)
        sub_sss = np.load('params/ineedyou.npy')
        t_m = t_m.submesh([sub_sss])[0]
        smplx_m_local = MeshExtractResult(
            vertices=torch.from_numpy(t_m.vertices).to(device).float(),
            faces=torch.from_numpy(t_m.faces).to(device).int()
        )

        smplx_m.vertices[..., -1] -= z_mean
        smplx_m_local.vertices[..., -1] -= z_mean
        mesh_gt.vertices[..., -1] -= z_mean

        extrinsic[2, 3] = z_mean
        smplx_dicts = render(smplx_m, extrinsic, proj[0], HW=[H, W])
        smplx_dicts_l = render(smplx_m_local, extrinsic@rotas_t, proj[0], HW=[H, W])
        smplx_dicts_local = render(smplx_m, extrinsic@rotas_t, proj[0], HW=[H, W])
        # print(smplx_dicts_local['normal'].shape)
        # exit()
        dicts_gt_ = render(mesh_gt, extrinsic@rotas_t, proj[0], HW=[H, W])

        local_mask = smplx_dicts_l['mask'].detach()
        print(smplx_dicts['mask'].shape)
        local = smplx_dicts_local['mask']
    
    masks = []
    structure = generate_binary_structure(2, 1)

    local_mask[0] *= face_mask[0, 0]

    for x in local_mask:
        masks.append(binary_dilation(x.detach().cpu().numpy(), structure=structure, iterations=4))
    local_mask_dilated = np.stack(masks, axis=0)
    # local_mask_dilated = binary_dilation(local_mask.detach().cpu().numpy(), structure=structure, iterations=1)
    local_mask_dilated = torch.from_numpy(local_mask_dilated).to(local_mask)

    input = sp.SparseTensor(
        coords=coarse_coords.to(dtype=torch.int32),
        feats=coarse_feats
    )
    iters = 600
    loss_dict = {}
    for epoch in range(iters + 1):
        # zero_grad(model_params)
        # delta = training_net(input)
        optimizer_d.zero_grad()
        # optimizer_sing.zero_grad()
        ff = torch.cat(
            [
                feats[..., :53].detach() + delta,
                feats[..., 53:].detach()
            ],
            dim=-1
        ).to(device)

        # with torch.no_grad():
        #     sdf_d = model(None, dcoords=coords, dfeats=ff, debug=True)
        #     sdf_d = sdf_d.reshape((257, 257, 257))
        #     sdf_d = sdf_d.detach().cpu().numpy()
        #     sdf_n = flood_fill(sdf_d)
        #     sdf_n = sdf_n.reshape(-1)
        #     sdf_n = torch.from_numpy(sdf_n).to(device)
        #     sdf_mask = (sdf_n == -1)
        mesh = model(None, dcoords=coords, dfeats=ff, training=True, mask=sdf_mask)
        mesh2smplx(name, mesh, output_dir=OUTPUT_PATH)
        # distances, face_id, uvw = BVH.unsigned_distance(mesh.vertices, return_uvw=True)
        # dis_mask = torch.topk(distances, k=bvh_v.shape[0], largest=False)[1]
        # # dis_mask = distances < 0.005
        # source = mesh.vertices[dis_mask]
        # target = bvh_v[bvh_f[face_id[dis_mask]]]
        # target_v = torch.einsum('nij, ni->nj', target, uvw[dis_mask])
        # loss_geo = hub(source, target_v)

        n = torch.cat(
            [
                mesh.vertices[..., :2].clone(),
                mesh.vertices[..., -1:].clone() - z_mean
            ],
            dim=-1
        )
        mesh.vertices = n
        dicts = render(mesh, extrinsic@rotas_t, proj[0], HW=[H, W])
        # B, C, H, W = dicts['normal'].shape
        # out = dicts['normal'].permute(2, 0, 3, 1).reshape(H, B*W, C).detach().cpu().numpy() * 255.
        # out1 = dicts['normal'].permute(2, 0, 3, 1).reshape(H, B*W, C).detach().cpu().numpy() * 255.

        

        # with torch.no_grad():
        #     smplx_dicts = render(smplx_m, extrinsic, proj[0], HW=[H, W])

        

        # depth_loss_eye = 10 * l1_loss(dicts['depth'][0][eye_mask[0, 0] > 0], smplx_dicts['depth'][eye_mask[0, 0] > 0])
        # perceptual_loss_eye = l1_loss(dicts['normal'][0][eye_mask[0] > 0], smplx_dicts['normal'][eye_mask[0] > 0])

        # I_dep = torch.cat(
        #     [
        #         dicts['depth'][0],
        #         smplx_dicts['depth']
        #     ],
        #     dim=-1
        # )
        # print(dicts_gt['depth'].max())
        # print(dicts['depth'].max())
        # print(dicts['depth'].shape)
        # print(dicts['normal'].shape)
        # print(smplx_dicts['normal'].shape)
        # I_nor = torch.cat(
        #     [
        #         dicts['normal'][0],
        #         smplx_dicts['normal']
        #     ],
        #     dim=-1
        # )
        # I_dep1 = torch.cat(
        #     [
        #         dicts['depth'][0],
        #         smplx_dicts['depth']
        #     ],
        #     dim=-1
        # )
        # I_nor1 = torch.cat(
        #     [
        #         dicts['normal'][0] * eye_mask[0],
        #         smplx_dicts['normal'] * eye_mask[0]
        #     ],
        #     dim=-1
        # )
        # print(mesh.vertices.shape)
        # exit()
        # meshes = Meshes(verts=[mesh.vertices], faces=[mesh.faces])
        # loss_laplacian = mesh_laplacian_smoothing(meshes, method="uniform")

        # depth_loss = 10 * l1_loss(dicts['depth'][0], face_mask[0, 0] * smplx_dicts['depth'] + (1 - face_mask[0, 0]) * dicts_gt['depth'][0])
        # perceptual_loss = loss_recon(dicts['normal'][0], smplx_dicts['normal'] * face_mask[0] + (1 - face_mask[0]) * dicts_gt['normal'][0])
        # mmask = smplx_dicts_local['mask'] != 0
        # depth_loss_local = 10 * l1_loss(dicts['depth'][1:], smplx_dicts_local['depth'] * local_mask + (1 - local_mask) * dicts_gt_['depth'][1:])
        # perceptual_loss_local = l1_loss(dicts['normal'][1:], smplx_dicts_local['normal'] * local_mask[:, None] + (1 - local_mask[:, None]) * dicts_gt_['normal'][1:])
        # print(smplx_dicts_local['normal'][1:].shape)
        # print(smplx_dicts_local['normal'].shape)
        # print(local_mask[1:, None].shape)
        # a = dicts['normal'][1:] * (1 - local_mask[1:, None])
        # print(a.shape)
        # print(dicts['normal'][1:].shape)
        # exit()
        # a = dicts['normal'][1:] * (1 - local_mask[1:, None])
        # b = smplx_dicts_local['normal'][1:] * (1 - local_mask[1:, None])
        # c = (1 - local_mask[1:, None]) * dicts_gt_['normal'][1:]
        # for i in range(a.shape[0]):
        #     cv2.imwrite(f'{OUTPUT_PATH}/{name}/objects/test_{i}.png', a[i].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)
        #     cv2.imwrite(f'{OUTPUT_PATH}/{name}/objects/test1_{i}.png', b[i].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)
        #     cv2.imwrite(f'{OUTPUT_PATH}/{name}/objects/test2_{i}.png', c[i].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)


        # print(a.shape)
        # print(b.shape)
        # exit()
        # pred_front = dicts['normal'][0]
        # gt_front = normal_david
        # dot_david = (pred_front * gt_front).sum(dim=0)
        # loss_normal_david = ((1 - dot_david.clamp(-1, 1)) * face_mask[0]).sum() / (face_mask[0].sum() + 1e-6)

        # loss_normal_david_ssim = loss_recon(dicts['normal'][0] * face_mask[0, None], gt_front * face_mask[0, None], lambda_ssim=3, lambda_norm=0, mask=face_mask[0, None])




        pred = torch.nn.functional.normalize(dicts['normal'][0:-2], dim=1, eps=1e-6)
        gt   = torch.nn.functional.normalize(smplx_dicts_local['normal'][0:-2], dim=1, eps=1e-6)

        dot = (pred * gt).sum(dim=1)  # [b,h,w]
        # dot_in = (dot * mask).sum() / (mask.sum() + 1e-6)
        # print(dot.shape)
        loss_local_ = ((1 - dot[:3].clamp(-1, 1)) * local_mask_dilated[0:-3]).sum() / (local_mask_dilated[0:-3].sum() + 1e-6) + 5 * ((1 - dot[3:].clamp(-1, 1)) * local_mask_dilated[-3:-2]).sum() / (local_mask_dilated[-3:-2].sum() + 1e-6)

        # loss_local_ = (1 - torch.einsum('bchw, bchw -> bhw', dicts['normal'][1:-2] * local_mask_dilated[1:-2, None], smplx_dicts_local['normal'][1:-2] * local_mask_dilated[1:-2, None])).mean()
        # perceptual_loss_local_ = 1 * loss_recon(dicts['normal'][1:-3] * local_mask_dilated[1:-3, None], smplx_dicts_local['normal'][1:-3] * local_mask_dilated[1:-3, None], lambda_ssim=5, lambda_norm=0)

        # z = dicts_gt_['normal']
        # for i, x in enumerate(z):
        #     cv2.imwrite(f'{OUTPUT_PATH}/{name}/objects/gt_{i}.png', x.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)
        # z = dicts['normal']
        # for i, x in enumerate(z):
        #     cv2.imwrite(f'{OUTPUT_PATH}/{name}/objects/pred_{i}.png', x.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)
        # exit()
        perceptual_loss_local = 1 * loss_recon((1 - local_mask[1:-2, None]) * dicts['normal'][1:-2], (1 - local_mask[1:-2, None]) * dicts_gt_['normal'][1:-2], lambda_ssim=5, lambda_norm=0, mask=(1 - local_mask[1:-2, None]))

        perceptual_loss_local_back = 1 * loss_recon(dicts['normal'][-2:], dicts_gt_['normal'][-2:], lambda_ssim=5, lambda_norm=0, mask=dicts['mask'][-2:, None])
        # loss_reg = mesh.reg_loss

        # img_n1 = smplx_dicts_local['normal'].permute(1, 2, 0, 3).flatten(start_dim=-2).permute(1, 2, 0)
        # img_n2 = dicts['normal'][1:].permute(1, 2, 0, 3).flatten(start_dim=-2).permute(1, 2, 0)
        # img = torch.cat([img_n1, img_n2], dim=1)
        # cv2.imwrite('src/snnwzj.png', img.detach().cpu().numpy() * 255.)
        # loss = depth_loss + 2 * perceptual_loss + 0.1 * loss_reg + 5 * depth_loss_local + 10 * perceptual_loss_local
        # depth_loss_new = 10 * loss_recon(dicts['depth'][0, None] * local_mask_dilated[0, None], smplx_dicts_local['depth'][0, None] * local_mask_dilated[0, None]) + 5 * loss_recon((1 - local_mask)[0, None] * dicts['depth'][0, None], (1 - local_mask)[0, None] * dicts_gt_['depth'][0, None])
        depth_loss_show = 1 * cal_l1_loss((dicts['depth'][0, None] - smplx_dicts_local['depth'][0, None]), mask=local_mask_dilated[0, None])
        depth_loss_new = 1 * cal_l1_loss((dicts['depth'][0, None] - smplx_dicts_local['depth'][0, None]), mask=local_mask_dilated[0, None], l_type='huber')
        depth_loss_new_dis = 0.75 * cal_l1_loss((dicts['depth'][0, None] - smplx_dicts_local['depth'][0, None]), mask=(1 - local_mask[0, None]))

        depth_loss_local = 1 * cal_l1_loss((dicts['depth'][-3, None] - smplx_dicts_local['depth'][-3, None]), mask=local_mask_dilated[-3, None])
        # perceptual_loss_new = loss_recon(dicts['normal'][0], smplx_dicts_local['normal'][0] * local_mask_dilated[0, None] + (1 - local_mask[0, None]) * dicts_gt_['normal'][0]) ### 2025 11.9 22:33
        perceptual_loss_new = 1 * loss_recon(dicts['normal'][0] * local_mask_dilated[0, None], smplx_dicts_local['normal'][0] * local_mask_dilated[0, None], lambda_ssim=3, lambda_norm=0, mask=local_mask_dilated[0, None])
        normal_loss_local = 1 * loss_recon(dicts['normal'][1:-3] * local_mask_dilated[1:-3, None], smplx_dicts_local['normal'][1:-3] * local_mask_dilated[1:-3, None], lambda_ssim=3, lambda_norm=0, mask=local_mask_dilated[1:-3, None])
        perceptual_loss_new_up = 1 * loss_recon(dicts['normal'][-3] * local_mask_dilated[-3, None], smplx_dicts_local['normal'][-3] * local_mask_dilated[-3, None], lambda_ssim=3, lambda_norm=0, mask=local_mask_dilated[-3, None])
        perceptual_loss_new_dis = loss_recon(dicts['normal'][0] * (1 - local_mask[0, None]), (1 - local_mask[0, None]) * dicts_gt_['normal'][0], lambda_ssim=5, lambda_norm=0, mask=(1 - local_mask[0, None]))
        
        # depth_loss_new_dis = 1 * cal_l1_loss((dicts['depth'][-3, None] - smplx_dicts_local['depth'][-3, None]))
        # loss = 10 * depth_loss + 2 * perceptual_loss + 2 * perceptual_loss_local + 1.5 * loss_laplacian
        normal_loss = perceptual_loss_new
        normal_loss_front = perceptual_loss_new_dis
        # loss = 1 * normal_loss + 15 * perceptual_loss_new_dis / (1 - local_mask[0, None]).mean().detach() + 7.5 * perceptual_loss_local / (1 - local_mask[1:-2, None]).mean().detach() + 0.01 * torch.mean(torch.abs(delta)) + 10 * perceptual_loss_local_back + 0.1 * loss_local_ / (local_mask_dilated[1:-2, None]).mean().detach() + 30 * depth_loss_new
        if epoch < 100:
            lambda_depth = 30
        else:
            lambda_depth = 30
        loss_reg = mesh.reg_loss
        loss = 1 * normal_loss +  0.3 * normal_loss_local + 2 * normal_loss_front + 0.5 * perceptual_loss_local + 0.1 * torch.mean(torch.abs(delta)) + 0.5 * perceptual_loss_local_back + 0.1 * loss_local_ + lambda_depth * depth_loss_new + 30 * depth_loss_local + 1 * perceptual_loss_new_up + lambda_depth * depth_loss_new_dis + 2000 * loss_reg 

        # loss = depth_loss + 2 * perceptual_loss  + depth_loss_eye + 2 * perceptual_loss_eye
        # loss = depth_loss_eye + 2 * perceptual_loss_eye
        # loss = depth_loss + 100 * depth_loss_eye
        # loss = depth_loss_local + 2 * perceptual_loss_local
        
        # loss.backward()
        # optimizer_sing.step()
        # scheduler.step()
        # grad_norm = grad_clip([delta])
        

        #TODO 通过mask选出smplx对应的面，然后对于侧面视角和仰视视角，渲染pred + smplx筛出的面部作为gt，因为此时面部高度重合，只是为了把接缝处处理一下
        #TODO 如果一开始面部过于偏怎么办，去掉这些例子算了
        loss.backward()
        grad_norm = grad_clip(delta)

        optimizer_d.step()

        if epoch % 5 == 0:
            loss_dict.update({'loss': loss.item(), 'depth_loss': depth_loss_show.item(), 
            'depth_loss_new': depth_loss_new.item(), 'normal_loss': normal_loss.item()})
            print(f'epoch: {epoch}, loss: {loss}, depth_loss: {depth_loss_show}, depth_loss_new: {depth_loss_new},  normal_loss: {normal_loss}, local_loss: {perceptual_loss_local}, normal_loss_front: {normal_loss_front}, normal_loss_back: {perceptual_loss_local_back}, loss_local: {loss_local_}, loss_reg: {loss_reg}')

        # scaled_loss = loss * (2 ** log_scale)
        # scaled_loss.backward()

        # print(model_params)
        # exit()

        # model_grads_to_master_grads(model_params, master_params)
        # master_params[0].grad.mul_(1.0 / (2 ** log_scale))
        # grad_norm = grad_clip(master_params)

        # if not any(not p.grad.isfinite().all() for p in model_params):
        #     optimizer.step()
        #     # scheduler.step()
        #     master_params_to_model_params(model_params, master_params)
        #     log_scale += fp16_scale_growth
        # else:
        #     log_scale -= 1
        if epoch == 300:
            mesh.vertices[..., -1:] += z_mean
            trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/objects/single_300.obj')
            torch.save(ff, f'{OUTPUT_PATH}/{name}/params/delta_geo_300.pt')

        if epoch == iters:
            mesh.vertices[..., -1:] += z_mean
            trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/objects/single_show_ffhq.obj')
            smplx_m.vertices[..., -1:] += z_mean
            trimesh.Trimesh(vertices=smplx_m.vertices.detach().cpu().numpy(), faces=smplx_m.faces.cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/objects/single_smplx.obj')
            torch.save(ff, f'{OUTPUT_PATH}/{name}/params/delta_geo_show_ffhq.pt')
            # cv2.imwrite(f'{OUTPUT_PATH}/{name}/objects/dd.png', I_dep.detach().cpu().numpy()[..., ::-1] * 255.)
            # cv2.imwrite(f'{OUTPUT_PATH}/{name}/objects/nn.png', I_nor.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)
            # cv2.imwrite(f'{OUTPUT_PATH}/{name}/objects/dd1.png', I_dep1.detach().cpu().numpy()[..., ::-1] * 255.)
            # cv2.imwrite(f'{OUTPUT_PATH}/{name}/objects/nn1.png', I_nor1.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)


            # cv2.imwrite(f'{OUTPUT_PATH}/{name}/objects/local_mask.png', local_mask.permute(1, 0, 2).flatten(start_dim=1).detach().cpu().numpy() * 255.)
            # cv2.imwrite(f'{OUTPUT_PATH}/{name}/objects/local_normal.png', ((1 - local_mask[1:, None]) * dicts_gt_['normal'][1:]).permute(1, 2, 0, 3).flatten(start_dim=2).permute(1, 2, 0).detach().cpu().numpy() * 255.)
            
        if epoch % 5 == 0:
            # n = dicts['normal']
            cv2.imwrite(f'{OUTPUT_PATH}/{name}/geo/normal_{epoch}.png', dicts['normal'][0].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)

        if epoch == 600:
            # n = dicts['normal']
            cv2.imwrite(f'{OUTPUT_PATH}/{name}/geo/normal_{epoch}.png', dicts['normal'][0].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)

    return loss_dict


def cal_l1_loss(x, mask = None, l_type='l1'):
    if l_type == 'l1':
        if mask is None:
            return torch.abs(x).mean()
        else:
            return torch.abs(x*mask).mean() / (mask.mean() + 1e-7)
    elif l_type == 'huber':
        loss = F.huber_loss(x * 100, torch.zeros_like(x), reduction='none', delta=0.05)
        
        # 2. 处理 Mask
        if mask is None:
            return loss.mean()
        else:
            return (loss * mask).sum() / (mask.sum() + 1e-7)

def cal_depth_loss(depth_pred, depth_gt, mask, parsing_map, face_ldmks):
        height, width = depth_pred.shape[:2]
        device = depth_pred.device
        with torch.inference_mode():
            for i in range(1):
                mask_for_compare = (mask > 0.99) & (parsing_map == 2)
                l = face_ldmks[33, 0].long()
                r = face_ldmks[46, 0].long()
                t = face_ldmks[35, 1].long()
                b = face_ldmks[16, 1].long()
                grid_y, grid_x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
                # grid_x = grid_x.unsqueeze(0).expand(batch_size, -1, -1)
                # grid_y = grid_y.unsqueeze(0).expand(batch_size, -1, -1)
                l, r, t, b = l.view(1, 1), r.view(1, 1), t.view(1, 1), b.view(1, 1)
                bbox_mask = (grid_x >= l) & (grid_x <= r) & (grid_y >= t) & (grid_y <= b)
                mask_for_compare = (mask_for_compare & bbox_mask).float()
                n_pixels = mask_for_compare.sum(dim=[0, 1], keepdim=True) + 1e-8
                mean_gt = (depth_gt * mask_for_compare).sum(dim=[0, 1], keepdim=True) / n_pixels
                mean_pred = (depth_pred * mask_for_compare).sum(dim=[0, 1], keepdim=True) / n_pixels
                centered_gt = torch.where(mask_for_compare>0.5, depth_gt - mean_gt, 0.)
                centered_pred = torch.where(mask_for_compare>0.5, depth_pred - mean_pred, 0.)
                var_gt = (centered_gt ** 2).sum(dim=[0, 1], keepdim=True)
                cov_gt_pred = (centered_pred * centered_gt).sum(dim=[0, 1], keepdim=True)
                scale = cov_gt_pred / (var_gt + 1e-8)
                trans = mean_pred - scale * mean_gt
                depth_gt_aligned = depth_gt * scale + trans
        valid_mask = (mask>0.5).float()
        # num_valid = valid_mask.sum() + 1e-8
        # diff = (depth_pred - depth_gt_aligned) * valid_mask
        # delta_value = 0.002
        # huber_loss = torch.nn.functional.huber_loss(
        #     input=diff, 
        #     target=torch.zeros_like(diff), 
        #     reduction='sum', 
        #     delta=delta_value
        # )
        loss_depth = cal_l1_loss(depth_pred - depth_gt_aligned, valid_mask)
        
        # return loss_depth + 0.3 * (1 - ssim((depth_pred * valid_mask)[None], (depth_gt_aligned * valid_mask)[None]))
        # return huber_loss / num_valid
        return loss_depth

def custom_lr_lambda(epoch):
    if epoch < 150:
        return 1.0  # 保持初始 LR
    elif epoch < 300:
        return 0.9 # 衰减到 90%
    elif epoch < 400:
        return 0.9 * 0.75 # 衰减到 81%
    elif epoch < 500:
        return 0.9 * 0.75 * 0.75 # 衰减到 60.75%
    elif epoch < 600:
        return 0.9 * 0.75 * 0.75 * 0.75 # 衰减到 45.56%
    else:
        # 确保在超过最大 epoch 后，学习率保持不变（或继续衰减）
        return 0.9 ** 2 * 0.75 ** 2

def main_daviad(name):
    from src.networks_bak import myNet

    op = '000001'
    depth_gt = cv2.imread(f'inputs/{name}/depth/{op}.png', cv2.IMREAD_UNCHANGED)
    depth_gt = torch.as_tensor(depth_gt.astype(np.float32), device=device).float() / 65535.
    normal_gt = cv2.imread(f'inputs/{name}/normals_david/{op}.png', cv2.IMREAD_UNCHANGED)
    normal_gt = cv2.cvtColor(normal_gt, cv2.COLOR_BGR2RGB)
    normal_gt = torch.as_tensor(normal_gt, device=device).float() / 255.
    face_ldmks = f'inputs/{name}/face_ldmks/000001.wflw'
    face_ldmks = np.array(np.loadtxt(face_ldmks, dtype=np.float32))
    face_ldmks = torch.as_tensor(face_ldmks, device=device)
    parsing_map = cv2.imread(f'inputs/{name}/parsing/{op}.png', cv2.IMREAD_UNCHANGED)
    parsing_map = torch.as_tensor(parsing_map, device=device)
    seg_mask = cv2.imread(f'inputs/{name}/seg_masks/{op}.png', cv2.IMREAD_UNCHANGED)
    seg_mask = torch.as_tensor(seg_mask, device = device).float() / 255.
    # print(seg_mask.shape)
    # print(face_ldmks.shape)
    # print(parsing_map.shape)
    # print(normal_gt.shape)
    # print(depth_gt.shape)
    # exit()
    # from utils.utils_geo import 
    # print(OUTPUT_PATH)
    # name = '001_01'
    os.makedirs(f'outputs/{name}/geo', exist_ok=True)
    training_net = myNet()
    training_net.to(device)
    training_net.train()
    training_net.out_layer.weight.register_hook(clip_columns_grad)
    training_net.out_layer.bias.register_hook(clip_columns_grad)


    track_path = f'inputs/{name}/body_track/smplx_track.pth'
    print(track_path)
    parse = read_png(f'inputs/{name}/parsing')

    #TODO
    H, W = parse.shape[1:3]
    parse = cv2.imread(f'inputs/{name}/parsing/000001.png')
    parse = parse[None]

    gt_mask = torch.from_numpy((parse!=0).astype(np.uint8))
    ffm = ((parse == 2) | ((parse > 5) & (parse < 14) & (parse != 11)))
    eye = ((parse == 8) | (parse == 9))
    eye_mask = (eye.astype(np.uint8) * (parse != 0)).astype(np.uint8)
    face_mask = (ffm.astype(np.uint8) * (parse!=0)).astype(np.uint8)
    smplx_params = torch.load(track_path)
    cam_para, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, shape, expr, global_orient, transl = to_device(**smplx_params)
    proj = get_ndc_proj_matrix(cam_para[0:1], [H, W])
    extrinsic = torch.tensor(
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ]
    ).to(device)
    extrinsic_o = torch.tensor(
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ]
    ).to(device)
    batch_size = body_pose.shape[0]
    lhand_pose = smplx.left_hand_pose.expand(batch_size, -1)
    rhand_pose = smplx.right_hand_pose.expand(batch_size, -1)
    # leye_pose = smplx.leye_pose.expand(batch_size, -1)
    # reye_pose = smplx.reye_pose.expand(batch_size, -1)
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

    # print(smplx_v.shape)
    # exit()


    coarse_feats = torch.load(f'outputs/{name}/slats/feats_coarse_back.pt').to(device)
    coarse_coords = torch.load(f'outputs/{name}/slats/coords_coarse_back.pt').to(device)
    feats = torch.load(f'outputs/{name}/slats/feats_0_new.pt').to(device)
    coords = torch.load(f'outputs/{name}/slats/coords_0_new.pt').to(device)


    from trellis.representations.mesh import SparseFeatures2Mesh
    model = SparseFeatures2Mesh(res=256, use_color=True)

    fp16_scale_growth = 0.0001
    log_scale = 20
    grad_clip = AdaptiveGradClipper(max_norm=1, clip_percentile=95)
    model_params = [p for p in training_net.parameters() if p.requires_grad]
    master_params = make_master_params(model_params)
    delta = torch.zeros((feats.shape[0], 53)).float().to(device).requires_grad_(True)
    optimizer_d = torch.optim.Adam([delta], lr=1.5e-3)
    optimizer = torch.optim.AdamW(master_params, lr=5e-4, weight_decay=0.0)
    # scheduler = LambdaLR(optimizer, lr_lambda=custom_lr_lambda)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
    scheduler_d = StepLR(optimizer_d, step_size=100, gamma=0.9)

    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=60, T_mult=1, eta_min=1e-6)
    seg_mask = torch.from_numpy(face_mask).to(device)[0, ..., 0]
    face_mask = torch.from_numpy(face_mask).to(device).permute(0, 3, 1, 2)
    eye_mask = torch.from_numpy(eye_mask).to(device).permute(0, 3, 1, 2)
    with torch.no_grad():
        temp_d = model(None, dcoords=coords, dfeats=feats, debug=True)
        sdf_d = temp_d.reshape((257, 257, 257))
        sdf_d = sdf_d.detach().cpu().numpy()
        sdf_mask = wzj_final(
            sdf_d,
        )
        sdf_mask = sdf_mask.reshape(-1)
        sdf_mask = torch.from_numpy(sdf_mask).to(device)
        print(sdf_mask.sum())
    with torch.no_grad():
        temp_d = model(None, dcoords=coords, dfeats=feats, debug=True)

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
    
    rotas_t = torch.Tensor([
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ],
        [
            [math.cos(math.pi / 2), 0, -math.sin(math.pi / 2), 0],
            [0, 1, 0, 0],
            [math.sin(math.pi / 2), 0, math.cos(math.pi / 2), 0],
            [0, 0, 0, 1]
        ],
        [
            [math.cos(-math.pi / 2), 0, -math.sin(-math.pi / 2), 0],
            [0, 1, 0, 0],
            [math.sin(-math.pi / 2), 0, math.cos(-math.pi / 2), 0],
            [0, 0, 0, 1]
        ],
        [
            [1, 0, 0, 0],
            [0, math.cos(-math.pi / 4), -math.sin(-math.pi / 4), 0],
            [0, math.sin(-math.pi / 4), math.cos(-math.pi / 4), 0],
            [0, 0, 0, 1]
        ]
    ]).to(device)
    with torch.no_grad():
        mesh_gt = model(None, dcoords=coords, dfeats=feats, training=False, mask=sdf_mask)
        mesh2smplx(name, mesh_gt, output_dir='outputs')
        z_mean = mesh_gt.vertices[..., -1].mean().item()
        dicts_gt = render(mesh_gt, extrinsic@rotas_t, proj[0], HW=[H, W])

    rotas = torch.Tensor([
        [
            [math.cos(math.pi / 4), 0, -math.sin(math.pi / 4), 0],
            [0, 1, 0, 0],
            [math.sin(math.pi / 4), 0, math.cos(math.pi / 4), 0],
            [0, 0, 0, 1]
        ],
        [
            [math.cos(-math.pi / 4), 0, -math.sin(-math.pi / 4), 0],
            [0, 1, 0, 0],
            [math.sin(-math.pi / 4), 0, math.cos(-math.pi / 4), 0],
            [0, 0, 0, 1]
        ],
        [
            [1, 0, 0, 0],
            [0, math.cos(-math.pi / 4), -math.sin(-math.pi / 4), 0],
            [0, math.sin(-math.pi / 4), math.cos(-math.pi / 4), 0],
            [0, 0, 0, 1]
        ]
    ]).to(device)

    with torch.no_grad():
        index = smplx_v.shape[1] - 1092 + 1
        mask = ~(smplx.faces > index).any(axis=-1)
        ff = smplx.faces[mask]
        p = smplx_v[0, :-1092]
        v, f = densify(p, ff)
        # v, f = densify(smplx_v[0], smplx.faces)

        v1, f1 = densify(smplx_v[0], smplx.faces)
        # region = trimesh.Trimesh(vertices=v1, faces=f1).submesh([face_region])[0]
        # bvh_v = torch.from_numpy(region.vertices).to(device).float()
        # bvh_f = torch.from_numpy(region.faces).to(device).int()
        # BVH = cubvh.cuBVH(bvh_v, bvh_f)
        # v, f = densify(p, ff)
        smplx_m = MeshExtractResult(
            vertices=torch.from_numpy(v).to(device).float(),
            faces=torch.from_numpy(f).to(device).int()
        )
        smplx_m1 = MeshExtractResult(
            vertices=torch.from_numpy(p).to(device).float(),
            faces=torch.from_numpy(ff).to(device).float()
        )
        smplx_rast = render(smplx_m1, extrinsic, proj[0], HW=[H, W], only_rast=True)
        m_d = (smplx_rast[..., -1] * face_mask[0])
        tris = m_d[m_d > 0].flatten().unique() - 1
        t_m = trimesh.Trimesh(vertices=smplx_v[0], faces=smplx.faces)
        sub_sss = np.load('params/ineedyou.npy')
        t_m = t_m.submesh([sub_sss])[0]
        smplx_m_local = MeshExtractResult(
            vertices=torch.from_numpy(t_m.vertices).to(device).float(),
            faces=torch.from_numpy(t_m.faces).to(device).int()
        )

        smplx_m.vertices[..., -1] -= z_mean
        smplx_m_local.vertices[..., -1] -= z_mean
        mesh_gt.vertices[..., -1] -= z_mean

        extrinsic[2, 3] = z_mean
        smplx_dicts = render(smplx_m, extrinsic, proj[0], HW=[H, W])
        smplx_dicts_l = render(smplx_m_local, extrinsic@rotas_t, proj[0], HW=[H, W])
        smplx_dicts_local = render(smplx_m, extrinsic@rotas_t, proj[0], HW=[H, W])
        dicts_gt_ = render(mesh_gt, extrinsic@rotas_t, proj[0], HW=[H, W])

        local_mask = smplx_dicts_l['mask']
        print(smplx_dicts['mask'].shape)
        local = smplx_dicts_local['mask']
    
    masks = []
    structure = generate_binary_structure(2, 1)

    for x in local_mask:
        masks.append(binary_dilation(x.detach().cpu().numpy(), structure=structure, iterations=4))
    local_mask_dilated = np.stack(masks, axis=0)
    # local_mask_dilated = binary_dilation(local_mask.detach().cpu().numpy(), structure=structure, iterations=1)
    local_mask_dilated = torch.from_numpy(local_mask_dilated).to(local_mask)

    input = sp.SparseTensor(
        coords=coarse_coords.to(dtype=torch.int32),
        feats=coarse_feats
    )
    iters = 300
    # with torch.no_grad():
    #     mesh_gt = model(None, dcoords=coords, dfeats=feats, training=False, mask=sdf_mask)
    #     mesh2smplx(name, mesh_gt)
    #     # z_mean = mesh_gt.vertices[..., -1].mean().item()
    #     dicts_gt = render(mesh_gt, extrinsic@rotas_t, proj[0], HW=[H, W])
    my_mask = ((dicts_gt['mask'][0].detach() * seg_mask) > 0.5).float()
    structure = generate_binary_structure(2, 1)
    local_reg_mask = binary_erosion(my_mask.detach().cpu().numpy(), structure=structure, iterations=2)
    local_normal_mask = binary_erosion(my_mask.detach().cpu().numpy(), structure=structure, iterations=1)
    

    local_mask_dilated = binary_dilation(my_mask.detach().cpu().numpy(), structure=structure, iterations=1)
    new_mask = torch.from_numpy(local_mask_dilated).to(device)
    local_reg_mask = torch.from_numpy(local_reg_mask).to(device).float()
    local_normal_mask = torch.from_numpy(local_normal_mask).to(device).float()

    

    for epoch in range(iters + 1):
        # zero_grad(model_params)
        # delta = training_net(input)

        optimizer_d.zero_grad()
        # ff = torch.cat(
        #     [
        #         feats[..., :53].detach() + delta.feats,
        #         feats[..., 53:].detach()
        #     ],
        #     dim=-1
        # ).to(device)
        
        ff = torch.cat(
            [
                feats[..., :53].detach() + delta,
                feats[..., 53:].detach()
            ],
            dim=-1
        ).to(device)

        # with torch.no_grad():
        #     sdf_d = model(None, dcoords=coords, dfeats=ff, debug=True)
        #     sdf_d = sdf_d.reshape((257, 257, 257))
        #     sdf_d = sdf_d.detach().cpu().numpy()
        #     sdf_n = flood_fill(sdf_d)
        #     sdf_n = sdf_n.reshape(-1)
        #     sdf_n = torch.from_numpy(sdf_n).to(device)
        #     sdf_mask = (sdf_n == -1)
        mesh = model(None, dcoords=coords, dfeats=ff, training=True, mask=sdf_mask)
        mesh2smplx(name, mesh, output_dir='outputs')
        # distances, face_id, uvw = BVH.unsigned_distance(mesh.vertices, return_uvw=True)
        # dis_mask = torch.topk(distances, k=bvh_v.shape[0], largest=False)[1]
        # # dis_mask = distances < 0.005
        # source = mesh.vertices[dis_mask]
        # target = bvh_v[bvh_f[face_id[dis_mask]]]
        # target_v = torch.einsum('nij, ni->nj', target, uvw[dis_mask])
        # loss_geo = hub(source, target_v)

        n = torch.cat(
            [
                mesh.vertices[..., :2].clone(),
                mesh.vertices[..., -1:].clone() - z_mean
            ],
            dim=-1
        )
        mesh.vertices = n
        dicts = render(mesh, extrinsic@rotas_t, proj[0], HW=[H, W])
        # print(dicts['depth'].min())
        # print(dicts['depth'].max())
        # exit()
        # B, C, H, W = dicts['normal'].shape
        # out = dicts['normal'].permute(2, 0, 3, 1).reshape(H, B*W, C).detach().cpu().numpy() * 255.
        # out1 = dicts['normal'].permute(2, 0, 3, 1).reshape(H, B*W, C).detach().cpu().numpy() * 255.

        

        # with torch.no_grad():
        #     smplx_dicts = render(smplx_m, extrinsic, proj[0], HW=[H, W])

        

        # depth_loss_eye = 10 * l1_loss(dicts['depth'][0][eye_mask[0, 0] > 0], smplx_dicts['depth'][eye_mask[0, 0] > 0])
        # perceptual_loss_eye = l1_loss(dicts['normal'][0][eye_mask[0] > 0], smplx_dicts['normal'][eye_mask[0] > 0])

        # I_dep = torch.cat(
        #     [
        #         dicts['depth'][0],
        #         smplx_dicts['depth']
        #     ],
        #     dim=-1
        # )
        # print(dicts_gt['depth'].max())
        # print(dicts['depth'].max())
        # print(dicts['depth'].shape)
        # print(dicts['normal'].shape)
        # print(smplx_dicts['normal'].shape)
        # I_nor = torch.cat(
        #     [
        #         dicts['normal'][0],
        #         smplx_dicts['normal']
        #     ],
        #     dim=-1
        # )
        # I_dep1 = torch.cat(
        #     [
        #         dicts['depth'][0],
        #         smplx_dicts['depth']
        #     ],
        #     dim=-1
        # )
        # I_nor1 = torch.cat(
        #     [
        #         dicts['normal'][0] * eye_mask[0],
        #         smplx_dicts['normal'] * eye_mask[0]
        #     ],
        #     dim=-1
        # )
        # cv2.imwrite(f'outputs/{name}/objects/nn11.png', dicts_gt['normal'][0].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)
        # # cv2.imwrite(f'outputs/{name}/objects/local_normal.png', dicts_gt_['normal'][1:].permute(1, 2, 0, 3).flatten(start_dim=2).permute(1, 2, 0).detach().cpu().numpy() * 255.)
        # # cv2.imwrite(f'outputs/{name}/objects/local_normal_2.png', dicts['normal'][1:].permute(1, 2, 0, 3).flatten(start_dim=2).permute(1, 2, 0).detach().cpu().numpy() * 255.)
        # exit()

        # print(mesh.vertices.shape)
        # exit()
        # meshes = Meshes(verts=[mesh.vertices], faces=[mesh.faces])
        # loss_laplacian = mesh_laplacian_smoothing(meshes, method="uniform")

        depth_loss = cal_depth_loss(dicts['depth'][0], depth_gt.clone(), local_normal_mask, parsing_map.clone(), face_ldmks.clone())
        # depth_loss = 10 * l1_loss(dicts['depth'][0], face_mask[0, 0] * smplx_dicts['depth'] + (1 - face_mask[0, 0]) * dicts_gt['depth'][0])
        # perceptual_loss = loss_recon(dicts['normal'][0], smplx_dicts['normal'] * face_mask[0] + (1 - face_mask[0]) * dicts_gt['normal'][0])
        # mmask = smplx_dicts_local['mask'] != 0
        # depth_loss_local = 10 * l1_loss(dicts['depth'][1:], smplx_dicts_local['depth'] * local_mask + (1 - local_mask) * dicts_gt_['depth'][1:])
        # perceptual_loss_local = l1_loss(dicts['normal'][1:], smplx_dicts_local['normal'] * local_mask[:, None] + (1 - local_mask[:, None]) * dicts_gt_['normal'][1:])
        # perceptual_loss_local = loss_recon(dicts['normal'][1:] * local_mask_dilated[1:, None], smplx_dicts_local['normal'][1:] * local_mask_dilated[1:, None]) + 0.8 * loss_recon((1 - local_mask[1:, None]) * dicts['normal'][1:], (1 - local_mask[1:, None]) * dicts_gt_['normal'][1:])
        # loss_reg = mesh.reg_loss

        # img_n1 = smplx_dicts_local['normal'].permute(1, 2, 0, 3).flatten(start_dim=-2).permute(1, 2, 0)
        # img_n2 = dicts['normal'][1:].permute(1, 2, 0, 3).flatten(start_dim=-2).permute(1, 2, 0)
        # img = torch.cat([img_n1, img_n2], dim=1)
        # cv2.imwrite('src/snnwzj.png', img.detach().cpu().numpy() * 255.)
        # loss = depth_loss + 2 * perceptual_loss + 0.1 * loss_reg + 5 * depth_loss_local + 10 * perceptual_loss_local
        # depth_loss_new = 10 * loss_recon(dicts['depth'][0, None] * local_mask_dilated[0, None], smplx_dicts_local['depth'][0, None] * local_mask_dilated[0, None]) + 5 * loss_recon((1 - local_mask)[0, None] * dicts['depth'][0, None], (1 - local_mask)[0, None] * dicts_gt_['depth'][0, None])

        # perceptual_loss_new = loss_recon(dicts['normal'][0], smplx_dicts_local['normal'][0] * local_mask_dilated[0, None] + (1 - local_mask[0, None]) * dicts_gt_['normal'][0]) ### 2025 11.9 22:33
        # perceptual_loss_new = 10 * loss_recon(dicts['normal'][0] * local_mask_dilated[0, None], smplx_dicts_local['normal'][0] * local_mask_dilated[0, None]) + 5 * loss_recon(dicts['normal'][0] * (1 - local_mask[0, None]), (1 - local_mask[0, None]) * dicts_gt_['normal'][0])

        # loss = 10 * depth_loss + 2 * perceptual_loss + 2 * perceptual_loss_local + 1.5 * loss_laplacian
        # loss = 1 * perceptual_loss_new + 0.5 * perceptual_loss_local + 0.2 * depth_loss
        # print(dicts['normal'].shape)
        # print(normal_gt.shape)
        # exit()
        # my_mask = ((dicts['mask'][0] * seg_mask) > 0.5).float()
        # normal_loss = cal_l1_loss(dicts['normal'][0] - normal_gt.permute(2, 0, 1), ((dicts['mask'][0] * seg_mask) > 0.5).float()) + 0.2 * (1 - ssim(dicts['normal'][0] * my_mask, normal_gt.permute(2, 0, 1) * my_mask))
        # loss = normal_loss
        # if epoch < 100:
        #     lambda_depth = 10
        # else:
        if epoch < 1001:
            lambda_depth = 10
            lambda_ssim = 1
            lambda_norm = 0
        else:
            lambda_depth = 6
            lambda_ssim = 0.2
            lambda_norm = 1
        # print(new_mask.shape)
        # exit()
        perceptual_normal_loss = loss_recon(dicts['normal'][0] * local_normal_mask[None], normal_gt.permute(2, 0, 1) * local_normal_mask[None], lambda_ssim=8, lambda_norm=0) 
        perceptual_normal_loss_dis = 0.3 * loss_recon(dicts['normal'][0] * (1 - local_reg_mask[None]), dicts_gt['normal'][0] * (1 - local_reg_mask[None]), lambda_ssim=3, lambda_norm=0)

        pred_ = torch.nn.functional.normalize(dicts['normal'][0], dim=0, eps=1e-6)
        gt_ = torch.nn.functional.normalize(normal_gt.permute(2, 0, 1), dim=0, eps=1e-6)
        dot = (pred_ * gt_).sum(dim=0)
        loss_local_ = ((1 - dot.clamp(-1, 1)) * local_normal_mask).sum() / (local_normal_mask.sum() + 1e-6)

        normal_1 = loss_recon(dicts['normal'][1], dicts_gt_['normal'][1], lambda_ssim=3, lambda_norm=0)
        normal_2 = loss_recon(dicts['normal'][2], dicts_gt_['normal'][2], lambda_ssim=3, lambda_norm=0)
        normal_3 = loss_recon(dicts['normal'][3], dicts_gt_['normal'][3], lambda_ssim=3, lambda_norm=0)


        reg_loss = mesh.reg_loss
        # loss = normal_loss + lambda_depth * depth_loss  + 0.05 * normal_1 + 0.05 * normal_2 + 0.05 * normal_3
        loss = perceptual_normal_loss + 1.5 * perceptual_normal_loss_dis + 0.3 * loss_local_ + 0.1 * torch.mean(torch.abs(delta)) + 0.5 * normal_1 + 0.5 * normal_2 + 0.25 * normal_3 + 30 * depth_loss
        # k = normal_gt.permute(2, 0, 1) * my_mask[None]
        # k = (k / 2. + 0.5) * 255.
        # p = (dicts['normal'][0] / 2. + 0.5) * 255.
        # cv2.imwrite('normal2.png', k.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1])
        # cv2.imwrite('normal1.png', p.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1])
        # exit()
        # loss = depth_loss + 2 * perceptual_loss  + depth_loss_eye + 2 * perceptual_loss_eye
        # loss = depth_loss_eye + 2 * perceptual_loss_eye
        # loss = depth_loss + 100 * depth_loss_eye
        # loss = depth_loss_local + 2 * perceptual_loss_local
        
        # loss.backward()
        # optimizer_sing.step()
        # scheduler.step()
        # grad_norm = grad_clip([delta])
        if epoch % 5 == 0:
            print(f'epoch: {epoch}, loss: {loss}, normal_loss: {loss_local_}, depth_loss: {depth_loss}, lr: {scheduler_d.get_lr()}')

        #TODO 通过mask选出smplx对应的面，然后对于侧面视角和仰视视角，渲染pred + smplx筛出的面部作为gt，因为此时面部高度重合，只是为了把接缝处处理一下
        #TODO 如果一开始面部过于偏怎么办，去掉这些例子算了
        loss.backward()
        # grad_norm = grad_clip([delta])

        optimizer_d.step()
        # scheduler_d.step()
        # scaled_loss = loss * (2 ** log_scale)
        # scaled_loss.backward()

        # # # print(model_params)
        # # # exit()

        # model_grads_to_master_grads(model_params, master_params)
        # master_params[0].grad.mul_(1.0 / (2 ** log_scale))
        # grad_norm = grad_clip(master_params)

        # if not any(not p.grad.isfinite().all() for p in model_params):
        #     optimizer.step()
        #     scheduler.step()
        #     master_params_to_model_params(model_params, master_params)
        #     log_scale += fp16_scale_growth
        # else:
        #     log_scale -= 1

        # if epoch % 100 == 0:
        #     mesh.vertices[..., -1:] += z_mean
        #     trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.cpu().numpy()).export(f'outputs/{name}/objects/single_{epoch}.obj')

        if epoch == iters:
            mesh.vertices[..., -1:] += z_mean
            trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.cpu().numpy()).export(f'outputs/{name}/objects/single_show_.obj')
            torch.save(ff, f'outputs/{name}/params/delta_geo_david_.pt')
            # break
            # cv2.imwrite(f'outputs/{name}/objects/dd.png', I_dep.detach().cpu().numpy()[..., ::-1] * 255.)
            # cv2.imwrite(f'outputs/{name}/objects/nn.png', I_nor.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)
            # cv2.imwrite(f'outputs/{name}/objects/dd1.png', I_dep1.detach().cpu().numpy()[..., ::-1] * 255.)
            # cv2.imwrite(f'outputs/{name}/objects/nn1.png', I_nor1.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)


            # cv2.imwrite(f'outputs/{name}/objects/local_mask.png', local_mask.permute(1, 0, 2).flatten(start_dim=1).detach().cpu().numpy() * 255.)
            # cv2.imwrite(f'outputs/{name}/objects/local_normal.png', ((1 - local_mask[1:, None]) * dicts_gt_['normal'][1:]).permute(1, 2, 0, 3).flatten(start_dim=2).permute(1, 2, 0).detach().cpu().numpy() * 255.)
            
        if epoch % 5 == 0:
            # n = dicts['normal']
            cv2.imwrite(f'outputs/{name}/geo/normal_{epoch}.png', dicts['normal'][0].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)

        if epoch == iters:
            # n = dicts['normal']
            cv2.imwrite(f'outputs/{name}/geo/normal_{epoch}.png', dicts['normal'][0].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)


def main_new(name):
    os.makedirs(f'{OUTPUT_PATH}/{name}/geo', exist_ok=True)

    track_path = f'inputs/{name}/body_track/smplx_track.pth'
    print(track_path)
    parse = read_png(f'inputs/{name}/parsing')
    H, W = parse.shape[1:3]
    gt_mask = torch.from_numpy((parse!=0).astype(np.uint8))
    ffm = ((parse == 2) | ((parse > 5) & (parse < 14)))
    eye = ((parse == 8) | (parse == 9))
    eye_mask = (eye.astype(np.uint8) * (parse != 0)).astype(np.uint8)
    face_mask = (ffm.astype(np.uint8) * (parse!=0)).astype(np.uint8)
    smplx_params = torch.load(track_path)
    cam_para, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, shape, expr, global_orient, transl = to_device(**smplx_params)
    proj = get_ndc_proj_matrix(cam_para[0:1], [H, W])
    extrinsic = torch.tensor(
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ]
    ).to(device)
    batch_size = body_pose.shape[0]
    lhand_pose = smplx.left_hand_pose.expand(batch_size, -1)
    rhand_pose = smplx.right_hand_pose.expand(batch_size, -1)
    # leye_pose = smplx.leye_pose.expand(batch_size, -1)
    # reye_pose = smplx.reye_pose.expand(batch_size, -1)
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


    coarse_feats = torch.load(f'outputs/{name}/slats/feats_coarse_back.pt').to(device)
    coarse_coords = torch.load(f'outputs/{name}/slats/coords_coarse_back.pt').to(device)
    feats = torch.load(f'outputs/{name}/slats/feats_0_new.pt').to(device)
    coords = torch.load(f'outputs/{name}/slats/coords_0_new.pt').to(device)


    from trellis.representations.mesh import SparseFeatures2Mesh
    model = SparseFeatures2Mesh(res=256, use_color=True)

    delta = torch.zeros((feats.shape[0], 8)).to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=5e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.75)
    face_mask = torch.from_numpy(face_mask).to(device).permute(0, 3, 1, 2)
    eye_mask = torch.from_numpy(eye_mask).to(device).permute(0, 3, 1, 2)
    with torch.no_grad():
        temp_d = model(None, dcoords=coords, dfeats=feats, debug=True)
        sdf_d = temp_d.reshape((257, 257, 257))
        sdf_d = sdf_d.detach().cpu().numpy()
        sdf_mask = wzj_final(
            sdf_d,
        )
        sdf_mask = sdf_mask.reshape(-1)
        sdf_mask = torch.from_numpy(sdf_mask).to(device)
        print(sdf_mask.sum())
    with torch.no_grad():
        temp_d = model(None, dcoords=coords, dfeats=feats, debug=True)

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
    
    rotas_t = torch.Tensor([
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ],
        [
            [math.cos(math.pi / 4), 0, -math.sin(math.pi / 4), 0],
            [0, 1, 0, 0],
            [math.sin(math.pi / 4), 0, math.cos(math.pi / 4), 0],
            [0, 0, 0, 1]
        ],
        [
            [math.cos(-math.pi / 4), 0, -math.sin(-math.pi / 4), 0],
            [0, 1, 0, 0],
            [math.sin(-math.pi / 4), 0, math.cos(-math.pi / 4), 0],
            [0, 0, 0, 1]
        ],
        [
            [1, 0, 0, 0],
            [0, math.cos(-math.pi / 4), -math.sin(-math.pi / 4), 0],
            [0, math.sin(-math.pi / 4), math.cos(-math.pi / 4), 0],
            [0, 0, 0, 1]
        ]
    ]).to(device)
    with torch.no_grad():
        mesh_gt = model(None, dcoords=coords, dfeats=feats, training=False, mask=sdf_mask)
        mesh2smplx(name, mesh_gt)
        z_mean = mesh_gt.vertices[..., -1].mean().item()
        dicts_gt = render(mesh_gt, extrinsic@rotas_t, proj[0], HW=[H, W])

    rotas = torch.Tensor([
        [
            [math.cos(math.pi / 4), 0, -math.sin(math.pi / 4), 0],
            [0, 1, 0, 0],
            [math.sin(math.pi / 4), 0, math.cos(math.pi / 4), 0],
            [0, 0, 0, 1]
        ],
        [
            [math.cos(-math.pi / 4), 0, -math.sin(-math.pi / 4), 0],
            [0, 1, 0, 0],
            [math.sin(-math.pi / 4), 0, math.cos(-math.pi / 4), 0],
            [0, 0, 0, 1]
        ],
        [
            [1, 0, 0, 0],
            [0, math.cos(-math.pi / 4), -math.sin(-math.pi / 4), 0],
            [0, math.sin(-math.pi / 4), math.cos(-math.pi / 4), 0],
            [0, 0, 0, 1]
        ]
    ]).to(device)

    with torch.no_grad():
        index = smplx_v.shape[1] - 1092 + 1
        mask = ~(smplx.faces > index).any(axis=-1)
        ff = smplx.faces[mask]
        p = smplx_v[0, :-1092]
        v, f = densify(p, ff)
        # v, f = densify(smplx_v[0], smplx.faces)

        v1, f1 = densify(smplx_v[0], smplx.faces)
        # region = trimesh.Trimesh(vertices=v1, faces=f1).submesh([face_region])[0]
        # bvh_v = torch.from_numpy(region.vertices).to(device).float()
        # bvh_f = torch.from_numpy(region.faces).to(device).int()
        # BVH = cubvh.cuBVH(bvh_v, bvh_f)
        # v, f = densify(p, ff)
        smplx_m = MeshExtractResult(
            vertices=torch.from_numpy(v).to(device).float(),
            faces=torch.from_numpy(f).to(device).int()
        )
        smplx_m1 = MeshExtractResult(
            vertices=torch.from_numpy(p).to(device).float(),
            faces=torch.from_numpy(ff).to(device).float()
        )
        smplx_rast = render(smplx_m1, extrinsic, proj[0], HW=[H, W], only_rast=True)
        m_d = (smplx_rast[..., -1] * face_mask[0])
        tris = m_d[m_d > 0].flatten().unique() - 1
        t_m = trimesh.Trimesh(vertices=smplx_v[0], faces=smplx.faces)
        sub_sss = np.load('params/smplx_faces_no_eye.npy')
        t_m = t_m.submesh([sub_sss])[0]
        smplx_m_local = MeshExtractResult(
            vertices=torch.from_numpy(t_m.vertices).to(device).float(),
            faces=torch.from_numpy(t_m.faces).to(device).int()
        )

        smplx_m.vertices[..., -1] -= z_mean
        smplx_m_local.vertices[..., -1] -= z_mean
        mesh_gt.vertices[..., -1] -= z_mean

        extrinsic[2, 3] = z_mean
        smplx_dicts = render(smplx_m, extrinsic, proj[0], HW=[H, W])
        smplx_dicts_l = render(smplx_m_local, extrinsic@rotas_t, proj[0], HW=[H, W])
        smplx_dicts_local = render(smplx_m, extrinsic@rotas_t, proj[0], HW=[H, W])
        dicts_gt_ = render(mesh_gt, extrinsic@rotas_t, proj[0], HW=[H, W])

        local_mask = smplx_dicts_l['mask']
        print(smplx_dicts['mask'].shape)
        local = smplx_dicts_local['mask']
    
    masks = []
    structure = generate_binary_structure(2, 1)

    for x in local_mask:
        masks.append(binary_dilation(x.detach().cpu().numpy(), structure=structure, iterations=4))
    local_mask_dilated = np.stack(masks, axis=0)
    # local_mask_dilated = binary_dilation(local_mask.detach().cpu().numpy(), structure=structure, iterations=1)
    local_mask_dilated = torch.from_numpy(local_mask_dilated).to(local_mask)

    input = sp.SparseTensor(
        coords=coarse_coords.to(dtype=torch.int32),
        feats=coarse_feats
    )
    iters = 400
    grad_clip = AdaptiveGradClipper(max_norm=0.01, clip_percentile=95)

    for epoch in range(iters + 1):
        optimizer.zero_grad()

        # optimizer_sing.zero_grad()
        ff = torch.cat(
            [
                feats[..., :53].detach() + delta,
                feats[..., 53:].detach()
            ],
            dim=-1
        ).to(device)

        # with torch.no_grad():
        #     sdf_d = model(None, dcoords=coords, dfeats=ff, debug=True)
        #     sdf_d = sdf_d.reshape((257, 257, 257))
        #     sdf_d = sdf_d.detach().cpu().numpy()
        #     sdf_n = flood_fill(sdf_d)
        #     sdf_n = sdf_n.reshape(-1)
        #     sdf_n = torch.from_numpy(sdf_n).to(device)
        #     sdf_mask = (sdf_n == -1)
        mesh = model(None, dcoords=coords, dfeats=ff, training=False, mask=sdf_mask)
        mesh2smplx(name, mesh)
        # distances, face_id, uvw = BVH.unsigned_distance(mesh.vertices, return_uvw=True)
        # dis_mask = torch.topk(distances, k=bvh_v.shape[0], largest=False)[1]
        # # dis_mask = distances < 0.005
        # source = mesh.vertices[dis_mask]
        # target = bvh_v[bvh_f[face_id[dis_mask]]]
        # target_v = torch.einsum('nij, ni->nj', target, uvw[dis_mask])
        # loss_geo = hub(source, target_v)

        n = torch.cat(
            [
                mesh.vertices[..., :2].clone(),
                mesh.vertices[..., -1:].clone() - z_mean
            ],
            dim=-1
        )
        mesh.vertices = n
        dicts = render(mesh, extrinsic@rotas_t, proj[0], HW=[H, W])
        # B, C, H, W = dicts['normal'].shape
        # out = dicts['normal'].permute(2, 0, 3, 1).reshape(H, B*W, C).detach().cpu().numpy() * 255.
        # out1 = dicts['normal'].permute(2, 0, 3, 1).reshape(H, B*W, C).detach().cpu().numpy() * 255.

        

        # with torch.no_grad():
        #     smplx_dicts = render(smplx_m, extrinsic, proj[0], HW=[H, W])

        

        depth_loss_eye = 10 * l1_loss(dicts['depth'][0][eye_mask[0, 0] > 0], smplx_dicts['depth'][eye_mask[0, 0] > 0])
        perceptual_loss_eye = l1_loss(dicts['normal'][0][eye_mask[0] > 0], smplx_dicts['normal'][eye_mask[0] > 0])

        I_dep = torch.cat(
            [
                dicts['depth'][0],
                smplx_dicts['depth']
            ],
            dim=-1
        )
        I_nor = torch.cat(
            [
                dicts['normal'][0],
                smplx_dicts['normal']
            ],
            dim=-1
        )
        I_dep1 = torch.cat(
            [
                dicts['depth'][0],
                smplx_dicts['depth']
            ],
            dim=-1
        )
        I_nor1 = torch.cat(
            [
                dicts['normal'][0] * eye_mask[0],
                smplx_dicts['normal'] * eye_mask[0]
            ],
            dim=-1
        )
        meshes = Meshes(verts=[mesh.vertices], faces=[mesh.faces])
        loss_laplacian = mesh_laplacian_smoothing(meshes, method="uniform")

        depth_loss = 10 * l1_loss(dicts['depth'][0], face_mask[0, 0] * smplx_dicts['depth'] + (1 - face_mask[0, 0]) * dicts_gt['depth'][0])
        perceptual_loss = loss_recon(dicts['normal'][0], smplx_dicts['normal'] * face_mask[0] + (1 - face_mask[0]) * dicts_gt['normal'][0])
        mmask = smplx_dicts_local['mask'] != 0
        perceptual_loss_local = loss_recon(dicts['normal'][1:] * local_mask_dilated[1:, None], smplx_dicts_local['normal'][1:] * local_mask_dilated[1:, None]) + 0.8 * loss_recon((1 - local_mask[1:, None]) * dicts['normal'][1:], (1 - local_mask[1:, None]) * dicts_gt_['normal'][1:])
        depth_loss_new = 10 * loss_recon(dicts['depth'][0, None] * local_mask_dilated[0, None], smplx_dicts_local['depth'][0, None] * local_mask_dilated[0, None]) + 5 * loss_recon((1 - local_mask)[0, None] * dicts['depth'][0, None], (1 - local_mask)[0, None] * dicts_gt_['depth'][0, None])
        perceptual_loss_new = 10 * loss_recon(dicts['normal'][0] * local_mask_dilated[0, None], smplx_dicts_local['normal'][0] * local_mask_dilated[0, None]) + 5 * loss_recon(dicts['normal'][0] * (1 - local_mask[0, None]), (1 - local_mask[0, None]) * dicts_gt_['normal'][0])
        loss = 1 * perceptual_loss_new + 0.5 * perceptual_loss_local + 0.2 * depth_loss_new
        
        # loss.backward()
        # optimizer_sing.step()
        # scheduler.step()
        # grad_norm = grad_clip([delta])
        if epoch % 5 == 0:
            print(f'epoch: {epoch}, loss: {loss}, depth_loss: {depth_loss}')

        #TODO 通过mask选出smplx对应的面，然后对于侧面视角和仰视视角，渲染pred + smplx筛出的面部作为gt，因为此时面部高度重合，只是为了把接缝处处理一下
        #TODO 如果一开始面部过于偏怎么办，去掉这些例子算了
        loss.backward()
        grad_norm = grad_clip(delta)
        optimizer.step()
        scheduler.step()

        # scaled_loss = loss * (2 ** log_scale)
        # scaled_loss.backward()

        # # print(model_params)
        # # exit()

        # model_grads_to_master_grads(model_params, master_params)
        # master_params[0].grad.mul_(1.0 / (2 ** log_scale))
        # grad_norm = grad_clip(master_params)

        # if not any(not p.grad.isfinite().all() for p in model_params):
        #     optimizer.step()
        #     master_params_to_model_params(model_params, master_params)
        #     log_scale += fp16_scale_growth
        # else:
        #     log_scale -= 1

        if epoch == iters:
            mesh.vertices[..., -1:] += z_mean
            trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.cpu().numpy()).export(f'outputs/{name}/objects/single.obj')
            torch.save(ff, f'outputs/{name}/params/delta_geo.pt')
            cv2.imwrite(f'outputs/{name}/objects/dd.png', I_dep.detach().cpu().numpy()[..., ::-1] * 255.)
            cv2.imwrite(f'outputs/{name}/objects/nn.png', I_nor.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)
            cv2.imwrite(f'outputs/{name}/objects/dd1.png', I_dep1.detach().cpu().numpy()[..., ::-1] * 255.)
            cv2.imwrite(f'outputs/{name}/objects/nn1.png', I_nor1.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)


            cv2.imwrite(f'outputs/{name}/objects/local_mask.png', local_mask.permute(1, 0, 2).flatten(start_dim=1).detach().cpu().numpy() * 255.)
            cv2.imwrite(f'outputs/{name}/objects/local_normal.png', ((1 - local_mask[1:, None]) * dicts_gt_['normal'][1:]).permute(1, 2, 0, 3).flatten(start_dim=2).permute(1, 2, 0).detach().cpu().numpy() * 255.)
            
        if epoch % 5 == 0:
            # n = dicts['normal']
            cv2.imwrite(f'outputs/{name}/geo/normal_{epoch}.png', dicts['normal'][0].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)

def smplx2mesh(name):
    inputdir = cfg['input_dir']
    track_path = f'{inputdir}/{name}/body_track/smplx_track.pth'
    smplx_params = torch.load(track_path)
    cam_para, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, shape, expr, global_orient, transl = to_device(**smplx_params)
    batch_size = body_pose.shape[0]
    lhand_pose = smplx.left_hand_pose.expand(batch_size, -1)
    rhand_pose = smplx.right_hand_pose.expand(batch_size, -1)
    # leye_pose = smplx.leye_pose.expand(batch_size, -1)
    # reye_pose = smplx.reye_pose.expand(batch_size, -1)
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
    p = output.vertices[0].detach().cpu().numpy()
    f = smplx.faces
    trimesh.Trimesh(vertices=p, faces=f).export(f'{OUTPUT_PATH}/{name}/objects/origin_smplx.obj')
    # f = smplx.faces[mask]
    # p = output.vertices[0, :-1092].detach().cpu().numpy()
    trimesh.Trimesh(vertices=p, faces=f).export(f'{OUTPUT_PATH}/{name}/objects/origin_smplx_wo_eyes.obj')
    trans = np.load(f'{OUTPUT_PATH}/{name}/params/trans.npz')
    kt = np.load(f'{OUTPUT_PATH}/{name}/params/kt.npz')
    k, t = list(kt.values())
    s, R, T = list(trans.values())
    p = k * p + t
    from utils3d.torch import quaternion_to_matrix
    Rota = quaternion_to_matrix(torch.from_numpy(R)).squeeze(0)

    p = s * p @ Rota.transpose(-2, -1).cpu().numpy() + T
    aligned_smplx = trimesh.Trimesh(vertices=p, faces=f)
    aligned_smplx.export(f'{OUTPUT_PATH}/{name}/objects/sm.obj')
    aligned_smplx = trimesh.load(f'{OUTPUT_PATH}/{name}/objects/sm.obj')
    test = (f - aligned_smplx.faces).sum()
    print((p - aligned_smplx.vertices).sum())
    print(test)


def bind_no_eye(name, mesh=None, mouth=False):


    from trellis.representations.mesh import SparseFeatures2Mesh
    model = SparseFeatures2Mesh(res=256, use_color=True)
    coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_0_new.pt').to(device)
    feats = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_0_new.pt').to(device)
    if mouth:
        feats_n = torch.load(f'{OUTPUT_PATH}/{name}/params/delta_geo_mouth_new.pt').to(device)
    else:
        feats_n = torch.load(f'{OUTPUT_PATH}/{name}/params/{pt_name}.pt').to(device)


    # smplx = trimesh.load(f'../../output/{name}/objects/smplx.obj')
    if mesh is None:
        smplx = trimesh.load(f'{OUTPUT_PATH}/{name}/objects/sm.obj')
    else:
        smplx = mesh
    input = sp.SparseTensor(
        coords=coords,
        feats=feats_n
    )

    

    with torch.no_grad():
        temp_d = model(None, dcoords=coords, dfeats=feats, debug=True)
        sdf_d = temp_d.reshape((257, 257, 257))
        sdf_d = sdf_d.detach().cpu().numpy()
        sdf_mask = wzj_final(
            sdf_d,
        )
        # exit()
        sdf_mask = sdf_mask.reshape(-1)
        sdf_mask = torch.from_numpy(sdf_mask).to(device)

    with torch.no_grad():
        temp_d = model(None, dcoords=coords, dfeats=feats, debug=True)

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
    output = model(input, return_v_a=True, mask=sdf_mask, name=name, output_path=OUTPUT_PATH)
    # trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy(), faces=output.faces.detach().cpu().numpy()).export('src/tutils/sm_.obj')
    if hasattr(output, 'v_p'):
        print(output.v_p.shape)
    v_p = output.v_p.to(device)
    # v_p = (v_p / 256) - 0.5
    v_bvh, f_bvh = densify(smplx.vertices, smplx.faces)
    BVH = cubvh.cuBVH(torch.from_numpy(v_bvh).to(device), torch.from_numpy(f_bvh).to(device=device, dtype=torch.int32))
    distance, face_id, uvw = BVH.unsigned_distance(v_p, return_uvw=True)
    # np.savez(f'../../output/{name}/slats/motion_sdf.npz', face_id=face_id.detach().cpu().numpy(), uvw=uvw.detach().cpu().numpy())
    np.savez(f'{OUTPUT_PATH}/{name}/slats/motion_sdf.npz', face_id=face_id.detach().cpu().numpy(), uvw=uvw.detach().cpu().numpy())


def filter_z(mesh, mean):
    all_v = mesh.vertices[mesh.faces] #(N, 3, 3)
    # params = mesh.vertices.new_ones((*mesh.faces.shape)) * 1 / 3
    # face_v = torch.einsum('bij, bi -> bj', all_v, params)
    face_v = all_v.mean(dim=1)
    face_v_z = face_v[..., -1]
    face_mask = face_v_z > mean

    new_faces = mesh.faces[face_mask] #(F, 3)
    iso_faces, indices = new_faces.flatten().unique(return_inverse=True)

    new_vers = mesh.vertices[iso_faces]
    new_faces = indices.reshape(-1, 3)

    return new_vers, new_faces


def boundary_smooth_loss(boundary_points):
    """
    boundary_points: [N, 3] 有序的边界点 loop
    """
    # 1. 构建前后邻居
    # roll(-1) 把数组左移，得到 P_{i+1}
    # roll(1) 把数组右移，得到 P_{i-1}
    next_pts = torch.roll(boundary_points, shifts=-1, dims=0)
    prev_pts = torch.roll(boundary_points, shifts=1, dims=0)
    
    # 2. 目标位置是邻居的中点
    target_pos = (next_pts + prev_pts) / 2.0
    
    # 3. Loss
    loss = torch.nn.functional.mse_loss(boundary_points, target_pos)
    
    return loss


def get_boundary_edges(faces):
    """
    找出只出现一次的边 (边界边)。
    faces: [F, 3] LongTensor
    返回: boundary_edges [N_boundary, 2]
    """
    # 1. 把所有三角形的边拆出来 (v0-v1, v1-v2, v2-v0)
    # 也就是把 [F, 3] 变成 [3F, 2]
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=0)
    
    # 2. 排序，保证 (0, 1) 和 (1, 0) 是一样的
    edges, _ = torch.sort(edges, dim=1)
    
    # 3. 统计每条边出现的次数
    # unique_edges: 唯一的边
    # counts: 每条边出现的次数
    unique_edges, counts = torch.unique(edges, return_counts=True, dim=0)
    
    # 4. 只保留出现次数为 1 的边 (这就是边界！)
    boundary_edges = unique_edges[counts == 1]
    
    return boundary_edges

def boundary_length_loss(verts, faces):
    # 1. 获取边界边索引 [N, 2]
    # 注意：如果拓扑结构不变，这一步可以在循环外预计算一次
    b_edges = get_boundary_edges(faces) 
    
    if b_edges.shape[0] == 0:
        return torch.tensor(0.0).to(verts.device)
    
    # 2. 获取端点坐标
    v0 = verts[b_edges[:, 0]]
    v1 = verts[b_edges[:, 1]]
    
    # 3. 计算边长平方
    # 让所有边界边尽可能短 -> 就像拉紧橡皮筋 -> 锯齿被拉直
    loss = torch.sum((v0 - v1) ** 2)
    
    return loss

def boundary_smoothness_loss_approx(verts, faces):
    """
    近似的边界平滑 Loss (1D Laplacian)。
    让边界上的每个点，尽可能位于其“边界邻居”的中心。
    这会消除锯齿（高频抖动），使边缘变得圆滑，比单纯的边长 Loss 更好（不会过度收缩）。
    """
    
    # -----------------------------------------------------------
    # 1. 提取边界边 (Boundary Edges)
    # -----------------------------------------------------------
    # 构造所有边: [F, 3] -> [3F, 2]
    # edges 包含 (v0,v1), (v1,v2), (v2,v0)
    all_edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=0)
    
    # 排序边索引，确保 (u, v) 和 (v, u) 被视为同一条边
    all_edges_sorted, _ = torch.sort(all_edges, dim=1)
    
    # 统计每条边出现的次数
    unique_edges, counts = torch.unique(all_edges_sorted, return_counts=True, dim=0)
    
    # 边界边只出现 1 次
    # boundary_edges: [E_b, 2]
    boundary_edges = unique_edges[counts == 1]
    
    if boundary_edges.shape[0] == 0:
        return torch.tensor(0.0, device=verts.device, requires_grad=True)

    # -----------------------------------------------------------
    # 2. 构建边界邻接关系 (无需排序链表，利用 index_add)
    # -----------------------------------------------------------
    
    # 我们需要计算每个边界点的“边界邻居中心” (Centroid of boundary neighbors)
    # Target_Pos[i] = Sum(Neighbors_of_i) / Degree_of_i
    
    # 初始化累加器
    num_verts = verts.shape[0]
    neighbor_sum = torch.zeros_like(verts) # [V, 3]
    degree = torch.zeros((num_verts, 1), device=verts.device) # [V, 1]
    
    # 获取边界边的两个端点索引
    u = boundary_edges[:, 0]
    v = boundary_edges[:, 1]
    
    # 获取端点坐标
    p_u = verts[u]
    p_v = verts[v]
    
    # 把 v 的坐标加到 u 的邻居和里
    neighbor_sum.index_add_(0, u, p_v)
    degree.index_add_(0, u, torch.ones_like(degree[u]))
    
    # 把 u 的坐标加到 v 的邻居和里
    neighbor_sum.index_add_(0, v, p_u)
    degree.index_add_(0, v, torch.ones_like(degree[v]))
    
    # -----------------------------------------------------------
    # 3. 计算 Loss (Uniform 1D Laplacian)
    # -----------------------------------------------------------
    
    # 只取那些度数 > 0 的点（也就是边界点）
    # 通常边界点的度数应该是 2（连接前后两个点）。
    # 如果度数不是 2，说明拓扑比较复杂（8字形或非流形），但平均值逻辑依然有效。
    mask = degree.squeeze() > 0
    
    # 计算目标位置 (邻居的平均值)
    # 加上 1e-6 防止除以 0
    target_pos = neighbor_sum[mask] / (degree[mask] + 1e-6)
    
    # 当前位置
    current_pos = verts[mask]
    
    # Loss = || P_current - P_target ||^2
    # 这就是标准的 Laplacian Smoothing，但是只作用在边界线上
    loss = torch.nn.functional.mse_loss(current_pos, target_pos)
    
    return loss




def inner_hole_smooth_loss(verts, full_faces, ring_mask_indices):
    """
    只针对 Mesh 内部挖空的洞（内圈）进行平滑。
    利用“外圈是连接的，内圈是断开的”这一拓扑特性。
    
    verts: [V, 3] 全局顶点
    full_faces: [F, 3] 全局面索引 (必须包含环和周围的连接面)
    ring_mask_indices: [M] 属于这个环区域的顶点索引 (LongTensor)
    """
    
    # -----------------------------------------------------------
    # 1. 寻找全网格的边界边 (Global Boundary Edges)
    # -----------------------------------------------------------
    # 注意：连接处（外圈）的边因为被共用，count 会是 2，会被自动过滤掉。
    # 只有真正的孔洞边缘，count 才是 1。
    
    all_edges = torch.cat([
        full_faces[:, [0, 1]],
        full_faces[:, [1, 2]],
        full_faces[:, [2, 0]]
    ], dim=0)
    
    # 排序并去重
    all_edges_sorted, _ = torch.sort(all_edges, dim=1)
    unique_edges, counts = torch.unique(all_edges_sorted, return_counts=True, dim=0)
    
    # 得到所有的拓扑边界
    global_boundary_edges = unique_edges[counts == 1] # [E_b, 2]
    
    if global_boundary_edges.shape[0] == 0:
        return torch.tensor(0.0, device=verts.device)

    # -----------------------------------------------------------
    # 2. 用 Ring Mask 进行过滤 (Double Check)
    # -----------------------------------------------------------
    # 虽然理论上外圈已经被过滤了，但如果 Mesh 在很远的地方还有边缘，
    # 我们需要用 mask 确保只取“这个环”的内圈。
    
    # 创建一个布尔 mask
    is_in_ring = torch.zeros(verts.shape[0], dtype=torch.bool, device=verts.device)
    is_in_ring[ring_mask_indices] = True
    
    # 检查边的两个端点是否都在 ring mask 里
    # 只有端点都在 mask 里的边界边，才是我们想要的“内圈边”
    u = global_boundary_edges[:, 0]
    v = global_boundary_edges[:, 1]
    
    mask_edges = is_in_ring[u] & is_in_ring[v]
    
    # 最终的目标边：内圈边
    target_edges = global_boundary_edges[mask_edges]
    
    if target_edges.shape[0] == 0:
        return torch.tensor(0.0, device=verts.device)

    # -----------------------------------------------------------
    # 3. 计算 1D 平滑 Loss (只针对 target_edges)
    # -----------------------------------------------------------
    # 下面是标准的 graph laplacian 逻辑，但只用 target_edges 构建图
    
    u_target = target_edges[:, 0]
    v_target = target_edges[:, 1]
    
    neighbor_sum = torch.zeros_like(verts)
    degree = torch.zeros((verts.shape[0], 1), device=verts.device)
    
    # 累加坐标
    neighbor_sum.index_add_(0, u_target, verts[v_target])
    degree.index_add_(0, u_target, torch.ones_like(degree[u_target]))
    
    neighbor_sum.index_add_(0, v_target, verts[u_target])
    degree.index_add_(0, v_target, torch.ones_like(degree[v_target]))
    
    # 计算 Loss
    # 只计算 degree > 0 的点（即内圈点）
    active_mask = degree.squeeze() > 0
    
    target_pos = neighbor_sum[active_mask] / (degree[active_mask] + 1e-6)
    current_pos = verts[active_mask]
    
    loss = torch.nn.functional.mse_loss(current_pos, target_pos)
    
    return loss

def mouth(name):
    
    # pt_name = 'delta_geo_show_ffhq'
    from src.networks_bak import myNet
    training_net = myNet()
    training_net.to(device)
    training_net.train()
    training_net.out_layer.weight.register_hook(clip_columns_grad)
    training_net.out_layer.bias.register_hook(clip_columns_grad)
    inputdir = cfg['input_dir']
    parse = read_png(f'{inputdir}/{name}/parsing')
    parse = cv2.imread(f'{inputdir}/{name}/parsing/{cfg["img_path"]}.png')
    parse = parse[None]
    H, W = parse.shape[1:3]
    track_path = f'{inputdir}/{name}/body_track/smplx_track.pth'
    smplx_params = torch.load(track_path)
    cam_para, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, shape, expr, global_orient, transl = to_device(**smplx_params)
    batch_size = body_pose.shape[0]
    lhand_pose = smplx.left_hand_pose.expand(batch_size, -1)
    rhand_pose = smplx.right_hand_pose.expand(batch_size, -1)
    # leye_pose = smplx.leye_pose.expand(batch_size, -1)
    # reye_pose = smplx.reye_pose.expand(batch_size, -1)

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
    import utils3d
    from utils3d.torch import matrix_to_quaternion, quaternion_to_matrix
    s, R, T, k, t = mesh2smplx(name, return_=True, output_dir=OUTPUT_PATH)
    s = s.cpu().numpy()
    # R = R.cpu().numpy()
    T = T.cpu().numpy()
    k = k.cpu().numpy()
    t = t.cpu().numpy()
    vv_new = k * smplx_v[0] + t
    Rota = R[0]
    print(Rota)
    vv_new = s * vv_new @ Rota.transpose(-2, -1).cpu().numpy() + T
    m_new = trimesh.Trimesh(vertices=vv_new, faces=smplx.faces)
    m = trimesh.Trimesh(vertices=vv, faces=smplx.faces)
    sub = np.load('params/mouth.npy')
    sub_1 = np.load('params/smplx_faces_no_eye.npy')
    sub_m = m.submesh([sub])[0]
    sub_m_new = m_new.submesh([sub])[0]
    new_sub_m = m.submesh([sub_1])[0]
    # sub_m.export('src/filter/mouth.obj')

    
    # trellis = trimesh.load(f'exp/{name}/objects/mouth_22.obj')
    # trellis_v = torch.from_numpy(trellis.vertices).to(device)
    # trellis_f = torch.from_numpy(trellis.faces).to(device)
    
    

    # utils3d.io.write_ply('src/filter/mouth.ply', points.detach().cpu().numpy())
    # l = trellis_mask.nonzero().flatten().cpu().numpy()
    
    # trellis.submesh([l])[0].export('src/filter/filtered.obj')

    motion = np.load(f'{OUTPUT_PATH}/{name}/slats/motion_sdf.npz')
    face_id = motion['face_id']
    uvw = motion['uvw']
    index = output.vertices.shape[1] - 1092
    mask = ~(smplx.faces > index).any(axis=-1)
    ff = smplx.faces
    v_last, new_ff = densify(smplx_v[-1], smplx.faces)
    motion_id = new_ff[face_id]

    v_0, _ = densify(smplx_v[0], smplx.faces) 
    deformation = (v_last[None] - v_0[None])
    deformation = deformation[:, motion_id, :]
    trans = np.load(f'{OUTPUT_PATH}/{name}/params/trans.npz')
    kt = np.load(f'{OUTPUT_PATH}/{name}/params/kt.npz')
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
    path = f'{OUTPUT_PATH}/{name}/params/v_pos_i.pt'
    p = torch.load(path)
    new_p = (p).detach().cpu().numpy() + deformation[-1]
    # utils3d.io.write_ply('src/filter/cubes.ply', (p).detach().cpu().numpy() + deformation[15])
    # print(p.shape)
    # exit()
    indices = torch.load(f'{OUTPUT_PATH}/{name}/params/marching.pt')
    my_p = new_p[indices.detach().cpu().numpy()].reshape(-1, 8, 3).mean(axis=1)
    my_p = (my_p - T.cpu().numpy()) @ R[0].cpu().numpy() / s[0].cpu().numpy()
    my_p = (my_p - t.cpu().numpy()) / k.cpu().numpy()
    # utils3d.io.write_ply('src/filter/coords.ply', my_p)

    BVH1 = cubvh.cuBVH(torch.from_numpy(sub_m.vertices).to(device), torch.from_numpy(sub_m.faces).to(device))
    BVH_n = cubvh.cuBVH(torch.from_numpy(sub_m_new.vertices).to(device), torch.from_numpy(sub_m_new.faces).to(device))
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
    coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_0_new.pt').to(device)
    coord = ((coords + 0.5) / 256) - 0.5
    print(coord.shape)
    print(coord[:5])
    dis_c, face_id_c, uvw_c = BVH_n.unsigned_distance(coord[..., 1:], True)
    mmma = dis_c < 0.04
    v_show = coord[..., 1:][mmma].cpu().numpy()
    utils3d.io.write_ply('hello.ply', v_show)
    # print(v_show.shape)
    # print(sub_m_new.vertices.min())
    # print(sub_m_new.vertices.max())
    # exit()
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

    torch.save(mm, f'{OUTPUT_PATH}/{name}/params/marching_mask.pt')

    
    feats_n = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_0_new.pt').to(device)

    feats = torch.load(f'{OUTPUT_PATH}/{name}/params/{pt_name}.pt').to(device)
    model = SparseFeatures2Mesh(res=256)


    fp16_scale_growth = 0.0001
    log_scale = 20
    grad_clip = AdaptiveGradClipper(max_norm=1, clip_percentile=95)
    model_params = [p for p in training_net.parameters() if p.requires_grad]
    master_params = make_master_params(model_params)
    optimizer = torch.optim.AdamW(master_params, lr=5e-4, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

    delta = torch.zeros((feats.shape[0], 53)).float().to(device).requires_grad_(True)
    optimizer_d = torch.optim.Adam([delta], lr=5e-3)
    scheduler_d = StepLR(optimizer_d, step_size=50, gamma=0.9)

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
    
    mesh = model(None, dcoords=coords, dfeats=feats, mask=sdf_mask, n_coords=n_coords, marching_mask=m_mask, change_marching=True, name=name, output_path=OUTPUT_PATH)

    rotas_t = torch.Tensor([
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ],
        [
            [math.cos(math.pi / 4), 0, -math.sin(math.pi / 4), 0],
            [0, 1, 0, 0],
            [math.sin(math.pi / 4), 0, math.cos(math.pi / 4), 0],
            [0, 0, 0, 1]
        ],
        [
            [math.cos(-math.pi / 4), 0, -math.sin(-math.pi / 4), 0],
            [0, 1, 0, 0],
            [math.sin(-math.pi / 4), 0, math.cos(-math.pi / 4), 0],
            [0, 0, 0, 1]
        ],
        [
            [1, 0, 0, 0],
            [0, math.cos(-math.pi / 4), -math.sin(-math.pi / 4), 0],
            [0, math.sin(-math.pi / 4), math.cos(-math.pi / 4), 0],
            [0, 0, 0, 1]
        ]
    ]).to(device)

    rotas = torch.Tensor([
        [
            [math.cos(math.pi / 4), 0, -math.sin(math.pi / 4), 0],
            [0, 1, 0, 0],
            [math.sin(math.pi / 4), 0, math.cos(math.pi / 4), 0],
            [0, 0, 0, 1]
        ],
        [
            [math.cos(-math.pi / 4), 0, -math.sin(-math.pi / 4), 0],
            [0, 1, 0, 0],
            [math.sin(-math.pi / 4), 0, math.cos(-math.pi / 4), 0],
            [0, 0, 0, 1]
        ],
        [
            [1, 0, 0, 0],
            [0, math.cos(-math.pi / 4), -math.sin(-math.pi / 4), 0],
            [0, math.sin(-math.pi / 4), math.cos(-math.pi / 4), 0],
            [0, 0, 0, 1]
        ]
    ]).to(device)

    # return

    # with torch.no_grad():
    #     mesh_gt = model(None, dcoords=coords, dfeats=feats, training=False, mask=sdf_mask)
    #     mesh2smplx(name, mesh_gt)
    #     z_mean = mesh_gt.vertices[..., -1].mean().item()
    #     dicts_gt = render(mesh_gt, extrinsic@rotas_t, proj[0], HW=[H, W])

    
    proj = get_ndc_proj_matrix(cam_para[0:1], [H, W])
    extrinsic = torch.tensor(
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ]
    ).to(device)
    v, f = densify(smplx_v[-1], smplx.faces)
    mouth_detailed = np.load('params/mouth_detailed_.npy')
    head_detailed = np.load('params/head.npy')
    head_de = trimesh.Trimesh(vertices=v, faces=f).submesh([head_detailed])[0]
    mouth_de = trimesh.Trimesh(vertices=v, faces=f).submesh([mouth_detailed])[0]
    # import trimesh
    mean_z = head_de.vertices.mean(axis=0, keepdims=True)[..., -1]
    mean_z = torch.from_numpy(mean_z).to(device)
    # trimesh.Trimesh(vertices=v, faces=f).export('detailed.obj')
    smplx_m = MeshExtractResult(
        vertices=torch.from_numpy(mouth_de.vertices).to(device).float(),
        faces=torch.from_numpy(mouth_de.faces).to(device).int()
    )
    smplx_dicts = render(smplx_m, extrinsic, proj[0], HW=[H, W])
    out = smplx_dicts['normal']

    local_mask = smplx_dicts['mask']

    # masks = []
    structure = generate_binary_structure(2, 1)
    # for x in local_mask:
    #     masks.append(binary_dilation(x.detach().cpu().numpy(), structure=structure, iterations=4))
    local_mask_filled = binary_fill_holes(local_mask.detach().cpu().numpy())
    # cv2.imwrite(f'{OUTPUT_PATH}/{name}/objects/mask_filled.png', local_mask_filled.astype(np.float32) * 255.)
    # exit()
    
    local_mask_filled = torch.from_numpy(local_mask_filled).to(local_mask)

    local_mask_dilated = binary_dilation(local_mask_filled.detach().cpu().numpy(), structure=structure, iterations=4)
    # local_mask_dilated = binary_dilation(local_mask.detach().cpu().numpy(), structure=structure, iterations=1)
    local_mask_dilated = torch.from_numpy(local_mask_dilated).to(local_mask)
    print(out.shape)
    cv2.imwrite('mouth.png', out.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)
    
    model = SparseFeatures2Mesh(res=256)
    coarse_feats = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_coarse_back.pt').to(device)
    coarse_coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_coarse_back.pt').to(device)
    feats = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_0_new.pt').to(device)
    coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_0_new.pt').to(device)
    coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_0_new.pt').to(device)
    feats_n = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_0_new.pt').to(device)
    feats = torch.load(f'{OUTPUT_PATH}/{name}/params/{pt_name}.pt').to(device)
    input = sp.SparseTensor(
        coords=coords,
        feats=feats
    )
    motion = np.load(f'{OUTPUT_PATH}/{name}/slats/motion_sdf.npz')
    face_id = motion['face_id']
    uvw = motion['uvw']

    v_last, new_ff = densify(smplx_v[-1], smplx.faces)
    # motion_id = new_ff[face_id]

    v_0, _ = densify(smplx_v[0], smplx.faces) 
    deformation = (v_last[None] - v_0[None])

    # deformation = (v_last[None] - v_0[None])
    deformation = deformation[:, motion_id, :]
    deformation = np.einsum('bkij, bki->bkj', deformation, np.tile(uvw, (deformation.shape[0], 1, 1)))

    # with torch.no_grad():
    #     temp_d = model(None, dcoords=coords, dfeats=feats, debug=True)
    #     sdf_d = temp_d.reshape((257, 257, 257))
    #     sdf_d = sdf_d.detach().cpu().numpy()
    #     sdf_mask = wzj_final(
    #         sdf_d,
    #     )
    #     # exit()
    #     sdf_mask = sdf_mask.reshape(-1)
    #     sdf_mask = torch.from_numpy(sdf_mask).to(device)

    # with torch.no_grad():
    #     temp_d = model(None, dcoords=coords, dfeats=feats, debug=True)

    #     temp_d[sdf_mask] *= -1
    #     sdf_d = temp_d.reshape((257, 257, 257))
    #     sdf_d = sdf_d.detach().cpu().numpy()
    #     sdf_mask1 = wzj_final(
    #         sdf_d,
    #     )
    #     sdf_mask1 = sdf_mask1.reshape(-1)
    #     sdf_mask1 = torch.from_numpy(sdf_mask1).to(device)
    #     print(sdf_mask1.sum())
    # sdf_mask = sdf_mask | sdf_mask1


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

    n_coords = torch.load(f'{OUTPUT_PATH}/{name}/params/v_pos_.pt').to(coords)
    m_mask = torch.load(f'{OUTPUT_PATH}/{name}/params/marching_mask.pt').to(device)
    mesh = model(None, dcoords=coords, dfeats=feats, v_a=torch.from_numpy(deformation[-1]).to(dtype=torch.float32, device=device), mask=sdf_mask, n_coords=n_coords, marching_mask=m_mask, change_marching=True, name=name, output_path=OUTPUT_PATH)
    mesh.vertices = (mesh.vertices - T) @ R[0] / s[0]
    mesh.vertices = (mesh.vertices - t) / k
    mesh.face_normal = torch.matmul(mesh.face_normal, R)
    mesh_out = render(mesh, extrinsic, proj[0], HW=[H, W])
    out = mesh_out['normal']
    print(out.shape)
    cv2.imwrite('mesh.png', out.permute(1, 2, 0).detach().cpu().numpy() * 255.)

    input = sp.SparseTensor(
        coords=coarse_coords.to(dtype=torch.int32),
        feats=coarse_feats
    )
    iters = 400

    

    # iters = 501
    # delta = torch.zeros((feats.shape[0], 8)).to(device).requires_grad_(True)
    # optimizer = Adam([delta], lr=5e-5)

    os.makedirs(f'{OUTPUT_PATH}/{name}/mouth', exist_ok=True)
    indices = torch.load(f'{OUTPUT_PATH}/{name}/params/indices.pt')
    BCE = torch.nn.BCELoss()
    # grad_clip = AdaptiveGradClipper(max_norm=0.01, clip_percentile=95)

    with torch.no_grad():
        mesh_gt = model(None, dcoords=coords, dfeats=feats, v_a=torch.from_numpy(deformation[-1]).to(dtype=torch.float32, device=device), mask=sdf_mask, n_coords=n_coords, marching_mask=m_mask, indices=indices)
        mesh2smplx(name, mesh_gt, output_dir=OUTPUT_PATH)
        z_mean = mesh_gt.vertices[..., -1].mean().item()
        dicts_gt = render(mesh, extrinsic, proj[0], HW=[H, W])

    smplx_mesh = trimesh.Trimesh(vertices=v, faces=f)
    my_mesh = trimesh.Trimesh(vertices=mesh_gt.vertices.detach().cpu().numpy(), faces=mesh.faces.detach().cpu().numpy())
    smplx_mesh.export(f'{OUTPUT_PATH}/{name}/objects/smplx_mouth_use.obj')
    my_mesh.export(f'{OUTPUT_PATH}/{name}/objects/my_mouth_use.obj')
    BVH_op = cubvh.cuBVH(smplx_m.vertices, smplx_m.faces.to(torch.int32))
    v_sm = smplx_m.vertices.clone()
    v_sm[..., 2] = 0
    BVH_mm = cubvh.cuBVH(v_sm, smplx_m.faces.to(torch.int32))
    # distance, face_id, uvw = BVH_op.unsigned_distance(mesh_gt.vertices, return_uvw=True)
    # x_gt = mesh_gt.vertices.clone()
    # x_gt[..., 2] = 0
    # distance1, _, _ = BVH_mm.unsigned_distance(x_gt, return_uvw=True)
    # msk = distance < 0.02
    # msk1 = distance1 < 0.0025
    # msk = (msk & msk1)

    # ff = face_id[msk]
    # uv = uvw[msk]
    # v_uv = smplx_m.vertices[smplx_m.faces[ff]]
    # v_target = torch.einsum('bij, bi->bj', v_uv, uv)

    # out_p = mesh_gt.vertices[msk].detach().cpu().numpy()
    # utils3d.io.write_ply(f'{OUTPUT_PATH}/{name}/objects/my_mouth.ply', out_p)

    # delta_mesh = torch.zeros((out_p.shape[0], 3)).to(device).requires_grad_(True)
    # optimizer_m = torch.optim.Adam([delta_mesh], lr=5e-4)
    # scheduler_m = StepLR(optimizer_m, step_size=100, gamma=0.9)
    # print(v_target.shape, msk.shape)
    from pytorch3d.loss import chamfer_distance
    # for iter in range(200):
    #     optimizer_m.zero_grad()
    #     # print(mesh_gt.vertices[msk].shape)
    #     # print(delta.shape)
    #     # print(v_target.shape)
    #     loss = chamfer_distance((mesh_gt.vertices[msk] + delta_mesh)[None], v_target[None])[0]
    #     print(loss)
    #     print(loss)
    #     loss.backward()
    #     optimizer_m.step()
    
    # mesh_gt.vertices[msk] += delta_mesh
    # trimesh.Trimesh(vertices=mesh_gt.vertices.detach().cpu().numpy(), faces=mesh_gt.faces.detach().cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/objects/optim_mouse.obj')


    # exit()
    
    
    for epoch in range(iters + 1):
        # optimizer.zero_grad()
        optimizer_d.zero_grad()
        # zero_grad(model_params)
        # delta = training_net(input)
        ff = torch.cat(
            [
                feats[..., :53].detach() + delta * mmma[..., None],
                feats[..., 53:].detach()
            ],
            dim=-1
        ).to(device)
        # ff = torch.zeros(*feats.shape).to(device).requires_grad_(True)
        # ff[mmma][..., :53] = feats[mmma][..., :53].detach() + delta
        # ff[~mmma] = feats[~mmma].detach()
        mesh = model(None, dcoords=coords, dfeats=ff, v_a=torch.from_numpy(deformation[-1]).to(dtype=torch.float32, device=device), mask=sdf_mask, n_coords=n_coords, marching_mask=m_mask, indices=indices)

        mesh2smplx(name, mesh, output_dir=OUTPUT_PATH)
        new_v, new_f = filter_z(mesh, mean_z)
        mesh = MeshExtractResult(vertices=new_v, faces=new_f)
        dicts = render(mesh, extrinsic, proj[0], HW=[H, W])


        distance, face_id, uvw = BVH_op.unsigned_distance(mesh.vertices.detach(), return_uvw=True)
        x_gt = mesh.vertices.detach().clone()
        x_gt[..., 2] = 0
        distance1, _, _ = BVH_mm.unsigned_distance(x_gt, return_uvw=True)
        msk = distance < 0.02
        msk1 = distance1 < 0.0025
        msk = (msk & msk1)

        face_verts_mask = msk[mesh.faces]
        face_mask = face_verts_mask.all(dim=1)
        # perceptual_loss_new = 10 * loss_recon(dicts['normal'][0] * local_mask_dilated[0, None], smplx_dicts_local['normal'][0] * local_mask_dilated[0, None]) + 5 * loss_recon(dicts['normal'][0] * (1 - local_mask[0, None]), (1 - local_mask[0, None]) * dicts_gt_['normal'][0])
        perceptual_loss = loss_recon(dicts['normal'] * local_mask_dilated[None], smplx_dicts['normal'] * local_mask_dilated[None], lambda_ssim=8)
        pred = torch.nn.functional.normalize(dicts['normal'], dim=0, eps=1e-6)
        gt = torch.nn.functional.normalize(smplx_dicts['normal'], dim=0, eps=1e-6)
        dot = (pred * gt).sum(dim=0)
        loss_normal = ((1 - dot.clamp(-1, 1)) * local_mask_dilated).sum() / (local_mask_dilated.sum() + 1e-6)
        perceptual_loss_local = 20 * loss_recon((1 - local_mask_filled)[None] * dicts['normal'], (1 - local_mask_filled)[None] * dicts_gt['normal'], lambda_ssim=3, lambda_norm=0)

        depth_loss = cal_l1_loss((dicts['depth'][None] - smplx_dicts['depth'][None]), mask=local_mask_dilated[None])

        # mask_loss = loss_recon(dicts['mask'] * local_mask, smplx_dicts['mask'] * local_mask)
        # cv2.imwrite(f'{OUTPUT_PATH}/{name}/objects/mouth_mask.png', local_mask_dilated.detach().cpu().numpy() * 255.)
        # exit()
        # mask_loss = 5 * BCE(dicts['mask'] * local_mask_filled, smplx_dicts['mask'] * local_mask_filled) + 0.5 * dice_loss(dicts['mask'] * local_mask_filled, smplx_dicts['mask'] * local_mask_filled)
        # loss = perceptual_loss + 0.5 * depth_loss + 0.5 * mask_loss + perceptual_loss_local
        # reg_loss = boundary_smooth_loss(mesh.vertices[msk])

        # cf_loss, _ = chamfer_distance(mesh.vertices[msk][None], smplx_m.vertices[None])
        # loss =  perceptual_loss + 1 * perceptual_loss_local + 5 * loss_normal + 2e3 * cf_loss + 1e3 * reg_loss
        mesh_ = Meshes(verts=mesh.vertices[None], faces=mesh.faces[face_mask][None])

        # 2. 计算 Loss
        # # method="uniform": 简单的邻居平均 (适合一般去噪)
        lap_loss = mesh_laplacian_smoothing(mesh_, method="uniform")
        lap_loss = mesh_normal_consistency(mesh_)
        loss_bound_smooth = inner_hole_smooth_loss(mesh.vertices, mesh.faces, msk)
        if epoch < 200:
            lambda_bound = 0
            lambda_bound_smooth = 1.0
            lambda_perp = 4
            lambda_lap = 0.
        else:
            lambda_bound = 0
            lambda_perp = 1.5
            lambda_bound_smooth = 1e6
            lambda_lap = 1e3

        bound_loss = boundary_length_loss(mesh.vertices, mesh.faces[face_mask])
        loss =  lambda_perp * perceptual_loss + 10 * perceptual_loss_local + 5 * loss_normal + lambda_bound * bound_loss + lambda_bound_smooth * loss_bound_smooth + 0.1 * lap_loss


        loss.backward()
        # print(delta.grad)
        grad_norm = grad_clip(delta)
        scheduler_d.step()

        optimizer_d.step()

        # scaled_loss = loss * (2 ** log_scale)
        # scaled_loss.backward()

        # # print(model_params)
        # # exit()

        # model_grads_to_master_grads(model_params, master_params)
        # master_params[0].grad.mul_(1.0 / (2 ** log_scale))
        # grad_norm = grad_clip(master_params)

        # if not any(not p.grad.isfinite().all() for p in model_params):
        #     optimizer.step()
        #     # scheduler.step()
        #     master_params_to_model_params(model_params, master_params)
        #     log_scale += fp16_scale_growth
        # else:
        #     log_scale -= 1
        # loss.backward()
        # grad_norm = grad_clip(delta)
        # optimizer.step()
        print(f'epoch: {epoch}, loss: {loss}, perceptual_loss: {perceptual_loss}, depth_loss: {depth_loss}, normal: {loss_normal}, bound_loss: {bound_loss}, loss_bound_smooth: {loss_bound_smooth}, lap_loss: {lap_loss}')

        if epoch == iters:
            # mesh.vertices[..., -1:] += z_mean
            # trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/objects/single_n.obj')
            with torch.no_grad():
                mesh = model(None, dcoords=coords, dfeats=ff, mask=sdf_mask, marching_mask=m_mask, indices=indices)
                trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.detach().cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/params/final_mesh.obj')
            torch.save(ff, f'{OUTPUT_PATH}/{name}/params/delta_geo_mouth_new.pt')
            # cv2.imwrite(f'exp/{mname}/objects/dd.png', I_dep.detach().cpu().numpy()[..., ::-1] * 255.)
            # cv2.imwrite(f'exp/{mname}/objects/nn.png', I_nor.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)
            # cv2.imwrite(f'exp/{mname}/objects/dd1.png', I_dep1.detach().cpu().numpy()[..., ::-1] * 255.)
            # cv2.imwrite(f'exp/{mname}/objects/nn1.png', I_nor1.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)


            # cv2.imwrite(f'exp/{mname}/objects/local_mask.png', local_mask_dilated.permute(1, 0, 2).flatten(start_dim=1).detach().cpu().numpy() * 255.)
            # cv2.imwrite(f'exp/{mname}/objects/local_normal.png', ((1 - local_mask_dilated[:, None]) * dicts_gt_['normal']).permute(1, 2, 0, 3).flatten(start_dim=2).permute(1, 2, 0).detach().cpu().numpy() * 255.)
        if epoch % 5 == 0:
        # if epoch == 100:
        # if epoch == 200:
            # n = dicts['normal']
            cv2.imwrite(f'{OUTPUT_PATH}/{name}/mouth/normal_{epoch}.png', dicts['normal'].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)





def process(name):
    

    from src.networks_bak import myNet
    training_net = myNet()
    training_net.to(device)
    training_net.train()
    training_net.out_layer.weight.register_hook(clip_columns_grad)
    training_net.out_layer.bias.register_hook(clip_columns_grad)
    inputdir = cfg['input_dir']
    parse = read_png(f'{inputdir}/{name}/parsing')
    parse = cv2.imread(f'{inputdir}/{name}/parsing/{cfg["img_path"]}.png')
    parse = parse[None]
    H, W = parse.shape[1:3]
    track_path = f'{inputdir}/{name}/body_track/smplx_track.pth'
    smplx_params = torch.load(track_path)
    cam_para, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, shape, expr, global_orient, transl = to_device(**smplx_params)
    batch_size = body_pose.shape[0]
    lhand_pose = smplx.left_hand_pose.expand(batch_size, -1)
    rhand_pose = smplx.right_hand_pose.expand(batch_size, -1)
    # leye_pose = smplx.leye_pose.expand(batch_size, -1)
    # reye_pose = smplx.reye_pose.expand(batch_size, -1)

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

    motion = np.load(f'{OUTPUT_PATH}/{name}/slats/motion_sdf.npz')
    face_id = motion['face_id']
    uvw = motion['uvw']
    index = output.vertices.shape[1] - 1092
    mask = ~(smplx.faces > index).any(axis=-1)
    ff = smplx.faces


    v_last, new_ff = densify(smplx_v[-1], smplx.faces)
    motion_id = new_ff[face_id]
    v_0, _ = densify(smplx_v[0], smplx.faces) 
    deformation = (v_last[None] - v_0[None])

    # motion_id = ff[face_id]
    # deformation = (smplx_v[-1:] - smplx_v[0:1])
    deformation = deformation[:, motion_id, :]
    trans = np.load(f'{OUTPUT_PATH}/{name}/params/trans.npz')
    kt = np.load(f'{OUTPUT_PATH}/{name}/params/kt.npz')
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
    path = f'{OUTPUT_PATH}/{name}/params/v_pos_i.pt'
    p = torch.load(path)
    new_p = (p).detach().cpu().numpy() + deformation[-1]
    # utils3d.io.write_ply('src/filter/cubes.ply', (p).detach().cpu().numpy() + deformation[15])
    # print(p.shape)
    # exit()
    indices = torch.load(f'{OUTPUT_PATH}/{name}/params/marching.pt')
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

    torch.save(mm, f'{OUTPUT_PATH}/{name}/params/marching_mask.pt')

    coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_0_new.pt').to(device)
    feats_n = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_0_new.pt').to(device)
    feats = torch.load(f'{OUTPUT_PATH}/{name}/params/delta_geo_mouth_new.pt').to(device)
    model = SparseFeatures2Mesh(res=256)


    fp16_scale_growth = 0.0001
    log_scale = 20
    grad_clip = AdaptiveGradClipper(max_norm=0.05, clip_percentile=95)
    model_params = [p for p in training_net.parameters() if p.requires_grad]
    master_params = make_master_params(model_params)
    optimizer = torch.optim.AdamW(master_params, lr=5e-4, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

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
    # m_mask = torch.load(f'{OUTPUT_PATH}/{name}/params/marching_mask.pt').to(device)
    
    mesh = model(None, dcoords=coords, dfeats=feats, mask=sdf_mask, n_coords=n_coords, marching_mask=mm, change_marching=True, name=name, output_path=OUTPUT_PATH)

class OnlyInfo(logging.Filter):
    def filter(self, record):
        return record.levelno < logging.WARNING

import subprocess
from datetime import datetime

name_list = ['287', '290', '294', '285', '283', '282', '274', '262', '259', '253', '249', '248', '247', '240', '239', '238', '232', '227', '223', '220', '216', '212', '200', '199', '188', '179', '165', '149', '140', '139', '128', '115', '112', '108', '106', '104', '098', '083', '076', '075', '074', '071', '060', '055', '040', '036', '031', '030', '290', '294', '301', '306', '307', '313', '314', '315', '318', '319', '320', '326', '331', '371']

# print('wzj')

FaceVerse = ['007_21',
              '008_16', 
            #   '009_11', 
              '012_13', 
              '014_13', 
              '016_06'
              ]

ffhq = ['00000-00320.png', '00000-00502.png', '00000-00454.png', '00000-00447.png', '00000-00437.png', '00000-00348.png', '00000-00320.png', '00000-00247.png', '00000-00188.png', '00000-00145.png', '00000-00114.png', '00000-00040.png', '00000-00012.png']

ffhq = ['00000-00320.png', '00000-00502.png', '00000-00454.png' , '00000-00437.png', '00000-00247.png', '00000-00114.png', '00000-00012.png', '00000-00145.png']

pt_name = 'delta_geo_show_ffhq'
import subprocess

if __name__ == '__main__':
    # main_daviad('007_21')
    # exit()
    # main('nersemble_vids_315.mp4')
    # exit()
    name = 'pipe'
    args = ['ln', '-sfn', f'../{OUTPUT_PATH}/{name}', 'src/debug']
    subprocess.run(args)

    # main(name)
    smplx2mesh(name)
    bind_no_eye(name)
    mouth(name)
    bind_no_eye(name, mouth=True)
    process(name)
    exit()
    name = 'nersemble_vids_326.mp4'
    
    # for name in ffhq:
    #     main('00000-00502.png')
    exit()
    # name = '012_13'
    # main_daviad('012_13')
    # exit()
    # for name in FaceVerse:
    #     main_daviad(name)
    # main_daviad(name)
    # exit()
    # name = 'nersemble_vids_314.mp4'
    # smplx2mesh(name)
    # bind_no_eye(name)
    # mouth(name)
    # # main(name)
    # exit()
    # # l = main(name)
    # # exit()
    # # logger.info("loss_dict=%s", l)
    # smplx2mesh(name)
    # bind_no_eye(name)
    # mouth(name)
    # bind_no_eye(name, mouth=True)
    # process(name)
    # exit()
    # # main(name)
    # # exit()
    # smplx2mesh(name)
    # bind_no_eye(name)
    # mouth(name)
    # bind_no_eye(name)
    # process(name)
    # exit()
    # mouth(name)
    # smplx2mesh(name)
    # bind_no_eye(name)
    # process(name)
    # exit()
    # main('pipe')
    # exit()
    # subprocess.run(['rm', '-f', 'info_geo.log'])
    # main('008_07')
    # name = '012_11'
    # main_daviad(name)
    # exit()
    # exit()
    # # exit()
    # # main(name)
    # smplx2mesh(name)
    # exit()
    # bind_no_eye(name)
    # mouth(name)
    # exit()
    # exit()
    time = datetime.now()
    s = time.strftime("%Y%m%d_%H%M%S")
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    info_handler = logging.FileHandler(f'logs/geo/info_geo_{s}.log', 'a', 'utf-8')
    info_handler.setFormatter(formatter)
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(OnlyInfo())

    error_handler = logging.FileHandler(f'logs/geo/error_geo_{s}.log', 'a', 'utf-8')
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)

    logger.addHandler(info_handler)
    logger.addHandler(error_handler)

    # paths = os.listdir('input_ffhq')
    paths = ffhq
    # paths = ['nersemble_vids_' + name + '.mp4' for name in name_list]

    for path in paths:
        if os.path.exists(f'outputs_ffhq/{path}/objects/single_new___.obj'):
            logger.info(f'{path} has been processed')
            # exit()
            continue
        if os.path.exists(f'outputs_ffhq/{path}/objects/new_outer_filtered.obj'):
            # print(path)
            try:
                print(f'processing {path}')
                l = main(path)
                logger.info("loss_dict=%s", l)
                # smplx2mesh(path)
                # bind_no_eye(path)
                # mouth(path)
                # bind_no_eye(path, mouth=True)
                # process(path)
                logger.info(f'{path} has been processed')
            except Exception as e:
                logger.error(
                    f'Path: {path} 发生错误',
                    exc_info=True
                )
    exit()
    name = '002_02'
    # main(name)
    # exit()
    # main(name)
    smplx2mesh(name)
    bind_no_eye(name)
    mouth(name)
