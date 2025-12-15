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
from trellis.modules import sparse as sp
from trellis.representations.mesh import SparseFeatures2Mesh
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR, CosineAnnealingWarmRestarts
import logging

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
    max_norm = 5e-6
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
    os.makedirs(f'{OUTPUT_PATH}/{name}/geo', exist_ok=True)
    training_net = myNet()
    training_net.to(device)
    # training_net.train()
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
    grad_clip = AdaptiveGradClipper(max_norm=0.01, clip_percentile=95)
    model_params = [p for p in training_net.parameters() if p.requires_grad]
    master_params = make_master_params(model_params)
    optimizer = torch.optim.AdamW(master_params, lr=5e-4, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

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
    print(dicts_gt['depth'].shape)
    print(dicts_gt['depth'].min())
    print(dicts_gt['depth'].max())
    exit()

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
    iters = 400
    for epoch in range(iters + 1):
        zero_grad(model_params)
        delta = training_net(input)

        # optimizer_sing.zero_grad()
        ff = torch.cat(
            [
                feats[..., :32].detach() + delta.feats,
                feats[..., 32:].detach()
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
        # print(dicts_gt['depth'].max())
        # print(dicts['depth'].max())
        # print(dicts['depth'].shape)
        # print(dicts['normal'].shape)
        # print(smplx_dicts['normal'].shape)
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
        # print(mesh.vertices.shape)
        # exit()
        meshes = Meshes(verts=[mesh.vertices], faces=[mesh.faces])
        loss_laplacian = mesh_laplacian_smoothing(meshes, method="uniform")

        depth_loss = 10 * l1_loss(dicts['depth'][0], face_mask[0, 0] * smplx_dicts['depth'] + (1 - face_mask[0, 0]) * dicts_gt['depth'][0])
        perceptual_loss = loss_recon(dicts['normal'][0], smplx_dicts['normal'] * face_mask[0] + (1 - face_mask[0]) * dicts_gt['normal'][0])
        mmask = smplx_dicts_local['mask'] != 0
        # depth_loss_local = 10 * l1_loss(dicts['depth'][1:], smplx_dicts_local['depth'] * local_mask + (1 - local_mask) * dicts_gt_['depth'][1:])
        # perceptual_loss_local = l1_loss(dicts['normal'][1:], smplx_dicts_local['normal'] * local_mask[:, None] + (1 - local_mask[:, None]) * dicts_gt_['normal'][1:])
        perceptual_loss_local = loss_recon(dicts['normal'][1:] * local_mask_dilated[1:, None], smplx_dicts_local['normal'][1:] * local_mask_dilated[1:, None]) + 0.8 * loss_recon((1 - local_mask[1:, None]) * dicts['normal'][1:], (1 - local_mask[1:, None]) * dicts_gt_['normal'][1:])
        # loss_reg = mesh.reg_loss

        # img_n1 = smplx_dicts_local['normal'].permute(1, 2, 0, 3).flatten(start_dim=-2).permute(1, 2, 0)
        # img_n2 = dicts['normal'][1:].permute(1, 2, 0, 3).flatten(start_dim=-2).permute(1, 2, 0)
        # img = torch.cat([img_n1, img_n2], dim=1)
        # cv2.imwrite('src/snnwzj.png', img.detach().cpu().numpy() * 255.)
        # loss = depth_loss + 2 * perceptual_loss + 0.1 * loss_reg + 5 * depth_loss_local + 10 * perceptual_loss_local
        depth_loss_new = 10 * loss_recon(dicts['depth'][0, None] * local_mask_dilated[0, None], smplx_dicts_local['depth'][0, None] * local_mask_dilated[0, None]) + 5 * loss_recon((1 - local_mask)[0, None] * dicts['depth'][0, None], (1 - local_mask)[0, None] * dicts_gt_['depth'][0, None])

        # perceptual_loss_new = loss_recon(dicts['normal'][0], smplx_dicts_local['normal'][0] * local_mask_dilated[0, None] + (1 - local_mask[0, None]) * dicts_gt_['normal'][0]) ### 2025 11.9 22:33
        perceptual_loss_new = 10 * loss_recon(dicts['normal'][0] * local_mask_dilated[0, None], smplx_dicts_local['normal'][0] * local_mask_dilated[0, None]) + 5 * loss_recon(dicts['normal'][0] * (1 - local_mask[0, None]), (1 - local_mask[0, None]) * dicts_gt_['normal'][0])

        # loss = 10 * depth_loss + 2 * perceptual_loss + 2 * perceptual_loss_local + 1.5 * loss_laplacian
        loss = 1 * perceptual_loss_new + 0.5 * perceptual_loss_local + 0.2 * depth_loss_new
        # loss = depth_loss + 2 * perceptual_loss  + depth_loss_eye + 2 * perceptual_loss_eye
        # loss = depth_loss_eye + 2 * perceptual_loss_eye
        # loss = depth_loss + 100 * depth_loss_eye
        # loss = depth_loss_local + 2 * perceptual_loss_local
        
        # loss.backward()
        # optimizer_sing.step()
        # scheduler.step()
        # grad_norm = grad_clip([delta])
        if epoch % 5 == 0:
            print(f'epoch: {epoch}, loss: {loss}, depth_loss: {depth_loss}')

        #TODO 通过mask选出smplx对应的面，然后对于侧面视角和仰视视角，渲染pred + smplx筛出的面部作为gt，因为此时面部高度重合，只是为了把接缝处处理一下
        #TODO 如果一开始面部过于偏怎么办，去掉这些例子算了


        scaled_loss = loss * (2 ** log_scale)
        scaled_loss.backward()

        # print(model_params)
        # exit()

        model_grads_to_master_grads(model_params, master_params)
        master_params[0].grad.mul_(1.0 / (2 ** log_scale))
        grad_norm = grad_clip(master_params)

        if not any(not p.grad.isfinite().all() for p in model_params):
            optimizer.step()
            # scheduler.step()
            master_params_to_model_params(model_params, master_params)
            log_scale += fp16_scale_growth
        else:
            log_scale -= 1

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
            
        # if epoch % 5 == 0:
        #     # n = dicts['normal']
        #     cv2.imwrite(f'outputs/{name}/geo/normal_{epoch}.png', dicts['normal'][0].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)

        if epoch == 400:
            # n = dicts['normal']
            cv2.imwrite(f'outputs/{name}/geo/normal_{epoch}.png', dicts['normal'][0].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)

def cal_l1_loss(x, mask = None):
    if mask is None:
        return torch.abs(x).mean()
    else:
        return torch.abs(x*mask).mean() / (mask.mean() + 1e-7)

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
        # delta_value = 0.1
        # huber_loss = torch.nn.functional.huber_loss(
        #     input=depth_pred * valid_mask, 
        #     target=depth_gt_aligned * valid_mask, 
        #     reduction='mean', 
        #     delta=delta_value
        # )
        loss_depth = cal_l1_loss(depth_pred - depth_gt_aligned, valid_mask)
        
        # return loss_depth + 0.3 * (1 - ssim((depth_pred * valid_mask)[None], (depth_gt_aligned * valid_mask)[None]))
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
    os.makedirs(f'{OUTPUT_PATH}/{name}/geo', exist_ok=True)
    training_net = myNet()
    training_net.to(device)
    # training_net.train()
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
    grad_clip = AdaptiveGradClipper(max_norm=0.01, clip_percentile=95)
    model_params = [p for p in training_net.parameters() if p.requires_grad]
    master_params = make_master_params(model_params)
    optimizer = torch.optim.AdamW(master_params, lr=5e-4, weight_decay=0.0)
    # scheduler = LambdaLR(optimizer, lr_lambda=custom_lr_lambda)
    scheduler = StepLR(optimizer, step_size=120, gamma=0.9)
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
    iters = 1000
    for epoch in range(iters + 1):
        zero_grad(model_params)
        delta = training_net(input)

        # optimizer_sing.zero_grad()
        ff = torch.cat(
            [
                feats[..., :32].detach() + delta.feats,
                feats[..., 32:].detach()
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
        # print(dicts['depth'].min())
        # print(dicts['depth'].max())
        # exit()
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
        # print(dicts_gt['depth'].max())
        # print(dicts['depth'].max())
        # print(dicts['depth'].shape)
        # print(dicts['normal'].shape)
        # print(smplx_dicts['normal'].shape)
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
        # print(mesh.vertices.shape)
        # exit()
        # meshes = Meshes(verts=[mesh.vertices], faces=[mesh.faces])
        # loss_laplacian = mesh_laplacian_smoothing(meshes, method="uniform")

        depth_loss = cal_depth_loss(dicts['depth'][0], depth_gt.clone(), dicts['mask'][0] * seg_mask.clone(), parsing_map.clone(), face_ldmks.clone())
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
        my_mask = ((dicts['mask'][0] * seg_mask) > 0.5).float()
        # normal_loss = cal_l1_loss(dicts['normal'][0] - normal_gt.permute(2, 0, 1), ((dicts['mask'][0] * seg_mask) > 0.5).float()) + 0.2 * (1 - ssim(dicts['normal'][0] * my_mask, normal_gt.permute(2, 0, 1) * my_mask))
        # loss = normal_loss
        # if epoch < 100:
        #     lambda_depth = 10
        # else:
        lambda_depth = 3
        normal_loss = loss_recon(dicts['normal'][0] * my_mask[None], normal_gt.permute(2, 0, 1) * my_mask[None])
        loss = normal_loss + lambda_depth * depth_loss
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
            print(f'epoch: {epoch}, loss: {loss}, normal_loss: {normal_loss}, depth_loss: {depth_loss}, lr: {scheduler.get_lr()}')

        #TODO 通过mask选出smplx对应的面，然后对于侧面视角和仰视视角，渲染pred + smplx筛出的面部作为gt，因为此时面部高度重合，只是为了把接缝处处理一下
        #TODO 如果一开始面部过于偏怎么办，去掉这些例子算了


        scaled_loss = loss * (2 ** log_scale)
        scaled_loss.backward()

        # print(model_params)
        # exit()

        model_grads_to_master_grads(model_params, master_params)
        master_params[0].grad.mul_(1.0 / (2 ** log_scale))
        grad_norm = grad_clip(master_params)

        if not any(not p.grad.isfinite().all() for p in model_params):
            optimizer.step()
            scheduler.step()
            master_params_to_model_params(model_params, master_params)
            log_scale += fp16_scale_growth
        else:
            log_scale -= 1

        # if epoch % 100 == 0:
        #     mesh.vertices[..., -1:] += z_mean
        #     trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.cpu().numpy()).export(f'outputs/{name}/objects/single_{epoch}.obj')

        if epoch == iters:
            mesh.vertices[..., -1:] += z_mean
            trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.cpu().numpy()).export(f'outputs/{name}/objects/single_david.obj')
            torch.save(ff, f'outputs/{name}/params/delta_geo_david.pt')
            # cv2.imwrite(f'outputs/{name}/objects/dd.png', I_dep.detach().cpu().numpy()[..., ::-1] * 255.)
            # cv2.imwrite(f'outputs/{name}/objects/nn.png', I_nor.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)
            # cv2.imwrite(f'outputs/{name}/objects/dd1.png', I_dep1.detach().cpu().numpy()[..., ::-1] * 255.)
            # cv2.imwrite(f'outputs/{name}/objects/nn1.png', I_nor1.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)


            # cv2.imwrite(f'outputs/{name}/objects/local_mask.png', local_mask.permute(1, 0, 2).flatten(start_dim=1).detach().cpu().numpy() * 255.)
            # cv2.imwrite(f'outputs/{name}/objects/local_normal.png', ((1 - local_mask[1:, None]) * dicts_gt_['normal'][1:]).permute(1, 2, 0, 3).flatten(start_dim=2).permute(1, 2, 0).detach().cpu().numpy() * 255.)
            
        # if epoch % 5 == 0:
        #     # n = dicts['normal']
        #     cv2.imwrite(f'outputs/{name}/geo/normal_{epoch}.png', dicts['normal'][0].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)

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
                feats[..., :8].detach() + delta,
                feats[..., 8:].detach()
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
    
    track_path = f'inputs/{name}/body_track/smplx_track.pth'
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
    p = output.vertices[0].detach().cpu().numpy()
    f = smplx.faces
    trimesh.Trimesh(vertices=p, faces=f).export(f'outputs/{name}/objects/origin_smplx.obj')
    f = smplx.faces[mask]
    p = output.vertices[0, :-1092].detach().cpu().numpy()
    trimesh.Trimesh(vertices=p, faces=f).export(f'outputs/{name}/objects/origin_smplx_wo_eyes.obj')
    trans = np.load(f'outputs/{name}/params/trans.npz')
    kt = np.load(f'outputs/{name}/params/kt.npz')
    k, t = list(kt.values())
    s, R, T = list(trans.values())
    p = k * p + t
    from utils3d.torch import quaternion_to_matrix
    Rota = quaternion_to_matrix(torch.from_numpy(R)).squeeze(0)

    p = s * p @ Rota.transpose(-2, -1).cpu().numpy() + T
    aligned_smplx = trimesh.Trimesh(vertices=p, faces=f)
    aligned_smplx.export(f'outputs/{name}/objects/sm.obj')
    aligned_smplx = trimesh.load(f'outputs/{name}/objects/sm.obj')
    test = (f - aligned_smplx.faces).sum()
    print((p - aligned_smplx.vertices).sum())
    print(test)


def bind_no_eye(name, mesh=None):
    from trellis.representations.mesh import SparseFeatures2Mesh
    model = SparseFeatures2Mesh(res=256, use_color=True)
    coords = torch.load(f'outputs/{name}/slats/coords_0_new.pt').to(device)
    feats = torch.load(f'outputs/{name}/slats/feats_0_new.pt').to(device)
    feats_n = torch.load(f'outputs/{name}/params/delta_geo.pt').to(device)

    # smplx = trimesh.load(f'../../output/{name}/objects/smplx.obj')
    if mesh is None:
        smplx = trimesh.load(f'outputs/{name}/objects/sm.obj')
    else:
        smplx = mesh
    input = sp.SparseTensor(
        coords=coords,
        feats=feats_n
    )
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
    output = model(input, return_v_a=True, mask=sdf_mask, name=name)
    # trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy(), faces=output.faces.detach().cpu().numpy()).export('src/tutils/sm_.obj')
    if hasattr(output, 'v_p'):
        print(output.v_p.shape)
    v_p = output.v_p.to(device)
    # v_p = (v_p / 256) - 0.5
    BVH = cubvh.cuBVH(torch.from_numpy(smplx.vertices).to(device), torch.from_numpy(smplx.faces).to(device=device, dtype=torch.int32))
    distance, face_id, uvw = BVH.unsigned_distance(v_p, return_uvw=True)
    # np.savez(f'../../output/{name}/slats/motion_sdf.npz', face_id=face_id.detach().cpu().numpy(), uvw=uvw.detach().cpu().numpy())
    np.savez(f'outputs/{name}/slats/motion_sdf.npz', face_id=face_id.detach().cpu().numpy(), uvw=uvw.detach().cpu().numpy())


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



def mouth(name):
    
    parse = read_png(f'inputs/{name}/parsing')
    parse = cv2.imread(f'inputs/{name}/parsing/000001.png')
    parse = parse[None]
    H, W = parse.shape[1:3]
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

    coords = torch.load(f'outputs/{name}/slats/coords_0_new.pt').to(device)
    feats = torch.load(f'outputs/{name}/params/delta_geo.pt').to(device)
    model = SparseFeatures2Mesh(res=256)

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
    n_coords = torch.load(f'outputs/{name}/params/v_pos_.pt').to(coords)
    m_mask = torch.load(f'outputs/{name}/params/marching_mask.pt').to(device)
    
    mesh = model(None, dcoords=coords, dfeats=feats, mask=sdf_mask, n_coords=n_coords, marching_mask=m_mask, change_marching=True, name=name)

    return
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
    mouth_detailed = np.load('params/mouth_detailed.npy')
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
    print(out.shape)
    cv2.imwrite('mouth.png', out.permute(1, 2, 0).detach().cpu().numpy() * 255.)
    
    model = SparseFeatures2Mesh(res=256)
    coords = torch.load(f'outputs/{name}/slats/coords_0_new.pt').to(device)
    feats_n = torch.load(f'outputs/{name}/slats/feats_0_new.pt').to(device)
    feats = torch.load(f'outputs/{name}/params/delta_geo.pt').to(device)
    input = sp.SparseTensor(
        coords=coords,
        feats=feats
    )
    motion = np.load(f'outputs/{name}/slats/motion_sdf.npz')
    face_id = motion['face_id']
    uvw = motion['uvw']

    deformation = (smplx_v - smplx_v[0:1])[:, :-1092]
    deformation = deformation[:, motion_id, :]
    deformation = np.einsum('bkij, bki->bkj', deformation, np.tile(uvw, (deformation.shape[0], 1, 1)))

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


    trans = np.load(f'outputs/{name}/params/trans.npz')
    kt = np.load(f'outputs/{name}/params/kt.npz')
    k, t = list(kt.values())
    s, R, T = list(trans.values())
    s = torch.from_numpy(s).to(device)
    R = quaternion_to_matrix(torch.from_numpy(R).to(device))
    T = torch.from_numpy(T).to(device)
    k = torch.from_numpy(k).to(device)
    t = torch.from_numpy(t).to(device)
    # R = quaternion_to_matrix(torch.from_numpy(R).to(device))
    deformation = s.cpu().numpy() * k.cpu().numpy() * deformation @ R[0].transpose(-2, -1).cpu().numpy()

    n_coords = torch.load(f'outputs/{name}/params/v_pos_.pt').to(coords)
    m_mask = torch.load(f'outputs/{name}/params/marching_mask.pt').to(device)
    mesh = model(None, dcoords=coords, dfeats=feats, v_a=torch.from_numpy(deformation[-1]).to(dtype=torch.float32, device=device), mask=sdf_mask, n_coords=n_coords, marching_mask=m_mask, change_marching=True, name=name)
    mesh.vertices = (mesh.vertices - T) @ R[0] / s[0]
    mesh.vertices = (mesh.vertices - t) / k
    mesh.face_normal = torch.matmul(mesh.face_normal, R)
    mesh_out = render(mesh, extrinsic, proj[0], HW=[H, W])
    out = mesh_out['normal']
    print(out.shape)
    cv2.imwrite('mesh.png', out.permute(1, 2, 0).detach().cpu().numpy() * 255.)


    iters = 501
    delta = torch.zeros((feats.shape[0], 8)).to(device).requires_grad_(True)
    optimizer = Adam([delta], lr=5e-5)

    os.makedirs(f'outputs/{name}/mouth', exist_ok=True)
    indices = torch.load(f'outputs/{name}/params/indices.pt')
    BCE = torch.nn.BCELoss()
    grad_clip = AdaptiveGradClipper(max_norm=0.01, clip_percentile=95)

    for epoch in range(iters):
        optimizer.zero_grad()
        ff = torch.cat(
            [
                feats[..., :8].detach() + delta,
                feats[..., 8:].detach()
            ],
            dim=-1
        ).to(device)
        mesh = model(None, dcoords=coords, dfeats=ff, v_a=torch.from_numpy(deformation[-1]).to(dtype=torch.float32, device=device), mask=sdf_mask, n_coords=n_coords, marching_mask=m_mask, indices=indices)

        mesh2smplx(name, mesh)
        new_v, new_f = filter_z(mesh, mean_z)
        mesh = MeshExtractResult(vertices=new_v, faces=new_f)
        dicts = render(mesh, extrinsic, proj[0], HW=[H, W])

        # perceptual_loss_new = 10 * loss_recon(dicts['normal'][0] * local_mask_dilated[0, None], smplx_dicts_local['normal'][0] * local_mask_dilated[0, None]) + 5 * loss_recon(dicts['normal'][0] * (1 - local_mask[0, None]), (1 - local_mask[0, None]) * dicts_gt_['normal'][0])
        perceptual_loss = loss_recon(dicts['normal'] * local_mask[None], smplx_dicts['normal'] * local_mask[None])
        depth_loss = loss_recon(dicts['depth'][None] * local_mask[None], smplx_dicts['depth'][None] * local_mask[None])

        # mask_loss = loss_recon(dicts['mask'] * local_mask, smplx_dicts['mask'] * local_mask)
        mask_loss = 5 * BCE(dicts['mask'] * local_mask, smplx_dicts['mask'] * local_mask) + 0.5 * dice_loss(dicts['mask'] * local_mask, smplx_dicts['mask'] * local_mask)
        loss = perceptual_loss + 0.1 * depth_loss + 0.001 * mask_loss


        loss.backward()
        grad_norm = grad_clip(delta)
        optimizer.step()
        print(f'epoch: {epoch}, loss: {loss}')

        if epoch == 200:
            # mesh.vertices[..., -1:] += z_mean
            trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.cpu().numpy()).export(f'outputs/{name}/objects/single_n.obj')
            # torch.save(ff, f'exp/{mname}/params/delta_geo.pt')
            # cv2.imwrite(f'exp/{mname}/objects/dd.png', I_dep.detach().cpu().numpy()[..., ::-1] * 255.)
            # cv2.imwrite(f'exp/{mname}/objects/nn.png', I_nor.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)
            # cv2.imwrite(f'exp/{mname}/objects/dd1.png', I_dep1.detach().cpu().numpy()[..., ::-1] * 255.)
            # cv2.imwrite(f'exp/{mname}/objects/nn1.png', I_nor1.permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)


            # cv2.imwrite(f'exp/{mname}/objects/local_mask.png', local_mask_dilated.permute(1, 0, 2).flatten(start_dim=1).detach().cpu().numpy() * 255.)
            # cv2.imwrite(f'exp/{mname}/objects/local_normal.png', ((1 - local_mask_dilated[:, None]) * dicts_gt_['normal']).permute(1, 2, 0, 3).flatten(start_dim=2).permute(1, 2, 0).detach().cpu().numpy() * 255.)
        if epoch % 5 == 0:
        # if epoch == 200:
            # n = dicts['normal']
            cv2.imwrite(f'outputs/{name}/mouth/normal_{epoch}.png', dicts['normal'].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255.)

class OnlyInfo(logging.Filter):
    def filter(self, record):
        return record.levelno < logging.WARNING

import subprocess
from datetime import datetime


if __name__ == '__main__':
    # subprocess.run(['rm', '-f', 'info_geo.log'])
    # main('008_07')
    # name = '011_20'
    # main_daviad(name)
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

    paths = os.listdir('outputs')

    for path in paths:
        if os.path.exists(f'outputs/{path}/objects/single_david.obj'):
            logger.info(f'{path} has been processed')
            # exit()
            continue
        if os.path.exists(f'outputs/{path}/objects/new_outer_filtered.obj'):
            # print(path)
            try:
                print(f'processing {path}')
                main_daviad(path)
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