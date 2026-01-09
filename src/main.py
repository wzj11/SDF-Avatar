from ast import Str
from collections import defaultdict
import argparse
import datetime
import json
import math
import matplotlib.pyplot as plt
import os
import pickle
import random
import time
from multiprocessing import Pool
import os
import yaml

import cubvh
import cv2
import face_alignment
import numpy as np
import nvdiffrast.torch as dr
import open3d as o3d
import pymeshlab
import pyrender
import torch
import trimesh
from PIL import Image
from pymeshlab import Percentage
from pytorch3d.ops import knn_points
from pytorch3d.transforms import so3_exp_map
from scipy.spatial import cKDTree
from torch.optim.adam import Adam
from utils3d.numpy import quaternion_to_matrix as Quaternion
from utils3d.torch import matrix_to_quaternion, quaternion_to_matrix
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.utils_io import preprocess, read_png
from utils.utils_io import read_png

from models.smplx.smplx import SMPLX
from trellis.modules import sparse as sp
print('a')
from trellis.pipelines import TrellisImageTo3DPipeline
print('b')
from trellis.representations import Gaussian
from trellis.representations.mesh import SparseFeatures2Mesh, MeshExtractResult
from trellis.utils.render_utils import render_multiview, render_frames
# from flame1.flame import FLAME
from utils.utils_cam import sphere_hammersley_sequence, yaw_pitch_r_fov_to_extrinsics_intrinsics
from utils.utils_render import render as render_n


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)

args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

root_path = os.path.dirname(
    os.path.dirname(__file__)
)

print(cfg.keys())
# cfg['output_dir'] = 'outputs_track'
OUTPUT_PATH = cfg['output_dir']
MAX_SEED = np.iinfo(np.int32).max
COLOR = {
    'black': 40,
    'red': 41,
    'green': 42,
    'yellow': 43,
    'blue': 44,
    'purple': 45,
    'white': 47
}
num_betas = 300
num_expression_coeffs = 100
model_path = 'models/smplx/SMPLX2020'
smplx = SMPLX(num_betas=num_betas, num_expression_coeffs=num_expression_coeffs, model_path=model_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(OUTPUT_PATH, exist_ok=True)
smplx.to(device)
# lazy load

    
pipeline = TrellisImageTo3DPipeline.from_pretrained(f"{root_path}/TRELLIS-image-large")
pipeline.cuda()
# flameNet = FLAME().to(device)

def cprint(content, color='green'):
    color = COLOR[color]
    print(f'\033[{color}m{content}\033[0m')

def filter(name, mesh=None, obj_name='sample'):
    if mesh is None:
        mesh = trimesh.load(f'/home/wzj/project/TRELLIS/output/video/objects/{obj_name}_outer.obj')
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    ms.apply_filter(
        "meshing_remove_connected_component_by_diameter",
        mincomponentdiag=Percentage(10)
    )
    v = ms.current_mesh().vertex_matrix()
    f = ms.current_mesh().face_matrix()
    trimesh.Trimesh(vertices=v, faces=f).export(f'{OUTPUT_PATH}/{name}/objects/{obj_name}_outer_filtered.obj')
    return v, f, trimesh.Trimesh(vertices=v, faces=f)

def outer(
        mesh: trimesh.Trimesh,
        name,
        obj_name='sample',
):
    if mesh is None:
        mesh = trimesh.load('')
    glctx = dr.RasterizeCudaContext()
    verts = torch.from_numpy(mesh.vertices).to(device).float()
    faces = torch.from_numpy(mesh.faces).to(device, torch.int32)
    verts = torch.nn.functional.pad(verts, (0, 1), 'constant', 1)
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, 120) for i in range(120)]
    cams = [item for item in cams if item[1]>-math.pi/6]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    num = len(cams)
    extrinsics = torch.stack(extrinsics, dim=0).to(device).float()
    intrinsics = torch.stack(intrinsics, dim=0).to(device).float()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    near, far = 1, 100.
    proj = intrinsics.new_zeros((num, 4, 4))
    proj[:, 0, 0] = 2 * fx
    proj[:, 1, 1] = 2 * fy
    proj[:, 0, 2] = 2 * cx - 1
    proj[:, 1, 2] = - 2 * cy + 1
    proj[:, 2, 2] = far / (far - near)
    proj[:, 2, 3] = near * far / (near - far)
    proj[:, 3, 2] = 1.
    verts_cam = verts.expand(extrinsics.shape[0], -1, -1).bmm(extrinsics.transpose(-1, -2))
    verts_clip = verts_cam.bmm(proj.transpose(-1, -2))
    rast, _ = dr.rasterize(
        glctx,
        verts_clip,
        faces,
        resolution=[1024, 1024]
    )
    tris = rast[..., -1].cpu().numpy().flatten()
    tris = np.unique(tris)
    tris = tris[tris!=0] - 1
    tris = tris.astype(np.int32)
    submesh = mesh.submesh([tris])
    # submesh[0].export(f'{OUTPUT_PATH}/{name}/objects/{obj_name}_outer.obj')
    v, f, tmesh = filter(name, submesh[0], obj_name)
    return tmesh 

def generate(image_path, name, obj_name='sample', coords=None): 
    
    os.makedirs(f'{OUTPUT_PATH}/{name}/objects', exist_ok=True)
    # image = Image.open(image_path)
    image = preprocess(name, cfg['input_dir'], '000000')
    seed = random.randint(0, MAX_SEED)
    cprint('start generating')
    outputs, slat = pipeline.run(
        image,
        seed=seed,
        preprocess_image=True,
        sparse_structure_sampler_params={
            "steps": 30,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 30,
            "cfg_strength": 3,
        },
        coords=coords,
        save=True,
        name=None,
        obj_name=obj_name
    )

    os.makedirs(f'{OUTPUT_PATH}/{name}/slats', exist_ok=True)
    # TODO
    # torch.save(slat.coords, f'{OUTPUT_PATH}/{name}/slats/coords_coarse_{obj_name}.pt')
    # torch.save(slat.feats, f'{OUTPUT_PATH}/{name}/slats/feats_coarse_{obj_name}.pt')
    v = outputs['mesh'][0].vertices
    f = outputs['mesh'][0].faces
    # outputs['gaussian'][0].save_ply(f'{OUTPUT_PATH}/{name}/objects/{obj_name}.ply')
    m = trimesh.Trimesh(vertices=v.detach().cpu().numpy(), faces=f.detach().cpu().numpy())
    # m.export(f'{OUTPUT_PATH}/{name}/objects/{obj_name}.obj')
    output = outer(m, name, obj_name)

    mesh_gene = outputs['mesh'][0]
    mesh_info = {
        'vertices': mesh_gene.vertices,
        'faces': mesh_gene.faces,
        'vertex_attrs': mesh_gene.vertex_attrs
    }
    torch.save(mesh_info, f'{OUTPUT_PATH}/{name}/objects/mesh_gene.pt')
    return output, outputs['mesh'][0]


def generate_old(image_path, name, obj_name='sample', coords=None): 
    # global pipeline
    # os.system('ln -sf trellis/slat_flow_img_dit_L_64l8p2_fp16.safetensors TRELLIS-image-large/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors')
    # pipeline = TrellisImageTo3DPipeline.from_pretrained(f"{root_path}/TRELLIS-image-large")
    # pipeline.cuda()
    if not os.path.exists(f'{OUTPUT_PATH}/{name}/objects'):
        os.makedirs(f'{OUTPUT_PATH}/{name}/objects', exist_ok=True)
        image = Image.open(image_path)
        seed = random.randint(0, MAX_SEED)
        cprint('start generating')
        outputs, slat = pipeline.run(
            image,
            seed=seed,
            preprocess_image=True,
            sparse_structure_sampler_params={
                "steps": 25,
                "cfg_strength": 7.5,
            },
            slat_sampler_params={
                "steps": 25,
                "cfg_strength": 3,
            },
            coords=coords,
            save=True,
            name=None,
            obj_name=obj_name
        )
        outputs['gaussian'][0].save_ply(f'{OUTPUT_PATH}/{name}/objects/gaussian.ply')
        if not os.path.exists(f'{OUTPUT_PATH}/{name}/slats'):
            os.makedirs(f'{OUTPUT_PATH}/{name}/sl ats', exist_ok=True)
        # TODO
        # torch.save(slat.coords, f'{OUTPUT_PATH}/{name}/slats/coords_coarse_{obj_name}.pt')
        # torch.save(slat.feats, f'{OUTPUT_PATH}/{name}/slats/feats_coarse_{obj_name}.pt')
        v = outputs['mesh'][0].vertices
        f = outputs['mesh'][0].faces
        # outputs['gaussian'][0].save_ply(f'{OUTPUT_PATH}/{name}/objects/{obj_name}.ply')
        m = trimesh.Trimesh(vertices=v.detach().cpu().numpy(), faces=f.detach().cpu().numpy())
        # m.export(f'{OUTPUT_PATH}/{name}/objects/{obj_name}.obj')
        output = outer(m, name, obj_name)
        return output, outputs['gaussian'][0], slat, outputs['mesh'][0]
    else:
        mesh = trimesh.load(f'{OUTPUT_PATH}/{name}/objects/sample_outer_filtered.obj')
        gs = Gaussian(
            sh_degree=0,
            aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
            mininum_kernel_size = 9e-4,
            scaling_bias = 4e-3,
            opacity_bias = 0.1,
            scaling_activation = 'softplus'
        )
        
        gs.load_ply(f'{OUTPUT_PATH}/{name}/objects/gaussian.ply')
        # print(type(gaussian))
        # exit()
        # coords = torch.load( f'{OUTPUT_PATH}/{name}/slats/coords_coarse_sample.pt')
        # feats = torch.load( f'{OUTPUT_PATH}/{name}/slats/feats_coarse_sample.pt')
        # slat = sp.SparseTensor(
        #     coords=coords,
        #     feats=feats
        # )
        return mesh, gs, None

def flame2voxel(code_path, k=None, t=None, other_mesh=None):
    # stand_v, stand_f = code2flame('', random=True)
    # flame_v, flame_f = code2flame(code_path)
    v = code_path.vertices
    # f = flame_f.cpu().numpy()
    f = code_path.faces
    # subdivide
    # if k is None:
    #     k = 0.9 / (v.max() - v.min())
    #     t = - 0.45 * (v.max() + v.min()) / (v.max() - v.min())
    # k = np.array([2.7039])
    # t = np.array([0.0651])
    # print(k, t)
    # exit()
    # v = v * k + t
    # v = np.clip(v, -0.5 + 1e-6, 0.5 - 1e-6)
    # trans = np.array([
    #     [1., 0., 0.],
    #     [0., 0., -1.],
    #     [0., 1., 0.]
    # ])
    # v = v @ trans.T
    fm = trimesh.Trimesh(vertices=v, faces=f)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(v, f))
    ms.apply_filter(
        "meshing_surface_subdivision_ls3_loop",
        iterations=3,
        threshold=Percentage(0),
    )
    v = ms.current_mesh().vertex_matrix()
    f = ms.current_mesh().face_matrix()
    # preprocess for trellis
    
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(v),
        triangles=o3d.utility.Vector3iVector(f),
    )
    if other_mesh is not None:
        second_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(other_mesh.vertices),
            triangles=o3d.utility.Vector3iVector(other_mesh.faces),
        )
    my_v = v
    my_f = f
    my_mesh = trimesh.Trimesh(vertices=my_v, faces=my_f)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    voxel_grid1 = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(second_mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    vertices1 = np.array([voxel.grid_index for voxel in voxel_grid1.get_voxels()])

    print(vertices.min())
    coords = torch.from_numpy(vertices).to(device)
    coords1 = torch.from_numpy(vertices1).to(device)
    coords = torch.cat([coords, coords1], dim=0)
    coords = coords.unique(dim=0)
    coords = torch.nn.functional.pad(coords, [1, 0], 'constant', 0)
    # fm = trimesh.Trimesh(vertices=flame_v.cpu().numpy(), faces=flame_f.cpu().numpy())
    return coords, my_mesh, fm, k, t

def render(gaussian, name, return_=False, extr=None, intr=None):
    # if return_:
    #     extr = torch.from_numpy(extr).to(device).float()
    #     intr = torch.from_numpy(intr).to(device).float()


    #     color = render_frames(gaussian, extr, intr, {'resolution': 1024, 'bg_color': (0, 0, 0)})
    #     cc = color['color']
    #     for num, i in enumerate(cc):
    #         cv2.imwrite(f"{OUTPUT_PATH}/{name}/images/samplex_{num}.png", i[..., ::-1])
    #     return
    if not os.path.exists(f'{OUTPUT_PATH}/{name}/images/sample_0.png'):
        os.makedirs(f'{OUTPUT_PATH}/{name}/images', exist_ok=True)
        os.makedirs(f'{OUTPUT_PATH}/{name}/params', exist_ok=True)
        cprint('rendering trellis gaussians')
        observations, extrinsics, intrinsics = render_multiview(gaussian, resolution=1024, nviews=100, img_for_lmk=True)
        extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
        intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]
        for num, i in enumerate(observations):
            cv2.imwrite(f"{OUTPUT_PATH}/{name}/images/sample_{num}.png", i[..., ::-1])
        extrinsics = np.stack(extrinsics, axis=0)
        intrinsics = np.stack(intrinsics, axis=0)
        np.save(f"{OUTPUT_PATH}/{name}/params/extrinsics.npy", extrinsics)
        np.save(f"{OUTPUT_PATH}/{name}/params/intrinsics.npy", intrinsics)

        return np.stack([observations[0], observations[1], observations[2]], axis=0), extrinsics, intrinsics
    else:
        imgs = [cv2.imread(f'{OUTPUT_PATH}/{name}/images/sample_{num}.png') for num in [0, 1, 2]]
        imgs = np.stack(imgs, axis=0)
        imgs = imgs[..., ::-1]
        extrinsics = np.load(f'{OUTPUT_PATH}/{name}/params/extrinsics.npy')
        intrinsics = np.load(f'{OUTPUT_PATH}/{name}/params/intrinsics.npy')
        extrinsics = extrinsics
        intrinsics = intrinsics
        return imgs, extrinsics, intrinsics

def landmarks(name, images=None):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')
    cprint('detect landmarks from trellis renders')
    if images is None:
        preds = fa.get_landmarks_from_directory(f'{OUTPUT_PATH}/{name}/images')
    lmks = list(preds.values())
    # idx = (~np.array([lmk==None for lmk in lmks])).nonzero()[0]
    # selected = np.random.choice(idx, size=5, replace=False)
    selected = np.array([0, 1, 2])
    # print('\033[42msaving landmarks\033[0m')
    cprint('saving landmarks')
    os.makedirs(f'{OUTPUT_PATH}/{name}/lmks', exist_ok=True)
    # for num, i in enumerate(lmks):
    #     np.save(f'{OUTPUT_PATH}/{name}/lmks/lmk_{num}.npy', i[0])
    gts = np.stack([lmks[id][0] for id in selected], axis=0) # (3, 68, 2)
    imgs1 = np.zeros((1024, 1024))
    imgs1[lmks[0][0][:, 1].astype(np.int32), lmks[0][0][:, 0].astype(np.int32)] = 255
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/lmks/65.png', imgs1)
    imgs2 = np.zeros((1024, 1024))
    imgs2[lmks[1][0][:, 1].astype(np.int32), lmks[1][0][:, 0].astype(np.int32)] = 255
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/lmks/69.png', imgs2)
    imgs3 = np.zeros((1024, 1024))
    imgs3[lmks[2][0][:, 1].astype(np.int32), lmks[2][0][:, 0].astype(np.int32)] = 255
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/lmks/70.png', imgs3)

    
    return gts, selected

def flame_track(name, img_path=None, t=0, return_mesh=True):
    os.makedirs(f'{OUTPUT_PATH}/{name}/params', exist_ok=True)
    ...
    if os.path.exists(f'{OUTPUT_PATH}/{name}/params/{t}.npy'):
        output = np.load(f'{OUTPUT_PATH}/{name}/params/{t}.npy')
    else:
        ...
    if return_mesh:
        shape_code, exp_code = (
            torch.from_numpy(output[:, :300]).to(device),
            torch.from_numpy(output[:, 300:]).to(device)
        )
        flame_v = flameNet.forward_geo(shape_code ,exp_code)
        flame_f = flameNet.faces_tensor
        mesh = trimesh.Trimesh(vertices=flame_v[0].detach().cpu().numpy(), faces=flame_f.detach().cpu().numpy())

    # (1, 400)
    return output, mesh

def flame2smplx():
    flame_idx = 'models/flame_static_embedding_68.pkl'
    with open(flame_idx, 'rb') as f:
        # lmk_face_idx: (68, )
        lmk_face_idx, lmk_b_coords = pickle.load(f, encoding='latin1').values()
    mesh = trimesh.load('/home/wzj/project/TRELLIS/output/flame_018/objects/flame.obj')
    flame_v_idx = mesh.faces[lmk_face_idx] # (68, 3)
    flame_map_smplx = np.load('/home/wzj/project/TRELLIS/models/smplx/corresponding/SMPL-X__FLAME_vertex_ids.npy')
    smplx_v_idx = flame_map_smplx[flame_v_idx] #(68, 3) smplx index for 68 lmks
    return smplx_v_idx, lmk_b_coords

def smplx_track(name, img_path=None, t=0, return_mesh=True):
    os.makedirs(f'{OUTPUT_PATH}/{name}/params', exist_ok=True)
    ...
    data = torch.load(f"{cfg['input_dir']}/{name}/body_track/smplx_track.pth")
    # data = torch.load(f'{OUTPUT_PATH}/{name}/params/smplx_track.pth')
    _, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, shape, expr, global_orient, transl = data.values()
    betas = torch.from_numpy(shape).to(device)
    expr = torch.from_numpy(expr).to(device)
    leye_pose = torch.from_numpy(leye_pose).to(device)
    reye_pose = torch.from_numpy(reye_pose).to(device)
    jaw_pose = torch.from_numpy(jaw_pose).to(device)
    body_pose = torch.from_numpy(body_pose).to(device)
    global_orient = torch.from_numpy(global_orient).to(device)
    transl = torch.from_numpy(transl).to(device)
    

    if return_mesh:
        output = smplx.forward(
            betas=betas[t:t+1], 
            jaw_pose=jaw_pose[t:t+1],
            leye_pose=leye_pose[t:t+1],
            reye_pose=reye_pose[t:t+1], 
            expression=expr[t:t+1],
            body_pose=body_pose[t:t+1],
            global_orient=global_orient[t:t+1],
            transl=transl[t:t+1],
            with_rott_return=True
        )
        smplx_v = output['vertices'].detach().cpu().numpy()
        f = smplx.faces
        smplx_f = torch.from_numpy(f).to(device).int()
        # rota = np.array(
        #     [
        #         [1., 0., 0.],
        #         [0., 0., -1.],
        #         [0., 1., 0.]
        #     ]
        # )
        # smplx_v = smplx_v[0] @ rota.T
        mesh = trimesh.Trimesh(vertices=smplx_v[0], faces=f)
        # mesh.export(f'{OUTPUT_PATH}/{name}/objects/smplx_origin.obj')
    # (1, 400)
    return output, mesh

def align(name, flame_mesh, gts, exs, ins):
    cprint('beginning alignment')
    initial_rotation = torch.tensor([
        [1., 0., 0.],
        [0., 0., -1.],
        [0., 1., 0.]
    ]).unsqueeze(0).to(device)
    s, R, T = (
        torch.ones(1, 1).to(device).requires_grad_(),
        matrix_to_quaternion(initial_rotation).to(device).requires_grad_(),
        torch.zeros(1, 3).to(device).requires_grad_()
    )
    optimizer = Adam([s, R, T], lr=1e-3)
    with open('models/flame_static_embedding_68.pkl', 'rb') as f:
        lmk_face_idx, lmk_b_coords = pickle.load(f, encoding='latin1').values()
        lmk_face_idx = torch.tensor(lmk_face_idx).to(device)
        lmk_b_coords = torch.tensor(lmk_b_coords).to(device)
    v = flame_mesh.vertices
    k = 0.9 / (v.max() - v.min())
    t = - 0.45 * (v.max() + v.min()) / (v.max() - v.min())
    v = v * k + t
    np.savez(f'{OUTPUT_PATH}/{name}/params/kt.npz', k=k, t=t)
    
    vertices = np.clip(v, -0.5 + 1e-6, 0.5 - 1e-6)
    v_torch = torch.from_numpy(vertices).to(device)
    f_torch = torch.from_numpy(flame_mesh.faces).to(device).int()
    lmk_v = v_torch[f_torch[lmk_face_idx.to(torch.long)]]
    lmk_v = torch.einsum('bij, bi->bj', lmk_v.float(),     lmk_b_coords.float())
    gts, exs, ins = (
        torch.from_numpy(gts).to(device).float(),
        torch.from_numpy(exs).to(device).float(),
        torch.from_numpy(ins).to(device).float()
    )
    fx, fy, cx, cy = ins[0, 0, 0], ins[0, 1, 1], ins[0, 0, 2], ins[0, 1, 2]
    for iter in range(1500):
        optimizer.zero_grad()
        Rota = quaternion_to_matrix(R).squeeze(0)
        out = s * lmk_v @ Rota.transpose(-2, -1) + T
        if iter == 1000:
            output_v = s * v_torch.float() @ Rota.transpose(-2, -1) + T
            output_f = flameNet.faces_tensor.cpu().numpy()
            output_v = output_v.detach().cpu().numpy()
            output_mesh = trimesh.Trimesh(vertices=output_v, faces=output_f)
            output_mesh.export(f'{OUTPUT_PATH}/{name}/objects/flame.obj')
        world = torch.nn.functional.pad(out, [0, 1], 'constant', 1)
        # exs_t = torch.bmm(exs, Rota.repeat(3, 1, 1))
        cams = torch.einsum('bij,kj->bki', exs, world)
         # (3, 68, 4)
        X = cams[..., 0]
        Y = cams[..., 1]
        Z = cams[..., 2]
        result = torch.zeros(3, 68, 2).to(device)
        result[..., 0] = fx * X / Z + cx
        result[..., 1] = fy * Y / Z + cy
        # print(result[0].max(), result[0].min())
        loss_points = torch.nn.functional.l1_loss(result, gts / 1024)
        loss = loss_points

        loss.backward()

        optimizer.step()

        if iter % 100 == 0:
            print(f'epoch: {iter}, loss: {loss.item()}, loss_points: {loss_points.item()}')
    
    np.savez(f'{OUTPUT_PATH}/{name}/params/trans.npz', s=s.detach().cpu().numpy(), R=R.detach().cpu().numpy(), T=T.detach().cpu().numpy())
    return output_mesh



def mesh2smplx(mname, mesh=None, s=None, R=None, T=None, k=None, t=None, device='cuda:0'):
    from utils3d.torch import quaternion_to_matrix
    if s is None:
        trans = np.load(f'outputs_ners/{mname}/params/trans.npz')
        kt = np.load(f'outputs_ners/{mname}/params/kt.npz')
    if s is None:
        k, t = list(kt.values())

        s, R, T = list(trans.values())
        s = torch.from_numpy(s).to(device)
        R = quaternion_to_matrix(torch.from_numpy(R).to(device))
        T = torch.from_numpy(T).to(device)
        k = torch.from_numpy(k).to(device)
        t = torch.from_numpy(t).to(device)
    if mesh is not None:
        mesh.vertices = (mesh.vertices - T) @ R / s[0]
        mesh.vertices = (mesh.vertices - t) / k
        mesh.face_normal = torch.matmul(mesh.face_normal, R)


    # if return_:
    #     return s, R, T, k, t

def smplxalign(name, smplx_mesh, gts, exs, ins, gs=None, mmesh=None):
    iters = 2000
    head_index = np.load('models/smplx/corresponding/SMPL-X__FLAME_vertex_ids.npy')
    smplx_params = np.load('models/smplx/SMPLX2020/SMPLX_NEUTRAL.npz')
    lmk_face_idx, lmk_b_coords = smplx_params['lmk_faces_idx'], smplx_params['lmk_bary_coords']
    cprint('beginning alignment')
    initial_rotation = torch.tensor([
        [1., 0., 0.],
        [0., 0., -1.],
        [0., 1., 0.]
    ]).unsqueeze(0).to(device)
    s, R, T = (
        torch.ones(1, 1).to(device).requires_grad_(),
        matrix_to_quaternion(initial_rotation).to(device).requires_grad_(),
        torch.zeros(1, 3).to(device).requires_grad_()
    )
    optimizer = Adam([s, R, T], lr=1e-2)
    # lmk_v_idx, lmk_b_coords = flame2smplx()
    # lmk_v_idx = torch.from_numpy(lmk_v_idx).to(device)
    # lmk_b_coords = torch.from_numpy(lmk_b_coords).to(device)
    v = smplx_mesh.vertices
    v_head = v[head_index]
    k = 0.9 / (v_head.max() - v_head.min())
    t = - 0.45 * (v_head.max() + v_head.min()) / (v_head.max() - v_head.min())
    v = v * k + t
    np.savez(f'{OUTPUT_PATH}/{name}/params/kt.npz', k=k, t=t)
    
    vertices = np.clip(v, -0.5 + 1e-6, 0.5 - 1e-6)
    # vertices = v
    v_torch = torch.from_numpy(vertices).to(device)
    f_torch = torch.from_numpy(smplx.faces).to(device).int()
    lmk_v = v_torch[f_torch[torch.from_numpy(lmk_face_idx).to(device).int()]] #(51, 3, 3)
    lmk_v = torch.einsum('bij, bi->bj', lmk_v.float(),     torch.from_numpy(lmk_b_coords).to(device).float())
    # gts, exs, ins 
    gts = torch.from_numpy(gts).to(device).float()
    exs = torch.from_numpy(exs).to(device).float()
    # exs[..., :1] *= -1
    print(exs[[1]])
    print(torch.inverse(exs[[1]]))
    ins = torch.from_numpy(ins).to(device).float()
    # if gs is not None:
    #     print(type(gs))
    #     color = render_frames(gs, exs[[1]], ins[[1]], {'resolution': 1024, 'bg_color': (0, 0, 0)})
    #     cc = color['color']
    #     for num, i in enumerate(cc):
    #         cv2.imwrite(f"{OUTPUT_PATH}/{name}/images/samplex_{num}.png", i[..., ::-1])
    fx, fy, cx, cy = ins[0, 0, 0], ins[0, 1, 1], ins[0, 0, 2], ins[0, 1, 2]

    track_path = f'{cfg["input_dir"]}/{name}/body_track/smplx_track.pth'
    smplx_params = torch.load(track_path)
    cam_para, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, shape, expr, global_orient, transl = to_device(**smplx_params)
    parse = read_png(f'{cfg["input_dir"]}/{name}/parsing')[0]
    # ffm = ((parse == 2) | ((parse > 5) & (parse < 14)))
    ffm = ((parse != 1) & (parse != 3) & (parse != 0))
    cv2.imwrite('test_mask_mmm.png', ffm.astype(np.uint8) * 255.)
    face_mask = (ffm.astype(np.uint8) * (parse!=0)).astype(np.uint8)
    face_mask = torch.from_numpy(face_mask).to(device)
    x = face_mask.nonzero()
    x_ind = x[:, 0].max().item()
    my_mask = torch.zeros(*face_mask.shape[:2]).to(device)
    my_mask[:x_ind] = 1.
    print(my_mask.shape)
    cv2.imwrite('test_wzjwzj.png', my_mask.cpu().numpy() * 255.)

    H, W = parse.shape[:2]
    extrinsic = torch.tensor(
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ]
    ).to(device)
    proj = get_ndc_proj_matrix(cam_para[0:1], [H, W])
    gt_mask = cv2.imread(f'{cfg["input_dir"]}/{name}/seg_masks/000000.png', cv2.IMREAD_UNCHANGED)
    gt_mask = face_mask[..., 0]


    # BCE = torch.nn.BCELoss()

    for iter in range(iters+1):
        optimizer.zero_grad()
        Rota = quaternion_to_matrix(R).squeeze(0)
        out = s * lmk_v @ Rota.transpose(-2, -1) + T
        if iter == iters:
            output_v = s * v_torch.float() @ Rota.transpose(-2, -1) + T
            output_f = smplx.faces
            output_v = output_v.detach().cpu().numpy()
            output_mesh = trimesh.Trimesh(vertices=output_v, faces=output_f)
            output_mesh.export(f'{OUTPUT_PATH}/{name}/objects/smplx.obj')
        world = torch.nn.functional.pad(out, [0, 1], 'constant', 1)
        # exs_t = torch.bmm(exs, Rota.repeat(3, 1, 1))
        # cams = torch.einsum('bij,kj->bki', exs, world)
        cams = torch.bmm(world.repeat(3, 1, 1), exs.transpose(-2, -1))
         # (3, 68, 4)
        X = cams[..., 0]
        Y = cams[..., 1]
        Z = cams[..., 2]
        result = torch.zeros(exs.shape[0], 51, 2).to(device)
        result[..., 0] = fx * X / Z + cx
        # print()
        result[..., 1] = fy * Y / Z + cy
        with torch.no_grad():
            result1 = torch.zeros(exs.shape[0], 51, 2).to(device)
            result1[..., 0] = fx * X / Z + cx
            # print()
            result1[..., 1] = fy * Y / Z + cy
        # print(result[0].max(), result[0].min())
        if iter == iters:
            imgs1 = np.zeros((1024, 1024))
            imgs1[gts[0][17:, 1].detach().cpu().numpy().astype(np.int32), gts[0][17:, 0].detach().cpu().numpy().astype(np.int32)] = 255
            imgs11 = np.zeros((1024, 1024))
            imgs11[(result1[0][:, 1] * 1024).detach().cpu().numpy().astype(np.int32), (result1[0][:, 0] * 1024).detach().cpu().numpy().astype(np.int32)] = 255
            cv2.imwrite(f'{OUTPUT_PATH}/{name}/lmks/65_.png', imgs1)
            cv2.imwrite(f'{OUTPUT_PATH}/{name}/lmks/65__.png', imgs11)

            imgs2 = np.zeros((1024, 1024))
            imgs2[gts[1][17:, 1].detach().cpu().numpy().astype(np.int32), gts[1][17:, 0].detach().cpu().numpy().astype(np.int32)] = 255
            imgs21 = np.zeros((1024, 1024))
            imgs21[(result1[1][:, 1] * 1024).detach().cpu().numpy().astype(np.int32), (result1[1][:, 0] * 1024).detach().cpu().numpy().astype(np.int32)] = 255
            cv2.imwrite(f'{OUTPUT_PATH}/{name}/lmks/69_.png', imgs2)
            cv2.imwrite(f'{OUTPUT_PATH}/{name}/lmks/69__.png', imgs21)

            imgs3 = np.zeros((1024, 1024))
            imgs3[gts[2][17:, 1].detach().cpu().numpy().astype(np.int32), gts[2][17:, 0].detach().cpu().numpy().astype(np.int32)] = 255
            imgs31 = np.zeros((1024, 1024))
            imgs31[(result1[2][:, 1] * 1024).detach().cpu().numpy().astype(np.int32), (result1[2][:, 0] * 1024).detach().cpu().numpy().astype(np.int32)] = 255
            cv2.imwrite(f'{OUTPUT_PATH}/{name}/lmks/70_.png', imgs3)
            cv2.imwrite(f'{OUTPUT_PATH}/{name}/lmks/70__.png', imgs31)

        loss_points = 1 * torch.nn.functional.l1_loss(result, gts[:, 17:, :] / 1024)
        # loss = loss_points
        loss = 0.0
        lambda_points = 1

        if iter > 2000:
            lambda_points = 2.5
            new_mesh = MeshExtractResult(vertices=mmesh.vertices.detach(), faces=mmesh.faces.detach())
            # print(mmesh.vertices.shape)

            mesh2smplx(name, new_mesh, s=s, R=Rota, T=T, k=torch.from_numpy(np.asarray(k)).to(device), t=torch.from_numpy(np.asarray(t)).to(device))
            # print(new_mesh.vertices.shape)
            dicts = render_n(new_mesh, extrinsic, proj[0], HW=[H, W], return_types=['mask'])

            # bc_loss = BCE(dicts['mask'].squeeze() * my_mask.detach(), gt_mask * my_mask.detach())

            loss += torch.nn.functional.l1_loss(dicts['mask'].squeeze() * my_mask.detach(), gt_mask * my_mask.detach())

            if iter % 100 == 0:
                cv2.imwrite(f'test_new_mask_{iter}.png', dicts['mask'].squeeze().detach().cpu().numpy() * 255.)
        loss += loss_points

        
        # out = result.detach().cpu().numpy()
        # imgs1 = np.zeros((1024, 1024))
        # imgs1[out[0][:, 0].astype(np.int32), out[0][:, 1].astype(np.int32)] = 255
        # cv2.imwrite(f'{OUTPUT_PATH}/{name}/lmks/smplx_65.png', imgs1)
        # imgs2 = np.zeros((1024, 1024))
        # imgs2[out[1][:, 0].astype(np.int32), out[1][:, 1].astype(np.int32)] = 255
        # cv2.imwrite(f'{OUTPUT_PATH}/{name}/lmks/smplx_69.png', imgs2)
        # imgs3 = np.zeros((1024, 1024))
        # imgs3[out[2][:, 0].astype(np.int32), out[2][:, 1].astype(np.int32)] = 255
        # cv2.imwrite(f'{OUTPUT_PATH}/{name}/lmks/smplx_70.png', imgs3)
        # loss = loss_points

        loss.backward()

        optimizer.step()

        if iter % 100 == 0:
            print(f'epoch: {iter}, loss: {loss.item()}, loss_points: {loss_points.item()}')
    
    np.savez(f'{OUTPUT_PATH}/{name}/params/trans.npz', s=s.detach().cpu().numpy(), R=R.detach().cpu().numpy(), T=T.detach().cpu().numpy())
    return output_mesh

def smplx_sample(v, f):
    glctx = dr.RasterizeCudaContext()
    verts = v.float()
    faces = f.int()
    verts = torch.nn.functional.pad(verts, (0, 1), 'constant', 1)
    
    r = 2
    fov = 40
    cams = [[0., 0.]]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    num = len(cams)
    extrinsics = torch.stack(extrinsics, dim=0).to(device).float()
    intrinsics = torch.stack(intrinsics, dim=0).to(device).float()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    near, far = 1, 100.
    proj = intrinsics.new_zeros((num, 4, 4))
    proj[:, 0, 0] = 2 * fx
    proj[:, 1, 1] = 2 * fy
    proj[:, 0, 2] = 2 * cx - 1
    proj[:, 1, 2] = - 2 * cy + 1
    proj[:, 2, 2] = far / (far - near)
    proj[:, 2, 3] = near * far / (near - far)
    proj[:, 3, 2] = 1.
    verts_cam = verts.expand(extrinsics.shape[0], -1, -1).bmm(extrinsics.transpose(-1, -2))
    verts_clip = verts_cam.bmm(proj.transpose(-1, -2))
    rast, _ = dr.rasterize(
        glctx,
        verts_clip,
        faces,
        resolution=[1024, 1024]
    )
    
    return rast

def flame_sample(name, mesh):
    os.makedirs(f'{OUTPUT_PATH}/{name}/mask', exist_ok=True)
    if mesh is None:
        mesh = trimesh.load('')
    face_index = np.load('/home/wzj/project/TRELLIS/flame1/data/faces.npy')
    mesh1 = mesh.submesh([face_index])
    mesh = mesh1[0]
    glctx = dr.RasterizeCudaContext()
    verts = torch.from_numpy(mesh.vertices).to(device).float()
    faces = torch.from_numpy(mesh.faces).to(device, torch.int32)
    verts = torch.nn.functional.pad(verts, (0, 1), 'constant', 1)
    
    r = 2
    fov = 40
    cams = [[0., 0.]]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    num = len(cams)
    extrinsics = torch.stack(extrinsics, dim=0).to(device).float()
    intrinsics = torch.stack(intrinsics, dim=0).to(device).float()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    near, far = 1, 100.
    proj = intrinsics.new_zeros((num, 4, 4))
    proj[:, 0, 0] = 2 * fx
    proj[:, 1, 1] = 2 * fy
    proj[:, 0, 2] = 2 * cx - 1
    proj[:, 1, 2] = - 2 * cy + 1
    proj[:, 2, 2] = far / (far - near)
    proj[:, 2, 3] = near * far / (near - far)
    proj[:, 3, 2] = 1.
    verts_cam = verts.expand(extrinsics.shape[0], -1, -1).bmm(extrinsics.transpose(-1, -2))
    verts_clip = verts_cam.bmm(proj.transpose(-1, -2))
    rast, _ = dr.rasterize(
        glctx,
        verts_clip,
        faces,
        resolution=[1024, 1024]
    )
    image = ((rast[..., -1].detach().cpu().numpy())!=0).astype(np.int32) * 255
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/mask/mask.png', image[0])
    return image
    # tris = rast[..., -1].cpu().numpy().flatten()
    # tris = np.unique(tris)
    # tris = tris[tris!=0] - 1
    # tris = tris.astype(np.int32)
    # submesh = mesh.submesh([tris])
    # submesh[0].export(f'{OUTPUT_PATH}/{name}/objects/sample_outer.obj')
    # v, f, tmesh = filter(name, submesh[0])
    # return tmesh

def cotangent(u, v):
    '''
    u and v are all vectors,
    if a rectangle is (i, j, k),
    u = i - k, v = j - k,
    then the cot of k is
    (u dot v) / || u X v|| 
    '''
    dot = np.einsum('ij,ij->i', u, v)
    cross = np.linalg.norm(np.cross(u, v), axis=1)
    cot = dot / (cross + 1e-8)
    return cot

def compute_cot_weights(mesh):
    V = mesh.vertices
    F = mesh.faces

    # 每个三角面三个边方向（按逆时针）
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    v0, v1, v2 = V[i0], V[i1], V[i2]

    # 角 α 对应于点 v0
    u0 = v2 - v1
    v0_ = v0 - v1
    cot0 = cotangent(u0, v0_)

    # 角 β 对应于点 v1
    u1 = v0 - v2
    v1_ = v1 - v2
    cot1 = cotangent(u1, v1_)

    # 角 γ 对应于点 v2
    u2 = v1 - v0
    v2_ = v2 - v0
    cot2 = cotangent(u2, v2_)

    # 每条边贡献到两个顶点之间的权重
    W = {}

    def add_edge_weight(i, j, w):
        key = tuple(sorted((i, j)))
        W[key] = W.get(key, 0) + w

    for idx in range(F.shape[0]):
        add_edge_weight(i0[idx], i1[idx], cot2[idx] / 2)
        add_edge_weight(i1[idx], i2[idx], cot0[idx] / 2)
        add_edge_weight(i2[idx], i0[idx], cot1[idx] / 2)

    return W # {(i,j): weight}

def one_ring(trellis, num, seed=42):
    # why random seed?
    torch.manual_seed(seed)
    random.seed(seed)

    # use defaultdict because every point's neighbors in mesh is a set instead a list because duplicated adjacency in different faces, use set to delete duplicated points
    adjacency = defaultdict(set)
    for face in trellis.faces:
        i, j, k = face
        adjacency[i].update([j, k])
        adjacency[j].update([i, k])
        adjacency[k].update([i, j])
    
    W = compute_cot_weights(trellis)
    for i in range(trellis.vertices.shape[0]):
        W[(i, i)] = 0
    neighbors = []
    weights = []
    for i in range(trellis.vertices.shape[0]):
        nbrs = list(adjacency[i])
        if len(nbrs) == 0:
            # fallback: 用自己
            nbrs = [i]
        if len(nbrs) >= num:
            selected = random.sample(nbrs, num)
        else:
            # 填充：重复已有邻居直到数量达到 k
            selected = nbrs + random.choices(nbrs, k=num - len(nbrs))
        neighbors.append(selected)
        weights.append([W[tuple(sorted((i, j)))] for j in selected])
    index = np.array(neighbors) # (vertices.shape[0], num)
    
    return torch.tensor(neighbors, dtype=torch.long).to(device), torch.tensor(weights, dtype=torch.float).to(device)

def compute_vertex_normals(v, f, eps=1e-5):
    """
    v: (N, 3) float32 or float64 Tensor
    f: (M, 3) long/int64 Tensor
    eps: 小常数，防止除零
    返回: (N, 3) Tensor，单位化顶点法向
    """
    # 获取三角面三个顶点
    v0 = v[f[:, 0]]  # (M, 3)
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    # 面法向量（不归一化）
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # (M, 3)

    # 为每个顶点累加相邻三角面的法向量
    vertex_normals = torch.zeros_like(v)
    for i in range(3):
        vertex_normals = vertex_normals.index_add(0, f[:, i], face_normals)

    # 单位化（防止除以 0）
    norm = vertex_normals.norm(dim=1, keepdim=True).clamp(min=eps)
    vertex_normals = vertex_normals / norm

    return vertex_normals

def normal_train(name, angle=None, train=None):
    if train is not None:
        v = train['v']
        f = train['f'].int()
        flame = train['flame']
        sub_index = train['sub_index']
        face_index = np.load(f'{root_path}/flame1/data/faces.npy')
        flame_dense = densify(name, flame.submesh([face_index])[0])
        
        
        faces = flame.submesh([sub_index])[0]
        glctx = dr.RasterizeCudaContext()
        v = torch.nn.functional.pad(v, (0, 1), 'constant', 1)
        
        r = 1.3
        fov = 40
        cams = [[math.pi, 0.], [math.pi-math.pi/12, 0.], [math.pi+math.pi/12, 0.], [math.pi, -math.pi/6]] if angle is None else [angle]
        yaws = [cam[0] for cam in cams]
        pitchs = [cam[1] for cam in cams]
        extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
        num = len(cams)
        extrinsics = torch.stack(extrinsics, dim=0).to(device).float()
        intrinsics = torch.stack(intrinsics, dim=0).to(device).float()
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        near, far = 1, 100.
        proj = intrinsics.new_zeros((num, 4, 4))
        proj[:, 0, 0] = 2 * fx
        proj[:, 1, 1] = 2 * fy
        proj[:, 0, 2] = 2 * cx - 1
        proj[:, 1, 2] = - 2 * cy + 1
        proj[:, 2, 2] = far / (far - near)
        proj[:, 2, 3] = near * far / (near - far)
        proj[:, 3, 2] = 1.
        verts_cam = v.expand(extrinsics.shape[0], -1, -1).bmm(extrinsics.transpose(-1, -2))
        verts_clip = verts_cam.bmm(proj.transpose(-1, -2))
        rast, _ = dr.rasterize(
            glctx,
            verts_clip,
            f,
            resolution=[2048, 2048]
        )
        attributes = train['attributes'][None].expand(num, -1, -1)
        
        attrimap, *_ = dr.interpolate(
            attributes.contiguous(),
            rast,
            f
        )
        attris = torch.ones_like(attributes).to(device)
        attrimap1, *_ = dr.interpolate(
            attris.contiguous(),
            rast,
            f
        )
        
        
        outputs = []
        outputs1 = []
        for i in range(attrimap.shape[0]):
            out_i, *_ = dr.antialias(
                attrimap[i:i+1],         # (1, H, W, C)
                rast[i:i+1],             # (1, H, W, 4)
                verts_clip[i:i+1],       # (1, V, 4)
                f                        # (F, 3)
            )
            out1_i, *_ = dr.antialias(
                attrimap1[i:i+1],         # (1, H, W, C)
                rast[i:i+1],             # (1, H, W, 4)
                verts_clip[i:i+1],       # (1, V, 4)
                f                        # (F, 3)
            )
            outputs1.append(out1_i)
            outputs.append(out_i)
        output = torch.stack(outputs, dim=0)
        output1 = torch.stack(outputs1, dim=0)

        
        def process(x):
            return (x+1)/2*255.
        for num, i in enumerate(output):
            cv2.imwrite(f'{OUTPUT_PATH}/{name}/mask/trellis_{num}.png', process(i).detach().cpu().numpy())
            cv2.imwrite(f'{OUTPUT_PATH}/{name}/mask/trellismask_{num}.png', process(output1[num]).detach().cpu().numpy())
        return output, output1
        

def render_tool(name, mesh, sub_index=None, angle=None, train=None):
    

    os.makedirs(f'{OUTPUT_PATH}/{name}/mask', exist_ok=True)
    if mesh is None:
        mesh = trimesh.load('')
    # face_index = np.load('/home/wzj/project/TRELLIS/wzj/data/faces.npy')
    if sub_index is not None:
        mesh = mesh.submesh([sub_index])[0]
    glctx = dr.RasterizeCudaContext()
    verts = torch.from_numpy(mesh.vertices).to(device).float()
    faces = torch.from_numpy(mesh.faces).to(device, torch.int32)
    verts = torch.nn.functional.pad(verts, (0, 1), 'constant', 1)
    
    r = 1.3
    fov = 40
    cams = [[math.pi, 0.]] if angle is None else [angle]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    num = len(cams)
    extrinsics = torch.stack(extrinsics, dim=0).to(device).float()
    intrinsics = torch.stack(intrinsics, dim=0).to(device).float()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    near, far = 1, 100.
    proj = intrinsics.new_zeros((num, 4, 4))
    proj[:, 0, 0] = 2 * fx
    proj[:, 1, 1] = 2 * fy
    proj[:, 0, 2] = 2 * cx - 1
    proj[:, 1, 2] = - 2 * cy + 1
    proj[:, 2, 2] = far / (far - near)
    proj[:, 2, 3] = near * far / (near - far)
    proj[:, 3, 2] = 1.
    verts_cam = verts.expand(extrinsics.shape[0], -1, -1).bmm(extrinsics.transpose(-1, -2))
    verts_clip = verts_cam.bmm(proj.transpose(-1, -2))
    rast, _ = dr.rasterize(
        glctx,
        verts_clip,
        faces,
        resolution=[2048, 2048]
    )


    # more
    normals = torch.from_numpy(mesh.vertex_normals).to(device, torch.float32)
    normal, _ = dr.interpolate(
        normals[None, ...].contiguous(),
        rast,
        faces
    )
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/mask/normal.png', (((normal + 1) / 2) * 255).squeeze().detach().cpu().numpy())
    

    return rast
    image = ((rast[..., -1].detach().cpu().numpy())!=0).astype(np.int32) * 255
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/mask/mask.png', image[0])

def display(*args):
    num = len(args)
    geometries = [o3d.geometry.PointCloud() for i in range(num)]
    for num, pcd in enumerate(geometries):
        pcd.points = o3d.utility.Vector3dVector(args[num])
        pcd.paint_uniform_color([random.random() for j in range(3)])  # green
    o3d.visualization.draw_geometries(geometries)

def correspond(name, source, v, f):
    '''
    v and f are all trellis
    '''
    face_index = np.load(f'{root_path}/flame1/data/smplx_faces.npy')
    flame_face = source.submesh([face_index])[0]
    flame_out = render_tool(name, source, face_index)
    trellis = render_tool(name, trimesh.Trimesh(vertices=v.detach().cpu().numpy(), faces=f.detach().cpu().numpy()))
    flame_mask = (flame_out[..., -1] != 0) # (1, 1024, 1024)
    trellis_mask = (trellis[..., -1] != 0) # (1, 1024, 1024)
    mask = flame_mask & trellis_mask
    
    index = mask.nonzero()

    
    # index = flame_mask.nonzero()
    u = flame_out[..., 0][index[..., 0], index[..., 1], index[..., 2]]
    v_ = flame_out[..., 1][index[..., 0], index[..., 1], index[..., 2]]
    w = 1 - u - v_
    uvw = torch.stack([u, v_, w], dim=-1) #(K, 3)
    flame_tris = (flame_out[..., -1][index[..., 0], index[..., 1], index[..., 2]] - 1).int()
    flame_points = flame_face.vertices[flame_face.faces[flame_tris.detach().cpu().numpy()]] # (K, 3, 3)
    flame_points = torch.from_numpy(flame_points).to(device, torch.float32)
    coords = torch.einsum('bi, bij->bj', uvw.to(flame_points.device, flame_points.dtype), flame_points)


    
    u = trellis[..., 0][index[..., 0], index[..., 1], index[..., 2]] # (1, 1024, 1024)
    v_ = trellis[..., 1][index[..., 0], index[..., 1], index[..., 2]] # (1, 1024, 1024)
    w = 1 - u - v_
    uvw = torch.stack([u, v_, w], dim=-1) #(K, 3)
    trellis_tris = (trellis[..., -1][index[..., 0], index[..., 1], index[..., 2]] - 1).int()
    trellis_points = v[f[trellis_tris]] # (K, 3, 3)
    # trellis_points = torch.from_numpy(trellis_points).to(device, torch.float32)
    
    trellis_coords = torch.einsum('bi, bij->bj', uvw.to(trellis_points.device, trellis_points.dtype), trellis_points)

    # display(trellis_coords.detach().cpu().numpy(), coords.detach().cpu().numpy())
    # exit()

    p_index = np.unique(flame_face.faces[flame_tris.detach().cpu().numpy()].flatten())
    output_p = torch.from_numpy(flame_face.vertices[p_index]).to(device, torch.float32)
    return ((coords - trellis_coords) ** 2).sum(dim=-1).mean(), output_p

    


    render_tool(name, mesh)

# def densify(name, mesh):
#     ms = pymeshlab.MeshSet()
#     ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
#     ms.apply_filter(
#         "meshing_surface_subdivision_ls3_loop",
#         iterations=3,
#         threshold=PercentageValue(0),
#     )
#     v = ms.current_mesh().vertex_matrix()
#     f = ms.current_mesh().face_matrix()
#     return trimesh.Trimesh(vertices=v, faces=f)

from utils.utils_geo import densify

def densify_vf(vertices, faces):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertices, faces))
    ms.apply_filter(
        "meshing_surface_subdivision_ls3_loop",
        iterations=3,
        threshold=Percentage(0),
    )
    v = ms.current_mesh().vertex_matrix()
    f = ms.current_mesh().face_matrix()
    return v, f

def coord(name, rast, mesh):
    mask = (rast[..., -1] != 0) # (1, 1024, 1024)
    index = mask.nonzero()

    
    # index = flame_mask.nonzero()
    u = rast[..., 0][index[..., 0], index[..., 1], index[..., 2]]
    v_ = rast[..., 1][index[..., 0], index[..., 1], index[..., 2]]
    w = 1 - u - v_
    uvw = torch.stack([u, v_, w], dim=-1) #(K, 3)
    tris = (rast[..., -1][index[..., 0], index[..., 1], index[..., 2]] - 1).int()
    points = mesh.vertices[mesh.faces[tris.detach().cpu().numpy()]] # (K, 3, 3)
    points = torch.from_numpy(points).to(device, torch.float32)
    coords = torch.einsum('bi, bij->bj', uvw.to(points.device, points.dtype), points)
    return coords


def laplace_loss(name, vertices, neighbors):
    '''
    vertices: (N, 3)
    neighbors: (N, num)
    '''
    coords = vertices[neighbors] # (N, num, 3)
    second_term = coords.mean(dim=1)
    loss_laplace = ((vertices - second_term) ** 2).sum(dim=-1).mean()
    return loss_laplace
    pass

def flame_keypoints(name, flame):
    face_index = np.load(f'{root_path}/flame1/data/smplx_faces.npy')
    flame_dense = densify(name, flame.submesh([face_index])[0])
    rast_front = render_tool(name, flame_dense)
    rast_left = render_tool(name, flame_dense, angle=[math.pi-math.pi/12, 0.])
    rast_right = render_tool(name, flame_dense, angle=[math.pi+math.pi/12, 0.])
    rast_bottom = render_tool(name, flame_dense, angle=[math.pi, -math.pi/6])
    coord_front = coord(name, rast_front, flame_dense)
    coord_left = coord(name, rast_left, flame_dense)
    coord_right = coord(name, rast_right, flame_dense)
    coord_bottom = coord(name, rast_bottom, flame_dense)
    coords = torch.cat([coord_front, coord_left, coord_right, coord_bottom], dim=0)
    coords = coords.unique(dim=0)

    # display(coords.detach().cpu().numpy())
    # exit()10000
    
    return coords


    flame_mask = (rast[..., -1] != 0) # (1, 1024, 1024)
    mask = flame_mask
    
    index = mask.nonzero()

    
    # index = flame_mask.nonzero()
    u = rast[..., 0][index[..., 0], index[..., 1], index[..., 2]]
    v_ = rast[..., 1][index[..., 0], index[..., 1], index[..., 2]]
    w = 1 - u - v_
    uvw = torch.stack([u, v_, w], dim=-1) #(K, 3)
    flame_tris = (rast[..., -1][index[..., 0], index[..., 1], index[..., 2]] - 1).int()
    flame_points = flame_dense.vertices[flame_dense.faces[flame_tris.detach().cpu().numpy()]] # (K, 3, 3)
    flame_points = torch.from_numpy(flame_points).to(device, torch.float32)
    coords = torch.einsum('bi, bij->bj', uvw.to(flame_points.device, flame_points.dtype), flame_points)

    return coords

def from_mesh_to_coords(name, mesh):
    v = np.clip(mesh.vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    # coords_refine = sample_fromface(name, trimesh.Trimesh(vertices=v, faces=mesh.faces))
    m = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(v),
        triangles=o3d.utility.Vector3iVector(mesh.faces),
    )
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(m, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    coords = torch.from_numpy(vertices).to(device, torch.int32)
    # coords = torch.cat([coords, coords_refine], dim=0)
    coords = coords.unique(dim=0)
    print(coords.min())
    coords = torch.nn.functional.pad(coords, [1, 0], 'constant', 0)
    return coords

def backgene(name, image_path, mesh=None, obj_name='new', smplx=None):
    # global pipeline
    # os.system('ln -sf finetune/slat_flow_img_dit_L_64l8p2_fp16.safetensors TRELLIS-image-large/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors')
    # pipeline = TrellisImageTo3DPipeline.from_pretrained(f"{root_path}/TRELLIS-image-large")
    # pipeline.cuda()
    face_mask = np.load('params/ineedyou.npy')
    local_temp = smplx.submesh([face_mask])[0]
    smplx_local_v, smplx_local_f = densify(local_temp.vertices, local_temp.faces)
    smplx_local = trimesh.Trimesh(vertices=smplx_local_v, faces=smplx_local_f)

    if mesh is None:
        mesh = trimesh.load(f'{OUTPUT_PATH}/{name}/ICP/trellis_icp_500.obj')
    os.makedirs(f'{OUTPUT_PATH}/{name}/objects', exist_ok=True)
    coords = from_mesh_to_coords(name, mesh)
    coords_smplx = from_mesh_to_coords(name, smplx_local)
    coords = torch.cat([coords, coords_smplx], dim=0)
    coords = coords.unique(dim=0)
    image = Image.open(image_path)
    seed = random.randint(0, MAX_SEED)
    cprint('start generating')
    outputs, slat = pipeline.run(
        image,
        seed=seed,
        preprocess_image=True,
        sparse_structure_sampler_params={
            "steps": 25,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 25,
            "cfg_strength": 5,
        },
        coords=coords,
        save=True,
        name=name,
        obj_name=obj_name,
        path=cfg['output_dir']
    )
    v = outputs['mesh'][0].vertices
    f = outputs['mesh'][0].faces
    # outputs['gaussian'][0].save_ply(f'{OUTPUT_PATH}/{name}/objects/{obj_name}.ply')
    m = trimesh.Trimesh(vertices=v.detach().cpu().numpy(), faces=f.detach().cpu().numpy())
    if not os.path.exists(f'{OUTPUT_PATH}/{name}/slats'):
        os.makedirs(f'{OUTPUT_PATH}/{name}/slats', exist_ok=True)
    torch.save(slat.coords, f'{OUTPUT_PATH}/{name}/slats/coords_coarse_back.pt')
    torch.save(slat.feats, f'{OUTPUT_PATH}/{name}/slats/feats_coarse_back.pt')
    
    output = outer(m, name, obj_name)
    print(coords.shape)

def sample_fromface(name, mesh=None, face=None):
    if mesh is None:
        mesh = trimesh.load(f'{root_path}/exp/{name}/ICP/trellis_icp_500.obj')
    if face is None:
        face = trimesh.load(f'{root_path}/exp/{name}/objects/flame.obj')
        face_index = np.load(f'{root_path}/flame1/data/region.npy')
        face = face.submesh([face_index])[0]
    trellis = render_tool(name, mesh)
    face = render_tool(name, face)
    trellis_mask = trellis[..., -1]
    face_mask = face[..., -1]

    mask = (trellis_mask != 0) & (face_mask != 0)
    index = (trellis_mask[mask] - 1).int().unique().detach().cpu().numpy()
    p_index = np.unique(mesh.faces[index].flatten())
    v = mesh.vertices[p_index] # (K, 3)
    normals = mesh.vertex_normals[p_index]
    normal_points = v + normals * np.random.uniform(-0.001, 0.001, size=(len(v), 1))
    noise = np.random.normal(scale=0.001, size=v.shape)
    noisy_points = np.concatenate([v + noise, v, v - noise, normal_points], axis=0)

    coords = np.floor((noisy_points + 0.5) * 64).astype(np.int32)
    print(coords)
    return torch.from_numpy(coords).to(device)

def sample_from_surface(vertices, faces, nums):
    assert nums < len(faces), 'you need too many samples'
    # seed = np.random.randint(0, MAX_SEED)
    # np.random.seed(seed)
    face_index = faces[random.sample(list(range(len(faces))), nums)] #(nums, 3)
    r1 = torch.rand(nums)
    r2 = torch.rand(nums)
    mask = (r1 + r2) > 1
    r1[mask] = 1 - r1[mask]
    r2[mask] = 1 - r2[mask]
    u = 1 - r1 - r2
    v = r1
    w = r2
    uvw = torch.stack([u, v, w], axis=-1).to(vertices.device).float()
    selected_face_points = vertices[face_index] # (num, 3, 3)
    selected_points = torch.einsum('bi, bij->bj', uvw, selected_face_points)
    return selected_points, face_index, uvw


def mynewicp(name, source, target, face_mask='params/ineedyou.npy', lr=1e-3, iters=501):
    os.makedirs(f'{OUTPUT_PATH}/{name}/ICP', exist_ok=True)

    # source = trimesh.load('output/test_pink_1/objects/sample_outer_filtered.obj')
    # target = trimesh.load('output/test_pink_1/objects/smplx.obj')
    face_mask = np.load(face_mask)
    temp = target.submesh([face_mask])[0]
    v, f = densify(temp.vertices, temp.faces)
    bvh_v = torch.from_numpy(v).to(device).float()
    bvh_f = torch.from_numpy(f).to(device).int()
    BVH = cubvh.cuBVH(bvh_v, bvh_f)

    delta = torch.zeros(*source.vertices.shape).to(device).float().requires_grad_(True)
    source_v = torch.from_numpy(source.vertices).to(device).float()
    source_f = torch.from_numpy(source.faces).to(device).int()
    neighbors, _ = one_ring(source, 4)
    quaternion = torch.zeros(source_v.shape[0], 4).to(device)
    # exit()
    
    quaternion[:, 0] = 1
    quaternion = torch.nn.Parameter(quaternion.detach())
    optimizer = Adam([delta, quaternion], lr=lr)

    def ARAP_Loss(x):
        '''
        x: (N, 3)
        neighbors: (N, num)
        weights: (N, num)
        '''
        R = quaternion_to_matrix(quaternion)
        neigh_deform = delta[neighbors]
        neigh_coords = x[neighbors] # (N, num, 3)
        first_term = (x[:, None, :] - neigh_coords).bmm(R.transpose(-1, -2).to(x.device, x.dtype)) # (N, num, 3)
        second_term = (x + delta)[:, None, :] - (neigh_coords + neigh_deform) # (N, num, 3)

        output = ((first_term - second_term) ** 2).sum(dim=-1) # (N, num)
        loss = (output).mean()
        return loss
    
    
    for iter in range(iters):
        # print(iter)
        optimizer.zero_grad()
        selected_pts, selected_faces, selected_uvw = sample_from_surface(source_v+delta, source_f, 50000)
        # destination = bvh_v
        _, idx, _ = knn_points(bvh_v.unsqueeze(0), selected_pts.unsqueeze(0))
        idx = idx.squeeze().unique() #(num)
        selected_pts = selected_pts[idx]
        # selected_pts = selected_pts.unique(dim=0)
        
        distances, face_id, uvw = BVH.unsigned_distance(selected_pts, return_uvw=True)
        # exit()
        # mask = (distances < 0.01)
        # selected_pts = selected_pts[mask]
        
        # face_id = face_id[mask]
        # uvw = uvw[mask]
        # print(face_id)
        # print(type(face_id))
        # print(face_id.shape)
        # print(bvh_f)
        face_id = face_id.to(device).to(torch.int32)
        destination = bvh_v[bvh_f[face_id]]
        destination = torch.einsum('bi, bij->bj', uvw, destination)
        
        loss_deform = torch.nn.functional.huber_loss(selected_pts, destination)
        loss_laplace = laplace_loss(None, source_v+delta, neighbors)
        # loss_ARAP = ARAP_Loss(source_v)

        loss = 2 * loss_deform + 10 * ARAP_Loss(source_v+delta) + 10 * loss_laplace
        loss.backward()
        optimizer.step()
        # exit()
        if iter % 10 == 0:
            print(laplace_loss(None, source_v, neighbors))
            print(f'epoch: {iter}, loss: {loss.item()}, loss_deform: {loss_deform.item()}, loss_laplace: {loss_laplace.item()}')
        
        if iter % 100 == 0:
            with torch.no_grad():
                trimesh.Trimesh(vertices=(source_v+delta).cpu().numpy(), faces=source_f.cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/ICP/trellis_icp_{iter}.obj')

def myicp(name, source, target, face_mask='flame1/data/smplx_faces_no_eye.npy', lr=1e-3, iters=301):
    os.makedirs(f'{OUTPUT_PATH}/{name}/ICP', exist_ok=True)
    '''
    sample from trellis to optimization all points
    whether need normal loss, or only eye region normal loss
    laplace loss
    '''
    face_mask = np.load(face_mask)
    temp = target.submesh([face_mask])[0]

    v, f = densify(temp.vertices, temp.faces)
    bvh_v = torch.from_numpy(v).to(device).float()
    bvh_f = torch.from_numpy(f).to(device).int()
    # smplx_local = densify(None, target.submesh([face_mask])[0])
    # bvh_v = torch.from_numpy(smplx_local.vertices).to(device).float()
    # bvh_f = torch.from_numpy(smplx_local.faces).to(device).int()
    BVH = cubvh.cuBVH(bvh_v, bvh_f)

    delta = torch.zeros(*source.vertices.shape).to(device).float().requires_grad_(True)
    source_v = torch.from_numpy(source.vertices).to(device).float()
    source_f = torch.from_numpy(source.faces).to(device).int()
    neighbors, _ = one_ring(source, 4)
    quaternion = torch.zeros(source_v.shape[0], 4).to(device)
    
    quaternion[:, 0] = 1
    quaternion = torch.nn.Parameter(quaternion.detach())
    optimizer = Adam([delta, quaternion], lr=lr)

    def ARAP_Loss(x):
        '''
        x: (N, 3)
        neighbors: (N, num)
        weights: (N, num)
        '''
        R = quaternion_to_matrix(quaternion)
        neigh_deform = delta[neighbors]
        neigh_coords = x[neighbors] # (N, num, 3)
        first_term = (x[:, None, :] - neigh_coords).bmm(R.transpose(-1, -2).to(x.device, x.dtype)) # (N, num, 3)
        second_term = (x + delta)[:, None, :] - (neigh_coords + neigh_deform) # (N, num, 3)

        output = ((first_term - second_term) ** 2).sum(dim=-1) # (N, num)
        loss = (output).sum()
        return loss
    for iter in range(iters):
        optimizer.zero_grad()
        selected_pts, selected_faces, selected_uvw = sample_from_surface(source_v+delta, source_f, 50000)
        # destination = bvh_v
        _, idx, _ = knn_points(bvh_v.unsqueeze(0), selected_pts.unsqueeze(0))
        idx = idx.squeeze().unique() #(num)
        selected_pts = selected_pts[idx]
        # selected_pts = selected_pts.unique(dim=0)
        
        distances, face_id, uvw = BVH.unsigned_distance(selected_pts, return_uvw=True)
        # mask = (distances < 0.01)
        # selected_pts = selected_pts[mask]
        
        # face_id = face_id[mask]
        # uvw = uvw[mask]

        destination = bvh_v[bvh_f[face_id]]
        destination = torch.einsum('bi, bij->bj', uvw, destination)
        
        loss_deform = torch.nn.functional.huber_loss(selected_pts, destination)
        loss_laplace = laplace_loss(None, source_v+delta, neighbors)
        # loss_ARAP = ARAP_Loss(source_v)

        loss = loss_deform + 1 * loss_laplace
        loss.backward()
        optimizer.step()

        if iter % 10 == 0:
            print(laplace_loss(None, source_v, neighbors))
            print(f'epoch: {iter}, loss: {loss.item()}, loss_deform: {loss_deform.item()}, loss_laplace: {loss_laplace.item()}')
        
        if iter % 100 == 0:
            with torch.no_grad():
                trimesh.Trimesh(vertices=(source_v+delta).cpu().numpy(), faces=source_f.cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/ICP/trellis_icp_{iter}.obj')

def icp(name, trellis, flame, back=False):
    path = 'ICP' if not back else 'ICP_new'
    os.makedirs(f'{OUTPUT_PATH}/{name}/{path}', exist_ok=True)
    input = torch.from_numpy(trellis.vertices).to(device, torch.float32)
    face = torch.from_numpy(trellis.faces).to(device)
    face_index = np.load(f'{root_path}/flame1/data/faces.npy')
    flame_face = flame.submesh([face_index])[0]
    with open(f'{root_path}/models/flame_static_embedding_68.pkl', 'rb') as f:
        lmk_face_idx, lmk_b_coords = pickle.load(f, encoding='latin1').values()
        lmk_face_idx = np.array(lmk_face_idx)
        lmk_b_coords = np.array(lmk_b_coords)
    lmk_v = flame.vertices[flame.faces[lmk_face_idx.astype(np.int32)]] # (98, 3, 3)
    lmk_v = np.einsum('bij, bi->bj', lmk_v.astype(np.float32), lmk_b_coords.astype(np.float32))
    lmk_v = np.concatenate([lmk_v[27:31], lmk_v[36:]], axis=0) # only interior points
    lmk_v = torch.from_numpy(lmk_v).to(device, torch.float)

    # use nvdiffrast to find correspondence
    flame_mask = flame_sample(name, flame) # (B, 1024, 1024)
    flame_gt = flame_keypoints(name, flame)

    
    # parameters
    delta = torch.nn.Parameter(torch.zeros(trellis.vertices.shape[0], 3).to(device))
    quaternion = torch.zeros(trellis.vertices.shape[0], 4).to(device)
    quaternion[:, 0] = 1
    quaternion = torch.nn.Parameter(quaternion.detach())

    neighbors, weights = one_ring(trellis, 4)
    optimizer = torch.optim.Adam([delta, quaternion], lr=1e-3)

    def ARAP_Loss(x):
        '''
        x: (N, 3)
        neighbors: (N, num)
        weights: (N, num)
        '''
        R = quaternion_to_matrix(quaternion)
        neigh_deform = delta[neighbors]
        neigh_coords = x[neighbors] # (N, num, 3)
        first_term = (x[:, None, :] - neigh_coords).bmm(R.transpose(-1, -2).to(x.device, x.dtype)) # (N, num, 3)
        second_term = (x + delta)[:, None, :] - (neigh_coords + neigh_deform) # (N, num, 3)

        output = ((first_term - second_term) ** 2).sum(dim=-1) # (N, num)
        loss = (output).sum()
        return loss



    for epoch in range(1000):
        if epoch == 501 and path == 'ICP_new':
            np.save(f'{OUTPUT_PATH}/{name}/{path}/delta.npy', delta.detach().cpu().numpy())
            break
        optimizer.zero_grad()
        if epoch < 100 or epoch % 20 == 0:
            tree = cKDTree((input+delta).detach().cpu().numpy())
            _, index =  tree.query(flame_gt.detach().cpu().numpy())
        input_keypoints = input[index]
        deform_gt = flame_gt - input_keypoints
        loss_laplace = laplace_loss(name, input+delta, neighbors)
        loss_deformed = ((delta[index] - deform_gt) ** 2).sum(dim=-1).mean()
        # print(ARAP_Loss(input).item())
        loss_ARAP = ARAP_Loss(input)
        loss_render, flame_p = correspond(name, flame, input+delta, face)
        loss = 100 * loss_render + 10000 * loss_deformed + loss_ARAP + 1000 * loss_laplace
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'epcoh:{epoch}, loss:{loss.item()}, loss_ARAP:{loss_ARAP.item()}, loss_render:{loss_render.item()}, loss_deformed: {loss_deformed}, loss_laplace: {loss_laplace}')
            trimesh.Trimesh(vertices=(input+delta).detach().cpu().numpy(), faces=trellis.faces).export(f'{OUTPUT_PATH}/{name}/{path}/trellis_icp_{epoch}.obj')


def icpsmplx(name, trellis, smplx, back=False):
    path = 'ICP' if not back else 'ICP_new'
    os.makedirs(f'{OUTPUT_PATH}/{name}/{path}', exist_ok=True)
    input = torch.from_numpy(trellis.vertices).to(device, torch.float32)
    face = torch.from_numpy(trellis.faces).to(device)
    face_index = np.load(f'{root_path}/flame1/data/smplx_faces.npy')
    smplx_face = smplx.submesh([face_index])[0]
    # with open(f'{root_path}/models/flame_static_embedding_68.pkl', 'rb') as f:
    #     lmk_face_idx, lmk_b_coords = pickle.load(f, encoding='latin1').values()
    #     lmk_face_idx = np.array(lmk_face_idx)
    #     lmk_b_coords = np.array(lmk_b_coords)
    # lmk_v = flame.vertices[flame.faces[lmk_face_idx.astype(np.int32)]] # (98, 3, 3)
    # lmk_v = np.einsum('bij, bi->bj', lmk_v.astype(np.float32), lmk_b_coords.astype(np.float32))
    # lmk_v = np.concatenate([lmk_v[27:31], lmk_v[36:]], axis=0) # only interior points
    # lmk_v = torch.from_numpy(lmk_v).to(device, torch.float)

    # use nvdiffrast to find correspondence
    # flame_mask = flame_sample(name, flame) # (B, 1024, 1024)
    smplx_gt = flame_keypoints(name, smplx)

    
    # parameters
    delta = torch.nn.Parameter(torch.zeros(trellis.vertices.shape[0], 3).to(device))
    quaternion = torch.zeros(trellis.vertices.shape[0], 4).to(device)
    quaternion[:, 0] = 1
    quaternion = torch.nn.Parameter(quaternion.detach())

    neighbors, weights = one_ring(trellis, 4)
    optimizer = torch.optim.Adam([delta, quaternion], lr=1e-3)

    def ARAP_Loss(x):
        '''
        x: (N, 3)
        neighbors: (N, num)
        weights: (N, num)
        '''
        R = quaternion_to_matrix(quaternion)
        neigh_deform = delta[neighbors]
        neigh_coords = x[neighbors] # (N, num, 3)
        first_term = (x[:, None, :] - neigh_coords).bmm(R.transpose(-1, -2).to(x.device, x.dtype)) # (N, num, 3)
        second_term = (x + delta)[:, None, :] - (neigh_coords + neigh_deform) # (N, num, 3)

        output = ((first_term - second_term) ** 2).sum(dim=-1) # (N, num)
        loss = (output).sum()
        return loss



    for epoch in range(301):
        if epoch == 501 and path == 'ICP_new':
            np.save(f'{OUTPUT_PATH}/{name}/{path}/delta.npy', delta.detach().cpu().numpy())
            break
        optimizer.zero_grad()
        # if epoch < 100 or epoch % 20 == 0:
        if epoch % 20 == 0:
            tree = cKDTree((input+delta).detach().cpu().numpy())
            _, index =  tree.query(smplx_gt.detach().cpu().numpy())
        input_keypoints = input[index]
        deform_gt = smplx_gt - input_keypoints
        loss_laplace = laplace_loss(name, input+delta, neighbors)
        loss_deformed = ((delta[index] - deform_gt) ** 2).sum(dim=-1).mean()
        # print(ARAP_Loss(input).item())
        loss_ARAP = ARAP_Loss(input)
        loss_render, smplx_p = correspond(name, smplx, input+delta, face)
        loss = 100 * loss_render + 10000 * loss_deformed + loss_ARAP + 1000 * loss_laplace
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'epcoh:{epoch}, loss:{loss.item()}, loss_ARAP:{loss_ARAP.item()}, loss_render:{loss_render.item()}, loss_deformed: {loss_deformed}, loss_laplace: {loss_laplace}')
            trimesh.Trimesh(vertices=(input+delta).detach().cpu().numpy(), faces=trellis.faces).export(f'{OUTPUT_PATH}/{name}/{path}/trellis_icp_{epoch}.obj')


def test_model(name=None):
    coords = torch.randint(0, 255, (100000, 3))
    coords = torch.nn.functional.pad(coords, [1, 0], 'constant', 0).to(device)
    feats = torch.randn(100000, 101).to(device)
    feats.requires_grad = True
    noise = sp.SparseTensor(
        coords=coords.int(),
        feats=feats
    )
    start = time.time()
    
    
    model = SparseFeatures2Mesh(res=256, use_color=True)
    mesh = model(noise)
    loss = (mesh.vertices).norm()
    loss.backward()
    print(feats.grad)
    exit()
    
    trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.detach().cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/objects/icp_dense.obj')

# def render_normal(mesh_path, image_size=(1024, 1024), save_path=None):
    

def render_white_model(mesh_path, image_size=(1024, 1024), show=False, save_path=None):
    kt = np.load('/home/wzj/project/TRELLIS/src/test/params/kt.npz')
    k, t = list(kt.values())
    srt = np.load('src/test/params/trans.npz')
    s, R, T = list(srt.values())
    # 加载 mesh（支持 .obj, .ply 等）
    mesh = trimesh.load(mesh_path)
    mesh.vertices = mesh.vertices * k + t
    R = Quaternion(R)
    # mesh.vertices = mesh.vertices * s[0]
    mesh.vertices = s * mesh.vertices @ R[0].T + T
    # mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    
    # 设置纯白材质
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.8, 0.8, 0.8, 1.0],  # RGBA 全白
        metallicFactor=0.0,
        roughnessFactor=1.0
    )

    # 转换为 pyrender mesh
    render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    # 创建场景
    scene = pyrender.Scene(bg_color=[0., 0., 0., 0.], ambient_light=[0.3, 0.3, 0.3])
    scene.add(render_mesh)

    # 添加摄像机
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -1.3],
        [0.0, 1.0, 0.0, 0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    # 添加灯光
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    # 渲染
    r = pyrender.OffscreenRenderer(*image_size)
    color, _ = r.render(scene)

    # 显示
    if show:
        plt.imshow(color)
        plt.axis('off')
        plt.show()

    # 保存
    if save_path:
        from PIL import Image
        Image.fromarray(color).save(save_path)

    r.delete()
    return color

def voxel_move(name):
    voxel = f'{OUTPUT_PATH}/{name}/slats/coords_0_sample.pt'
    coords = torch.load(voxel)
    coords = (coords[..., 1:] + 0.5) / 256 - 0.5
    deformation = np.load(f'{OUTPUT_PATH}/{name}/ICP_new/delta.npy')
    deformation = torch.from_numpy(deformation).to(device, torch.float32)
    mesh = trimesh.load(f'{OUTPUT_PATH}/{name}/objects/new_outer_filtered.obj')
    v = torch.from_numpy(mesh.vertices).to(device, torch.float32)
    f = torch.from_numpy(mesh.faces).to(device, torch.int32)
    BVH = cubvh.cuBVH(v, f)
    distances, face_id, uvw = BVH.unsigned_distance(coords, return_uvw=True)
    face_id, uvw = face_id.squeeze(), uvw.squeeze()
    deform = deformation[f[face_id]] #(N, i, j)
    deform_detailed = torch.einsum('bij, bi->bj', deform, uvw)
    output = torch.floor((coords + deform_detailed + 0.5)*256).int()
    feats = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_0_sample.pt').to(device)
    output_coords = torch.nn.functional.pad(output.to(device), [1, 0], 'constant', 0)
    feats.requires_grad = True
    optimizer = Adam([feats], lr=1e-3)
    model = SparseFeatures2Mesh(res=256, use_color=True)
    names = ['front', 'left', 'right', 'bottom']
    masks = [torch.from_numpy(cv2.imread(f'{OUTPUT_PATH}/{name}/mask/flame_{n}.png')).to(device, torch.float32) / 255 for n in names]
    gts = [torch.from_numpy(cv2.imread(f'{OUTPUT_PATH}/{name}/mask/flamenormal_{n}.png')).to(device, torch.float32) / 255 for n in names]
    gts = torch.stack(gts, dim=0)
    masks = torch.stack(masks, dim=0)
    for epoch in range(1000):

        noise = sp.SparseTensor(
            coords=output_coords.int(),
            feats=feats
        )
        mesh = model(noise)
        train = {
            'v': mesh.vertices.squeeze(),
            'f': mesh.faces.squeeze(),
            'flame': trimesh.load(f'{OUTPUT_PATH}/{name}/objects/flame.obj'),
            'sub_index': np.load(f'{root_path}/flame1/data/faces.npy'),
            'attributes': compute_vertex_normals(mesh.vertices.squeeze(), mesh.faces.squeeze())
        }
        normal_map, mask = normal_train(name, train=train)
        loss = (normal_map - gts) * masks
        loss_mask = ((mask - masks)**2).mean()
        loss = (loss ** 2).mean() + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
    trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.detach().cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/objects/icp_dense.obj')

def render_gts(name):
    flame = trimesh.load(f'{OUTPUT_PATH}/{name}/objects/flame.obj')
    faces = np.load(f'{root_path}/flame1/data/faces.npy')
    flame_dense = densify(name, flame.submesh([faces])[0])
    attributes = torch.from_numpy(flame_dense.vertex_normals).to(device, torch.float32)
    rast_front = render_tool(name, flame_dense)
    rast_left = render_tool(name, flame_dense, angle=[math.pi-math.pi/12, 0.])
    rast_right = render_tool(name, flame_dense, angle=[math.pi+math.pi/12, 0.])
    rast_bottom = render_tool(name, flame_dense, angle=[math.pi, -math.pi/6])
    mask_front = (rast_front.squeeze()[..., -1] != 0) # (2048, 2048)
    mask_left = (rast_left.squeeze()[..., -1] != 0)
    mask_right = (rast_right.squeeze()[..., -1] != 0)
    mask_bottom = (rast_bottom.squeeze()[..., -1] != 0)
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/mask/flame_front.png', mask_front.to(torch.uint8).cpu().numpy()*255)
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/mask/flame_left.png', mask_left.to(torch.uint8).cpu().numpy()*255)
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/mask/flame_right.png', mask_right.to(torch.uint8).cpu().numpy()*255)
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/mask/flame_bottom.png', mask_bottom.to(torch.uint8).cpu().numpy()*255)
    faces = torch.from_numpy(flame_dense.faces).to(device, torch.int32)
    normal_front, *_ = dr.interpolate(
        attributes.contiguous(),
        rast_front,
        faces
    )
    normal_left, *_ = dr.interpolate(
        attributes.contiguous(),
        rast_left,
        faces
    )
    normal_right, *_ = dr.interpolate(
        attributes.contiguous(),
        rast_right,
        faces
    )
    normal_bottom, *_ = dr.interpolate(
        attributes.contiguous(),
        rast_bottom,
        faces
    )
    def process(x):
        return (x+1)/2*255.
    normal_front = process(normal_front)
    normal_left = process(normal_left)
    normal_right = process(normal_right)
    normal_bottom = process(normal_bottom)
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/mask/flamenormal_front.png', normal_front.squeeze().detach().cpu().numpy())
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/mask/flamenormal_left.png', normal_left.squeeze().detach().cpu().numpy())
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/mask/flamenormal_right.png', normal_right.squeeze().detach().cpu().numpy())
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/mask/flamenormal_bottom.png', normal_bottom.squeeze().detach().cpu().numpy())

def interpolate(name, rast, attributes, faces):
    attributes = torch.from_numpy(attributes).to(device, torch.float32)
    amax, amin = attributes.max(), attributes.min()
    faces = torch.from_numpy(faces).to(device, torch.int32)
    glctx = dr.RasterizeCudaContext()

    deform_image, *_ = dr.interpolate(
        attributes,
        rast,
        faces
    )
    deform_image = (deform_image - amin) / (amax - amin)
    cv2.imwrite(f'{OUTPUT_PATH}/{name}/images/deform.png', deform_image.squeeze().detach().cpu().numpy() * 255)
    

def submesh(mesh, face_indices):
    selected_faces = mesh.faces[face_indices]
    unique_vertex_indices = np.unique(selected_faces)
    old_idx = unique_vertex_indices
    # 建立原始索引到新索引的映射
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}

    # 映射原始面顶点到新索引
    new_faces = np.array([[index_map[vid] for vid in face] for face in selected_faces])

    # 提取新的顶点列表
    new_vertices = mesh.vertices[unique_vertex_indices]

    # 构造新的 trimesh 对象
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

    return old_idx, new_faces

def motion(name, trellis, source, target, old_idx, new_faces):
    s, R, T = list(np.load(f'{OUTPUT_PATH}/{name}/params/trans.npz').values())
    shapes, exprs, jaws = [], [], []
    params = torch.load('/home/wzj/project/TRELLIS/dataset/emotional_vids_confident-bearded-man-posing-on-white-background-SBV-346838037-4K.mp4/body_track/smplx_track.pth')
    shapes = params['shape']
    exprs = params['expr']
    jaws = params['jaw_pose']
    # for num in range(112):
    #     shapes.append(np.load(f'{OUTPUT_PATH}/{name}/params/{num}.npy')[:, :300])
    #     exprs.append(np.load(f'{OUTPUT_PATH}/{name}/params/{num}.npy')[:, 300:400])
    #     jaws.append(np.load(f'{OUTPUT_PATH}/{name}/params/{num}.npy')[:, 400:])
    shapes, exprs, jaws = torch.from_numpy(shapes).to(device), torch.from_numpy(exprs).to(device), torch.from_numpy(jaws).to(device)
    # points, face_idx = trimesh.sample.sample_surface(trellis, count=150000)
    # normals = trellis.face_normals[face_idx]
    # noise = np.random.normal(scale=0.001, size=points.shape)
    # noisy_points = points + normals * np.random.uniform(-0.001, 0.001, size=(len(points), 1))
    # noisy_points = np.concatenate([points, noisy_points, points+noise], axis=0)
    # noisy_points = torch.from_numpy(noisy_points).to(device)
    # noisy_points = noisy_points.clip(-0.5+1e-6, 0.5-1e-6)

    # TODO
    coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_0_sample.pt').to(device)
    feats = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_0_sample.pt').to(device)
    indices = coords[..., 1:]
    feats_records = -1000 * torch.ones(256, 256, 256, 101).to(device)
    feats_records[indices[..., 0], indices[..., 1], indices[..., 2]] = feats

    f = new_faces
    
    source_v = source.vertices[old_idx]
    
    # debug 
    noisy_points = (indices + 0.5) / 256 - 0.5

    cprint('begin motion')

    # rast = render_tool('video', m)
    # deform = deformation[old_idx]
    # interpolate('video', rast, deformation, f)
    # exit()
    BVH = cubvh.cuBVH(torch.from_numpy(source_v).to(device), torch.from_numpy(new_faces).to(device))
    distances, face_id, uvw = BVH.unsigned_distance(noisy_points, return_uvw=True)

    face_region = (distances < 0.01)
    move = noisy_points[face_region]
    still = noisy_points[~face_region]

    my_coords = torch.floor((noisy_points+0.5)*256).int()
    faces = f[face_id.cpu().numpy()]
    s, R, T = torch.from_numpy(s).to(device), torch.from_numpy(R).to(device), torch.from_numpy(T).to(device) 
    k, t = list(np.load(f'{OUTPUT_PATH}/{name}/params/kt.npz').values())


    move_coords = torch.floor((move+0.5)*256).int()
    still_coords = torch.floor((still+0.5)*256).int()


    move_feats = feats_records[move_coords[..., 0], move_coords[..., 1], move_coords[..., 2]]
    move_mask = (move_feats>-900).all(dim=-1)
    still_feats = feats_records[still_coords[..., 0], still_coords[..., 1], still_coords[..., 2]]
    still_mask = (still_feats>-900).all(dim=-1)

    move = move[move_mask]
    move_coords = move_coords[move_mask]
    move_feats = move_feats[move_mask]
    still = still[still_mask]
    still_coords = still_coords[still_mask]
    still_feats = still_feats[still_mask]

    move_face_id = face_id[face_region][move_mask]
    move_uvw = uvw[face_region][move_mask]

    faces = f[move_face_id.cpu().numpy()]
    s, R, T = torch.from_numpy(s).to(device), torch.from_numpy(R).to(device), torch.from_numpy(T).to(device) 
    k, t = list(np.load(f'{OUTPUT_PATH}/{name}/params/kt.npz').values())

    for num in range(112):
        shape = shapes[num]
        expr = exprs[num]
        flame_v = flameNet.forward_geo(shape, expr, jaw_pose_params=jaws[num]).squeeze().detach().cpu().numpy()
        flame_v = flame_v * k + t
        flame_v = torch.from_numpy(flame_v).to(device)
        flame_v = s * flame_v @ quaternion_to_matrix(R)[0].transpose(-2, -1) + T
        flame_v = flame_v.detach().cpu().numpy()
        v = flame_v[old_idx]
        deformation = v - source_v
        deformation = deformation[faces]
        deformation = np.einsum('bij, bi->bj', deformation, move_uvw.cpu().numpy())

        position = move.cpu().numpy() + deformation
        position = np.clip(position, -0.5+1e-6, 0.5-1e-6)
        output = np.floor((position + 0.5) * 256)
        output = np.concatenate([still_coords.cpu().numpy(), output], axis=0)
        output_feats = torch.cat([still_feats, move_feats], dim=0)
        output_coords = torch.nn.functional.pad(torch.from_numpy(output).to(device), [1, 0], 'constant', 0)

        output_coords, index = output_coords.unique(return_inverse=True, dim=0)
        temp_feats = torch.zeros(output_coords.shape[0], 101).to(device)
        temp_feats[index] = output_feats
        noise = sp.SparseTensor(
            coords=output_coords.int(),
            feats=temp_feats
        )
        start = time.time()
        
        # output_trellis = pipeline.decode_slat(noise, ['gaussian', 'mesh'])
        # end = time.time()
        # print(end-start)
        # output_trellis['gaussian'][0].save_ply(f'{OUTPUT_PATH}/{name}/objects/motion.ply')
        # v = output_trellis['mesh'][0].vertices
        # f = output_trellis['mesh'][0].faces
        # trimesh.Trimesh(vertices=v.detach().cpu().numpy(), faces=f.detach().cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/objects/motion.obj')
        
        model = SparseFeatures2Mesh(res=256, use_color=True)
        mesh = model(noise)
        # output = pipeline.decode_slat(noise, ['gaussian', 'mesh'])
        # output['gaussian'][0].save_ply(f'{OUTPUT_PATH}/{name}/objects/samplegg.ply')
        # v = output['mesh'][0].vertices
        # f = output['mesh'][0].faces
        trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.detach().cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/deform/motion_dense_{num}.obj')

def motion_save(name, source, new_faces):
    '''
    because save motion for every sample is a waste of space, so only save face_id and uvw for every id
    '''
    # s, R, T = list(np.load(f'{OUTPUT_PATH}/{name}/params/trans.npz').values())
    # shapes, exprs, jaws = [], [], []
    # params = torch.load(f'dataset/{name}/body_track/smplx_track.pth')
    # shapes = params['shape']
    # exprs = params['expr']
    # jaws = params['jaw_pose']
    # _, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, shape, expr, global_orient, transl = params.values()
    # betas = torch.from_numpy(shape).to(device)
    # expr = torch.from_numpy(expr).to(device)
    # leye_pose = torch.from_numpy(leye_pose).to(device)
    # reye_pose = torch.from_numpy(reye_pose).to(device)
    # jaw_pose = torch.from_numpy(jaw_pose).to(device)
    # body_pose = torch.from_numpy(body_pose).to(device)
    # global_orient = torch.from_numpy(global_orient).to(device)
    # transl = torch.from_numpy(transl).to(device)
    
    # shapes, exprs, jaws = torch.from_numpy(shapes).to(device), torch.from_numpy(exprs).to(device), torch.from_numpy(jaws).to(device)
    


    coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_0_sample.pt').to(device)
    # feats = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_0_sample.pt').to(device)
    indices = coords[..., 1:]
    
    v, f = densify_vf(source.vertices, new_faces)

    # f = new_faces
    
    # source_v = source.vertices
    

    noisy_points = (indices + 0.5) / 256 - 0.5

    cprint('begin motion')

    

    BVH = cubvh.cuBVH(torch.from_numpy(v).to(device), torch.from_numpy(f).to(device))
    _, face_id, uvw = BVH.unsigned_distance(noisy_points, return_uvw=True)
    np.savez(f'{OUTPUT_PATH}/{name}/slats/motion_dense.npz', face_id=face_id.cpu().numpy(), uvw=uvw.cpu().numpy())

def motion_coarse(name, trellis, source, target, old_idx, new_faces):
    s, R, T = list(np.load(f'{OUTPUT_PATH}/{name}/params/trans.npz').values())
    shapes, exprs, jaws = [], [], []
    params = torch.load(f'dataset/{name}/body_track/smplx_track.pth')
    shapes = params['shape']
    exprs = params['expr']
    jaws = params['jaw_pose']
    _, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, shape, expr, global_orient, transl = params.values()
    betas = torch.from_numpy(shape).to(device)
    expr = torch.from_numpy(expr).to(device)
    leye_pose = torch.from_numpy(leye_pose).to(device)
    reye_pose = torch.from_numpy(reye_pose).to(device)
    jaw_pose = torch.from_numpy(jaw_pose).to(device)
    body_pose = torch.from_numpy(body_pose).to(device)
    global_orient = torch.from_numpy(global_orient).to(device)
    transl = torch.from_numpy(transl).to(device)
    # for num in range(112):
    #     shapes.append(np.load(f'{OUTPUT_PATH}/{name}/params/{num}.npy')[:, :300])
    #     exprs.append(np.load(f'{OUTPUT_PATH}/{name}/params/{num}.npy')[:, 300:400])
    #     jaws.append(np.load(f'{OUTPUT_PATH}/{name}/params/{num}.npy')[:, 400:])
    shapes, exprs, jaws = torch.from_numpy(shapes).to(device), torch.from_numpy(exprs).to(device), torch.from_numpy(jaws).to(device)
    points, face_idx = trimesh.sample.sample_surface(trellis, count=50000)
    normals = trellis.face_normals[face_idx]
    noise = np.random.normal(scale=0.001, size=points.shape)
    noisy_points = points + normals * np.random.uniform(-0.001, 0.001, size=(len(points), 1))
    noisy_points = np.concatenate([points, noisy_points, points+noise], axis=0)
    noisy_points = torch.from_numpy(noisy_points).to(device)
    noisy_points = noisy_points.clip(-0.5+1e-6, 0.5-1e-6)

    # TODO
    # coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_0_sample.pt').to(device)
    # feats = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_0_sample.pt').to(device)
    coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_coarse_back.pt').to(device)
    feats = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_coarse_back.pt').to(device)
    indices = coords[..., 1:]
    # feats_records = -1000 * torch.ones(256, 256, 256, 101).to(device)
    # feats_records[indices[..., 0], indices[..., 1], indices[..., 2]] = feats

    f = new_faces
    
    source_v = source.vertices
    
    # debug 
    noisy_points = (indices + 0.5) / 64 - 0.5

    cprint('begin motion')

    # rast = render_tool('video', m)
    # deform = deformation[old_idx]
    # interpolate('video', rast, deformation, f)
    # exit()
    BVH = cubvh.cuBVH(torch.from_numpy(source_v).to(device), torch.from_numpy(new_faces).to(device))
    distances, face_id, uvw = BVH.unsigned_distance(noisy_points, return_uvw=True)
    # np.savez(f'{OUTPUT_PATH}/{name}/slats/motion.npz', face_id=face_id.cpu().numpy(), uvw=uvw.cpu().numpy())

    # face_region = (distances < 0.01)
    # move = noisy_points[face_region]
    # still = noisy_points[~face_region]

    my_coords = torch.floor((noisy_points+0.5)*64).int()
    faces = f[face_id.cpu().numpy()]
    s, R, T = torch.from_numpy(s).to(device), torch.from_numpy(R).to(device), torch.from_numpy(T).to(device) 
    k, t = list(np.load(f'{OUTPUT_PATH}/{name}/params/kt.npz').values())


    # move_coords = torch.floor((move+0.5)*256).int()
    # still_coords = torch.floor((still+0.5)*256).int()


    # move_feats = feats_records[move_coords[..., 0], move_coords[..., 1], move_coords[..., 2]]
    # move_mask = (move_feats>-900).all(dim=-1)
    # still_feats = feats_records[still_coords[..., 0], still_coords[..., 1], still_coords[..., 2]]
    # still_mask = (still_feats>-900).all(dim=-1)

    # move = move[move_mask]
    # move_coords = move_coords[move_mask]
    # move_feats = move_feats[move_mask]
    # still = still[still_mask]
    # still_coords = still_coords[still_mask]
    # still_feats = still_feats[still_mask]

    # move_face_id = face_id[face_region][move_mask]
    # move_uvw = uvw[face_region][move_mask]

    # faces = f[move_face_id.cpu().numpy()]
    # s, R, T = torch.from_numpy(s).to(device), torch.from_numpy(R).to(device), torch.from_numpy(T).to(device) 
    # k, t = list(np.load(f'{OUTPUT_PATH}/{name}/params/kt.npz').values())
    from tqdm import tqdm
    for num in tqdm(range(shapes.shape[0]), desc='motion'):
        # shape = shapes[num:num+1]
        # expr = exprs[num:num+1]
        # jaw = jaws[num:num+1]
        smplxout = smplx.forward(
            betas=betas[num:num+1], 
            jaw_pose=jaw_pose[num:num+1], 
            expression=expr[num:num+1],
            body_pose=body_pose[num:num+1],
            global_orient=global_orient[num:num+1],
            transl=transl[num:num+1],
            with_rott_return=True
        )
        smplx_v = smplxout['vertices'].detach().cpu().numpy().squeeze()
        smplx_v = smplx_v * k + t
        smplx_v = torch.from_numpy(smplx_v).to(device)
        smplx_v = s * smplx_v @ quaternion_to_matrix(R)[0].transpose(-2, -1) + T
        smplx_v = smplx_v.detach().cpu().numpy()
        deformation = smplx_v - source_v
        deformation = deformation[faces]
        deformation = np.einsum('bij, bi->bj', deformation, uvw.cpu().numpy())

        position = noisy_points.cpu().numpy() + deformation
        position = np.clip(position, -0.5+1e-6, 0.5-1e-6)
        output = np.floor((position + 0.5) * 64)
        # output = np.concatenate([still_coords.cpu().numpy(), output], axis=0)
        # output_feats = torch.cat([still_feats, move_feats], dim=0)
        output_feats = feats
        output_coords = torch.nn.functional.pad(torch.from_numpy(output).to(device), [1, 0], 'constant', 0)
        
        # with open(f'{OUTPUT_PATH}/{name}/slats/coords_coarse_back_{num}.pt', 'wb') as f:
        #     torch.save(output_coords.int(), f)
        # output_coords, index = output_coords.unique(return_inverse=True, dim=0)
        # temp_feats = torch.zeros(output_coords.shape[0], 101).to(device)
        # temp_feats[index] = output_feats
        # noise = sp.SparseTensor(
        #     coords=output_coords.int(),
        #     feats=temp_feats
        # )
        # start = time.time()
        
        # output_trellis = pipeline.decode_slat(noise, ['gaussian', 'mesh'])
        # end = time.time()
        # print(end-start)
        # output_trellis['gaussian'][0].save_ply(f'{OUTPUT_PATH}/{name}/objects/motion.ply')
        # v = output_trellis['mesh'][0].vertices
        # f = output_trellis['mesh'][0].faces
        # trimesh.Trimesh(vertices=v.detach().cpu().numpy(), faces=f.detach().cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/objects/motion.obj')
        
        # model = SparseFeatures2Mesh(res=256, use_color=True)
        # mesh = model(noise)
        # output = pipeline.decode_slat(noise, ['gaussian', 'mesh'])
        # output['gaussian'][0].save_ply(f'{OUTPUT_PATH}/{name}/objects/samplegg.ply')
        # v = output['mesh'][0].vertices
        # f = output['mesh'][0].faces
        # trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.detach().cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/deform/motion_dense_{num}.obj')

    
    # flame_f = flameNet.faces_tensor.cpu().numpy()
    

def motion_smplx(name, trellis, source, target, old_idx, new_faces):
    s, R, T = list(np.load(f'{OUTPUT_PATH}/{name}/params/trans.npz').values())
    shapes, exprs, jaws = [], [], []
    params = torch.load(f'dataset/{name}/body_track/smplx_track.pth')
    shapes = params['shape']
    exprs = params['expr']
    jaws = params['jaw_pose']
    _, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, shape, expr, global_orient, transl = params.values()
    betas = torch.from_numpy(shape).to(device)
    expr = torch.from_numpy(expr).to(device)
    leye_pose = torch.from_numpy(leye_pose).to(device)
    reye_pose = torch.from_numpy(reye_pose).to(device)
    jaw_pose = torch.from_numpy(jaw_pose).to(device)
    body_pose = torch.from_numpy(body_pose).to(device)
    global_orient = torch.from_numpy(global_orient).to(device)
    transl = torch.from_numpy(transl).to(device)
    # for num in range(112):
    #     shapes.append(np.load(f'{OUTPUT_PATH}/{name}/params/{num}.npy')[:, :300])
    #     exprs.append(np.load(f'{OUTPUT_PATH}/{name}/params/{num}.npy')[:, 300:400])
    #     jaws.append(np.load(f'{OUTPUT_PATH}/{name}/params/{num}.npy')[:, 400:])
    shapes, exprs, jaws = torch.from_numpy(shapes).to(device), torch.from_numpy(exprs).to(device), torch.from_numpy(jaws).to(device)
    # points, face_idx = trimesh.sample.sample_surface(trellis, count=150000)
    # normals = trellis.face_normals[face_idx]
    # noise = np.random.normal(scale=0.001, size=points.shape)
    # noisy_points = points + normals * np.random.uniform(-0.001, 0.001, size=(len(points), 1))
    # noisy_points = np.concatenate([points, noisy_points, points+noise], axis=0)
    # noisy_points = torch.from_numpy(noisy_points).to(device)
    # noisy_points = noisy_points.clip(-0.5+1e-6, 0.5-1e-6)

    # TODO
    # coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_0_sample.pt').to(device)
    # feats = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_0_sample.pt').to(device)
    coords = torch.load(f'{OUTPUT_PATH}/{name}/slats/coords_coarse_back.pt').to(device)
    feats = torch.load(f'{OUTPUT_PATH}/{name}/slats/feats_coarse_back.pt').to(device)
    indices = coords[..., 1:]
    # feats_records = -1000 * torch.ones(256, 256, 256, 101).to(device)
    # feats_records[indices[..., 0], indices[..., 1], indices[..., 2]] = feats

    f = new_faces
    
    source_v = source.vertices
    
    # debug 
    noisy_points = (indices + 0.5) / 64 - 0.5

    cprint('begin motion')

    # rast = render_tool('video', m)
    # deform = deformation[old_idx]
    # interpolate('video', rast, deformation, f)
    # exit()
    BVH = cubvh.cuBVH(torch.from_numpy(source_v).to(device), torch.from_numpy(new_faces).to(device))
    distances, face_id, uvw = BVH.unsigned_distance(noisy_points, return_uvw=True)
    np.savez(f'{OUTPUT_PATH}/{name}/slats/motion_coarse.npz', face_id=face_id.cpu().numpy(), uvw=uvw.cpu().numpy())

    # face_region = (distances < 0.01)
    # move = noisy_points[face_region]
    # still = noisy_points[~face_region]

    my_coords = torch.floor((noisy_points+0.5)*64).int()
    faces = f[face_id.cpu().numpy()]
    s, R, T = torch.from_numpy(s).to(device), torch.from_numpy(R).to(device), torch.from_numpy(T).to(device) 
    k, t = list(np.load(f'{OUTPUT_PATH}/{name}/params/kt.npz').values())


    # move_coords = torch.floor((move+0.5)*256).int()
    # still_coords = torch.floor((still+0.5)*256).int()


    # move_feats = feats_records[move_coords[..., 0], move_coords[..., 1], move_coords[..., 2]]
    # move_mask = (move_feats>-900).all(dim=-1)
    # still_feats = feats_records[still_coords[..., 0], still_coords[..., 1], still_coords[..., 2]]
    # still_mask = (still_feats>-900).all(dim=-1)

    # move = move[move_mask]
    # move_coords = move_coords[move_mask]
    # move_feats = move_feats[move_mask]
    # still = still[still_mask]
    # still_coords = still_coords[still_mask]
    # still_feats = still_feats[still_mask]

    # move_face_id = face_id[face_region][move_mask]
    # move_uvw = uvw[face_region][move_mask]

    # faces = f[move_face_id.cpu().numpy()]
    # s, R, T = torch.from_numpy(s).to(device), torch.from_numpy(R).to(device), torch.from_numpy(T).to(device) 
    # k, t = list(np.load(f'{OUTPUT_PATH}/{name}/params/kt.npz').values())
    from tqdm import tqdm
    for num in tqdm(range(shapes.shape[0]), desc='motion'):
        # shape = shapes[num:num+1]
        # expr = exprs[num:num+1]
        # jaw = jaws[num:num+1]
        smplxout = smplx.forward(
            betas=betas[num:num+1], 
            jaw_pose=jaw_pose[num:num+1], 
            expression=expr[num:num+1],
            body_pose=body_pose[num:num+1],
            global_orient=global_orient[num:num+1],
            transl=transl[num:num+1],
            with_rott_return=True
        )
        smplx_v = smplxout['vertices'].detach().cpu().numpy().squeeze()
        smplx_v = smplx_v * k + t
        smplx_v = torch.from_numpy(smplx_v).to(device)
        smplx_v = s * smplx_v @ quaternion_to_matrix(R)[0].transpose(-2, -1) + T
        smplx_v = smplx_v.detach().cpu().numpy()
        deformation = smplx_v - source_v
        deformation = deformation[faces]
        
        deformation = np.einsum('bij, bi->bj', deformation, uvw.cpu().numpy())

        position = noisy_points.cpu().numpy() + deformation
        position = np.clip(position, -0.5+1e-6, 0.5-1e-6)
        output = np.floor((position + 0.5) * 64)
        # output = np.concatenate([still_coords.cpu().numpy(), output], axis=0)
        # output_feats = torch.cat([still_feats, move_feats], dim=0)
        output_feats = feats
        output_coords = torch.nn.functional.pad(torch.from_numpy(output).to(device), [1, 0], 'constant', 0)
        
        

        # with open(f'{OUTPUT_PATH}/{name}/slats/coords_coarse_back_{num}.pt', 'wb') as f:
        #     torch.save(output_coords.int(), f)
        # output_coords, index = output_coords.unique(return_inverse=True, dim=0)
        # temp_feats = torch.zeros(output_coords.shape[0], 101).to(device)
        # temp_feats[index] = output_feats
        # noise = sp.SparseTensor(
        #     coords=output_coords.int(),
        #     feats=temp_feats
        # )
        # start = time.time()
        
        # output_trellis = pipeline.decode_slat(noise, ['gaussian', 'mesh'])
        # end = time.time()
        # print(end-start)
        # output_trellis['gaussian'][0].save_ply(f'{OUTPUT_PATH}/{name}/objects/motion.ply')
        # v = output_trellis['mesh'][0].vertices
        # f = output_trellis['mesh'][0].faces
        # trimesh.Trimesh(vertices=v.detach().cpu().numpy(), faces=f.detach().cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/objects/motion.obj')
        
        # model = SparseFeatures2Mesh(res=256, use_color=True)
        # mesh = model(noise)
        # output = pipeline.decode_slat(noise, ['gaussian', 'mesh'])
        # output['gaussian'][0].save_ply(f'{OUTPUT_PATH}/{name}/objects/samplegg.ply')
        # v = output['mesh'][0].vertices
        # f = output['mesh'][0].faces
        # trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.detach().cpu().numpy()).export(f'{OUTPUT_PATH}/{name}/deform/motion_dense_{num}.obj')

    
    # flame_f = flameNet.faces_tensor.cpu().numpy()
    

def worker(gpu, name, img_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device('cuda:0')
    os.makedirs(f'{OUTPUT_PATH}/{name}', exist_ok=True)
    with open(f'{OUTPUT_PATH}/{name}/name', 'w') as f:
        f.write(img_path)
    main(name, img_path)

def to_device(*args, **kwargs):
    func = lambda x: (x.to(device) if isinstance(x, torch.Tensor) else torch.from_numpy(x).float().to(device) if isinstance(x, np.ndarray) else x)
    # print(kwargs)
    # exit()
    if kwargs:
        return list(map(func, kwargs.values()))

def make_gts(path):
    # TODO
    smplx_params = torch.load(path)
    cam_para, body_pose, _, _, jaw_pose, _, _, shape, expr, global_orient, transl = to_device(**smplx_params)
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
    smplx_v = output['vertices'].detach()
    extrinsic = torch.tensor(
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ]
    ).to(device)
    smplx_v[:, -1092:, 2] -= 0.002
    smplxm = densify(None, trimesh.Trimesh(vertices=smplx_v[num].cpu().numpy(), faces=smplx.faces))
    proj = get_ndc_proj_matrix(cam_para, parse.shape[1:3])

    smplx_m = MeshExtractResult(vertices=torch.from_numpy(smplxm.vertices).to(device).float(), faces=torch.from_numpy(smplxm.faces).to(device).int())
    smplx_dicts = render(smplx_m, extrinsic, proj[num])


def multi():
    # new
    from tqdm import tqdm


    for name in tqdm(os.listdir('dataset_bak'), desc='test'):
        
        if os.path.exists(f'output/{name}/slats/feats_coarse_back.pt'):
            try:
                source = trimesh.load(f'/home/wzj/project/TRELLIS/output/{name}/objects/smplx.obj')
                new_faces = source.faces
                motion_save(name=name, source=source, new_faces=new_faces)
            except Exception as e:
                with open('src/errormotion1', 'a') as f:
                    f.write(f'{name} : {e}\n')
    exit()


    # old
    path = args.path
    with open(path, 'r') as f:
        mylist = json.load(f)
        print('mylist: ', len(mylist))
    # print(mylist)
    for img_path in mylist:
        name = img_path.split('/')[1]
        img_path = img_path.replace('jpg', 'png')
        cprint(f'processing {img_path}')
        try:
            backgene(name, img_path)
            # main(name, img_path)
        except Exception as e:
            # print(name)
            with open(f'src/error1', 'a') as f:
                f.write(f'{str(e)}, {name}\n')
    
def expe():
    success = 0
    from glob import glob
    lp = glob('dataset_bak/*')
    for name in lp:
        if os.path.exists(os.path.join('exp', os.path.basename(name))):
            print(f'{name} has been processed!')
            continue
        print('name is ', name)
        img_path = f'{name}/ori_imgs/000000.png'
        n = os.path.basename(name)
        try:
            main(n, img_path)
            success += 1
            torch.cuda.empty_cache()
        except Exception as e:
            # print(success)
            print(e)
        # execpt:
    print(success)



def main(
        name=None, 
        img_path=None,
):
    img_path = os.path.join(cfg['input_dir'], name, 'ori_imgs/000000.png')


    # 1. generate coarse mesh from trellis
    if os.path.exists(f'{OUTPUT_PATH}/{name}/objects/sample_outer_filtered.obj') \
        and os.path.exists(f'{OUTPUT_PATH}/{name}/objects/mesh_gene.pt'):

        print('Mesh have already been generated!')
        mesh = trimesh.load(f'{OUTPUT_PATH}/{name}/objects/sample_outer_filtered.obj')
        mesh_info = torch.load(f'{OUTPUT_PATH}/{name}/objects/mesh_gene.pt')
        m = MeshExtractResult(mesh_info['vertices'], mesh_info['faces'], mesh_info['vertex_attrs'])

    else:
        mesh, m = generate(img_path, name)

    
    # 2. render using trellis, get extrinsics and intrinsics
    obs, extrins, intrins = render(m, name)
    
    # exit()
    # print(extrins[[61, 62, 69]])
    # render(gaussian, name, return_=True, extr=extrins[[69]], intr=intrins[[69]])
    # exit()
    # 3. get landmarks from trellis renders
    gts, ids = landmarks(name)
    # exit()
    # 4. TODO tracking flame from img/video
    # initial_code, initial_mesh = flame_track(name, t=0, return_mesh=True)
    initial_code, initial_mesh = smplx_track(name, t=0, return_mesh=True) # all outputs are about flame
    # 4. align flame with trellis mesh
    mesh_aligned = smplxalign(name, initial_mesh, gts, extrins, intrins, mmesh=MeshExtractResult(vertices=torch.from_numpy(mesh.vertices).to(device).float(), faces=torch.from_numpy(mesh.faces).to(device))) # output is flame mesh
    # exit()
    cprint('align done')
    # exit()
    mynewicp(name, mesh, mesh_aligned)
    cprint('icp done')
    backgene(name, img_path, smplx=mesh_aligned)
    # exit()
    # # 5. icp trellis and initial flame
    # icp(name, mesh, mesh_aligned)

    # exit()

    # coords, _, my_mesh, k, t = flame2voxel(trimesh.load('/home/wzj/project/TRELLIS/output/video/objects_back/flame.obj'), other_mesh=other_mesh)

def transform(name):
    s, R, T = list(np.load(f'{OUTPUT_PATH}/{name}/params/trans.npz').values())
    s, R, T = torch.from_numpy(s).to(device), torch.from_numpy(R).to(device), torch.from_numpy(T).to(device) 
    k, t = list(np.load(f'{OUTPUT_PATH}/{name}/params/kt.npz').values())
    mesh = trimesh.load(f'{OUTPUT_PATH}/{name}/deform/motion_dense_150.obj')
    R = quaternion_to_matrix(R).T
    v = (mesh.vertices - T.cpu().numpy()) @ R[0].cpu().numpy() / s[0].cpu().numpy()
    v = (v - t) / k
    mesh.vertices = v
    mesh.export(f'src/0722.obj')

    data = torch.load('/home/wzj/project/TRELLIS/dataset/emotional_vids_confident-bearded-man-posing-on-white-background-SBV-346838037-4K.mp4/body_track/smplx_track.pth')
    shape = data['shape']
    expr = data['expr']
    jaw_pose = data['jaw_pose']
    shape = torch.from_numpy(shape).to(device)
    expr = torch.from_numpy(expr).to(device)
    jaw_pose = torch.from_numpy(jaw_pose).to(device)
    out = smplx.forward(
        betas=shape[150:151],
        expression=expr[150:151],
        jaw_pose=jaw_pose[150:151]
    )
    smplx_v = out['vertices'].squeeze().detach().cpu().numpy()
    f = smplx.faces
    mm = trimesh.Trimesh(vertices=smplx_v, faces=f)
    mm.export('src/0722_1.obj')

def feats_training(name):
    data = torch.load('/home/wzj/project/TRELLIS/dataset/emotional_vids_confident-bearded-man-posing-on-white-background-SBV-346838037-4K.mp4/body_track/smplx_track.pth')
    cam_para = data['cam_para'][150:151]
    rots = data['global_orient'][150:151]
    trans = data['transl'][150:151]
    cam_para = torch.from_numpy(cam_para).to(device)
    rots = torch.from_numpy(rots).to(device)
    trans = torch.from_numpy(trans).to(device)

    mesh =  trimesh.load('src/0722.obj')
    v = torch.from_numpy(mesh.vertices).to(device)
    f = torch.from_numpy(mesh.faces).to(device).int()
    v_cam = torch.bmm(v.unsqueeze(0), so3_exp_map(rots)) + trans
    proj = get_ndc_proj_matrix(cam_para, [1024, 1024])


    
    glctx = dr.RasterizeCudaContext(device=v_cam.device)
    
    
    vertices_homo = torch.cat((v_cam, torch.ones_like(v_cam[..., :1])), dim=-1)
    vertices_ndc = torch.bmm(vertices_homo, proj.permute(0, 2, 1)) # [5, 6305, 4]
    # print(vertices_ndc.max(),vertices_ndc.shape)
    # height, width = img_size_new
    # # 将 NDC 坐标[-1, 1]转换为图像空间坐标[0, width/height]
    # verts_space = ((vertices_ndc[:,:,:2] + 1) * torch.tensor([width, height], dtype=torch.float32, device = tris.device) / 2.0).int()
    # verts_space[:, :, 0] = torch.clamp(verts_space[:, :, 0], min=0, max=width - 1)
    # verts_space[:, :, 1] = torch.clamp(verts_space[:, :, 1], min=0, max=height - 1)
    
    
    rast_out_n, rast_db_n = dr.rasterize(glctx, vertices_ndc.contiguous(), f[0], resolution=[1024, 1024])
    vert_attr = 255 * torch.ones(*v_cam.shape)
    # depth_map = rast_db[...,0]
    rendered_attr = dr.interpolate(vert_attr, rast_out_n, f, rast_db=rast_db_n, diff_attrs='all')[0]
    print(rendered_attr.shape)

def get_ndc_proj_matrix(cam_paras, img_size, n=.1, f=50.):
    '''
    cam_paras: (b, 4), img_size: [h, w] -> ndc_proj_mat: (b, 4, 4)
    '''
    batch_size = cam_paras.shape[0]
    height, width = img_size
    ndc_proj_mat = cam_paras.new_zeros((batch_size, 4, 4))
    fx, fy, cx, cy = cam_paras[:, 0], cam_paras[:, 1], cam_paras[:, 2], cam_paras[:, 3]
    ndc_proj_mat[:, 0, 0] = 2*fx/(width-1)
    ndc_proj_mat[:, 0, 2] = 1-2*cx/(width-1)
    ndc_proj_mat[:, 1, 1] = -2*fy/(height-1)
    ndc_proj_mat[:, 1, 2] = 1-2*cy/(height-1)
    ndc_proj_mat[:, 2, 2] = -(f+n)/(f-n)
    ndc_proj_mat[:, 2, 3] = -(2*f*n)/(f-n)
    ndc_proj_mat[:, 3, 2] = -1.
    return ndc_proj_mat

import logging

class OnlyInfo(logging.Filter):
    def filter(self, record):
        return record.levelno < logging.WARNING

if __name__ == '__main__':
    # print(cfg['input_dir'])
    # name = 'nersemble_vids_200.mp4'
    main('celebvhq_vids_1CKEXvHN9es_2.mp4')
    exit()
    # main(name)
    # exit()
    # cfg['input_dir'] = 'inputs_track'
    # main('024.mp4')
    # exit()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    info_handler = logging.FileHandler('info.log', 'a', 'utf-8')
    info_handler.setFormatter(formatter)
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(OnlyInfo())

    error_handler = logging.FileHandler('error.log', 'a', 'utf-8')
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)

    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    # render_white_model('/home/wzj/project/TRELLIS/output/test_0/objects/new_outer_filtered.obj', show=False, save_path=f'{OUTPUT_PATH}/test_0/images/shape.png')
    # exit()
    # expe()
    # exit()
    
    # paths = [d for d in glob(os.path.expanduser('~/SDF-Avatars/inputs/*')) if os.path.isdir(d)]
    paths = os.listdir('input_ffhq')
    # print(paths)
    for path in paths:
        # print(f'processing {path}')
        # process_portrait_video(None, path, with_debug=False)
        # continue
        if os.path.exists(f'outputs_ffhq/{path}/objects/new_outer_filtered.obj') or os.path.basename(path) == '008_08':
            # print(f'{path} has been processed!')
            logger.info(f'{path} has been processed')
            continue
        else:
            if os.path.exists(f'input_ffhq/{path}/ori_imgs/000000.png'):
                print(f'processing {path}')
                # process_portrait_video(None, path, with_debug=False)
                try:
                    main(path)
                    logger.info(f'{path} has been processed')
                except Exception as e:
                    logger.error(
                        f"处理路径失败！Path: {path} 发生错误。", 
                        exc_info=True 
                    )

    exit()
    main('003_15')
    exit()
    multi()
    exit()
    for i in range(240):
        mesh_path = f'{OUTPUT_PATH}/emotional_vids_confident-bearded-man-posing-on-white-background-SBV-346838037-4K.mp4/output/output_{i}.obj'
        save_path = f'{OUTPUT_PATH}/emotional_vids_confident-bearded-man-posing-on-white-background-SBV-346838037-4K.mp4/out/{i:06d}.png'
        render_white_model(mesh_path, save_path=save_path)
    exit()
    name = 'emotional_vids_confident-bearded-man-posing-on-white-background-SBV-346838037-4K.mp4'
    source = trimesh.load(f'/home/wzj/project/TRELLIS/output/{name}/objects/smplx.obj')
    new_faces = source.faces
    motion_smplx(name=name, source=source, new_faces=new_faces, trellis=None, target=None, old_idx=None)
    exit()
    backgene('emotional_vids_confident-bearded-man-posing-on-white-background-SBV-346838037-4K.mp4', 'dataset/emotional_vids_confident-bearded-man-posing-on-white-background-SBV-346838037-4K.mp4/ori_imgs/000000.png')
    exit()
    multi()
    exit()
    main('test_0723', '/home/wzj/project/TRELLIS/dataset/emotional_vids_woman-feeling-disappointment-to-unfulfilled-hopes-in-studio-SBV-348766855-4K.mp4/ori_imgs/000190.png')
    exit()
    transform('wzj')
    exit()
    
    
    backgene('wzj', '/home/wzj/project/TRELLIS/dataset/emotional_vids_confident-bearded-man-posing-on-white-background-SBV-346838037-4K.mp4/ori_imgs/000001.jpg')
    exit()
    
    icpsmplx('test_smplx', trimesh.load('/home/wzj/project/TRELLIS/output/test_smplx/objects/sample_outer_filtered.obj'), trimesh.load('/home/wzj/project/TRELLIS/output/test_smplx/objects/smplx.obj'))
    exit()
    main('test_smplx', '/home/wzj/project/TRELLIS/dataset/emotional_vids_young-woman-smiling-in-urban-evening-setting-with-city-lights-and-vibrant-atmospher-SBV-349394705-4K.mp4/ori_imgs/000000.jpg')
    exit()
    # for i in range(112):
    #     mesh_path = f'{OUTPUT_PATH}/test_0/deform/motion_dense_{i}.obj'
    #     save_path = f'{OUTPUT_PATH}/test_0/out/{i}.png'
    #     render_white_model(mesh_path, save_path=save_path)
    # exit()
    face_indices = np.load(f'{root_path}/flame1/data/faces.npy')
    flame = trimesh.load(f'{OUTPUT_PATH}/test_0/objects/flame.obj')
    old_idx, new_faces = submesh(flame, face_indices)
    
    motion('test_0', trimesh.load(f'/home/wzj/project/TRELLIS/output/test_0/objects/new_outer_filtered.obj'), trimesh.load('/home/wzj/project/TRELLIS/output/test_0/objects/flame.obj'), 45, old_idx, new_faces)
    exit()
    voxel_move('test_0')
    exit()
    render_gts('test_0')
    exit()
    test_model()
    exit()
    render_white_model('/home/wzj/project/TRELLIS/output/test_0/objects/new_outer_filtered.obj', show=False, save_path=f'{OUTPUT_PATH}/test_0/images/shape.png')
    exit()
    voxel_move('test_0')
    exit()
    # main('test_0', f'{OUTPUT_PATH}/test_0/ori_imgs/000000.jpg')
    # exit()
    # generate('/home/wzj/project/TRELLIS/wzj/flame/ori_imgs/000010.jpg', 'hello')
    # exit()
    # print(globals())
    # m = trimesh.load('/home/wzj/project/TRELLIS/output/video/objects/sample_outer_filtered.obj')
    # loss, flame_p = correspond('video', trimesh.load('/home/wzj/project/TRELLIS/output/video/objects/flame.obj'), torch.from_numpy(m.vertices).to(device), torch.from_numpy(m.faces).to(device))
    # print(loss, flame_p.shape)
    # trellis = trimesh.load('/home/wzj/project/TRELLIS/output/video/objects/sample_outer_filtered.obj')
    # flame = trimesh.load('/home/wzj/project/TRELLIS/output/video/objects/flame.obj')
    # icp('video', trellis, flame)
    icp('test_0', trimesh.load('/home/wzj/project/TRELLIS/output/test_0/objects/new_outer_filtered.obj'), trimesh.load('/home/wzj/project/TRELLIS/output/test_0/objects/flame.obj'), back=True)
    exit()
    backgene('test_0', '/home/wzj/project/TRELLIS/output/test_0/ori_imgs/000000.jpg', None, 'new')
    exit()
    face_indices = np.load(f'{root_path}/flame1/data/faces.npy')
    flame = trimesh.load(f'{OUTPUT_PATH}/video/objects/flame.obj')
    old_idx, new_faces = submesh(flame, face_indices)
    
    motion('video', trimesh.load(f'/home/wzj/project/TRELLIS/output/video/objects/new_outer_filtered.obj'), trimesh.load('/home/wzj/project/TRELLIS/output/video/objects/flame.obj'), 45, old_idx, new_faces)