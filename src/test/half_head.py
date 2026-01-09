import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.utils_io import read_png
import numpy as np
from models.smplx.smplx import SMPLX
import torch
from utils.utils_geo import wzj_final, densify

device = 'cuda:0'
num_betas = 300
num_expression_coeffs = 100
model_path = 'models/smplx/SMPLX2020'
smplx = SMPLX(num_betas=num_betas, num_expression_coeffs=num_expression_coeffs, model_path=model_path)
smplx.to(device)

def to_device(*args, **kwargs):
    func = lambda x: (x.to(device) if isinstance(x, torch.Tensor) else torch.from_numpy(x).float().to(device) if isinstance(x, np.ndarray) else x)
    # print(kwargs)
    # exit()
    if kwargs:
        return list(map(func, kwargs.values()))

def mouth(name):
    parse = read_png(f'inputs/{name}/parsing')
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

    vertices = smplx_v[0]
    faces = smplx.faces
    vertices, faces = densify(vertices, faces)
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    heads = np.load('params/head.npy')

    half = mesh.submesh([heads])[0]
    half.export('half.obj')

    all_v = half.vertices[half.faces] #(N, 3, 3)
    mean_v = half.vertices.mean(axis=0, keepdims=True)[..., -1]
    # print(mean_v.shape)
    # exit()
    params = np.ones((all_v.shape[0], 3)) * 1 / 3
    final_v = np.einsum('bij, bi -> bj', all_v, params)[..., -1]
    final_mask = final_v > mean_v
    my_faces = final_mask.nonzero()[0]
    # my_faces = half.faces[final_mask]
    my_out = half.submesh([my_faces])[0]
    my_out.export('half_head.obj')


if __name__ == '__main__':
    name = 'pipe'
    mouth(name)