import torch
import numpy as np
import trimesh
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.smplx.smplx import SMPLX


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

def split():
    mname = 'emotional_vids_cheerful-young-man-in-pink-casual-t-shirt-talks-to-camera-in-modern-kitchen-at-home-SBV-338781050-4K.mp4'
    mname = 'pipe'
    track_path = f'/home/wzj/project/TRELLIS/dataset_bak/{mname}/body_track/smplx_track.pth'

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
    f = smplx.faces[mask]
    p = output.vertices[0, :-1092].detach().cpu().numpy()
    trans = np.load(f'outputs/{mname}/params/trans.npz')
    kt = np.load(f'outputs/{mname}/params/kt.npz')
    k, t = list(kt.values())
    s, R, T = list(trans.values())
    p = k * p + t
    from utils3d.torch import quaternion_to_matrix
    Rota = quaternion_to_matrix(torch.from_numpy(R)).squeeze(0)

    p = s * p @ Rota.transpose(-2, -1).cpu().numpy() + T
    x = trimesh.Trimesh(vertices=p, faces=f)


    fff = smplx.faces[~mask]
    vvv = output.vertices[0, -1092:]
    fff -= fff.min()
    trimesh.Trimesh(vertices=vvv.detach().cpu().numpy(), faces=fff).export('src/tutils/eye.obj')
    x.export(f'exp/{mname}/objects/sm.obj')
    x = trimesh.load(f'exp/{mname}/objects/sm.obj')
    test = (f - x.faces).sum()
    print((p - x.vertices).sum())
    print(test)

if __name__ == '__main__':
    split()