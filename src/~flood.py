import numpy as np
import torch
import cubvh
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trellis.representations.mesh import SparseFeatures2Mesh
from trellis.modules import sparse as sp
import trimesh
from utils.utils_geo import wzj_final

model = SparseFeatures2Mesh(res=256, use_color=True)
device = 'cuda:0'

def main():
    coords = torch.load('outputs/004_04/slats/coords_0_new.pt').to(device)
    feats = torch.load('outputs/004_04/slats/feats_0_new.pt').to(device)
    inputs = sp.SparseTensor(
        coords=coords,
        feats=feats,
    )

    mesh = model(inputs)
    x = trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.detach().cpu().numpy())
    x.export('src/flood.obj')

    with torch.no_grad():
        temp_d = model(inputs, debug=True)
        sdf_d = temp_d.reshape((257, 257, 257))
        sdf_d = sdf_d.detach().cpu().numpy()
        sdf_mask = wzj_final(
            sdf_d,
        )
        sdf_mask = sdf_mask.reshape(-1)
        sdf_mask = torch.from_numpy(sdf_mask).to(device)
        print(sdf_mask.sum())

    with torch.no_grad():
        temp_d = model(inputs, debug=True)
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
    mesh = model(inputs, mask=sdf_mask)
    x = trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.detach().cpu().numpy())
    x.export('src/flood_new.obj')

if __name__ == '__main__':
    main()
