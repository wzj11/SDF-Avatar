import torch
cube_corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [
        1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.int)
cube_neighbor = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
cube_edges = torch.tensor([0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6,
                2, 0, 3, 1, 7, 5, 6, 4], dtype=torch.long, requires_grad=False)
     
def construct_dense_grid(res, device='cuda'):
    '''construct a dense grid based on resolution'''
    res_v = res + 1
    vertsid = torch.arange(res_v ** 3, device=device)
    coordsid = vertsid.reshape(res_v, res_v, res_v)[:res, :res, :res].flatten()
    cube_corners_bias = (cube_corners[:, 0] * res_v + cube_corners[:, 1]) * res_v + cube_corners[:, 2]
    cube_fx8 = (coordsid.unsqueeze(1) + cube_corners_bias.unsqueeze(0).to(device))
    verts = torch.stack([vertsid // (res_v ** 2), (vertsid // res_v) % res_v, vertsid % res_v], dim=1)
    return verts, cube_fx8


def construct_voxel_grid(coords):
    verts = (cube_corners.unsqueeze(0).to(coords) + coords.unsqueeze(1)).reshape(-1, 3)
    verts_unique, inverse_indices = torch.unique(verts, dim=0, return_inverse=True)
    cubes = inverse_indices.reshape(-1, 8)
    return verts_unique, cubes


def cubes_to_verts(num_verts, cubes, value, reduce='mean'):
    """
    Args:
        cubes [Vx8] verts index for each cube
        value [Vx8xM] value to be scattered
    Operation:
        reduced[cubes[i][j]][k] += value[i][k]
    """
    M = value.shape[2] # number of channels
    reduced = torch.zeros(num_verts, M, device=cubes.device)
    return torch.scatter_reduce(reduced, 0, 
        cubes.unsqueeze(-1).expand(-1, -1, M).flatten(0, 1), 
        value.flatten(0, 1), reduce=reduce, include_self=False)
    
def sparse_cube2verts(coords, feats, training=True, wzj=False):
    new_coords, cubes = construct_voxel_grid(coords)
    new_feats = cubes_to_verts(new_coords.shape[0], cubes, feats)
    if training:
        con_loss = torch.mean((feats - new_feats[cubes]) ** 2)
    else:
        con_loss = 0.0
    if not wzj:
        return new_coords, new_feats, con_loss
    else:
        return new_coords, new_feats, con_loss, cubes
    

def get_dense_attrs(coords : torch.Tensor, feats : torch.Tensor, res : int, sdf_init=True, no_other=False, training=False):
    F = feats.shape[-1]
    # print(coords.shape)
    # exit()
    dense_attrs = torch.zeros([res] * 3 + [F], device=feats.device)
    if sdf_init:
        dense_attrs[..., 0] = 100 if not no_other else 0.1 # initial outside sdf value 
    dense_attrs[coords[:, 0], coords[:, 1], coords[:, 2], :] = feats
    
    

    return dense_attrs.reshape(-1, F)


def get_defomed_verts(v_pos : torch.Tensor, deform : torch.Tensor, res):
    return v_pos / res - 0.5 + (1 - 1e-8) / (res * 2) * torch.tanh(deform)


def deform_wzj(sdf, cube, name=None, output_path=None):
    # coos = sdf.abs() < 100
    # surf1 = coos[cube.reshape(-1)].reshape(-1, 8).any(dim=-1)
    coos = sdf < 0
    coo_fx8 = coos[cube.reshape(-1)].reshape(-1, 8)
    _occ_sum = torch.sum(coo_fx8, -1)
    surf_cubes = (_occ_sum > 0) & (_occ_sum < 8)
    coo_fx8 = (surf_cubes).nonzero()
    x = coo_fx8 // (256 ** 2)
    # y = (coo_fx8 % (256 ** 2)) // 256
    y = (coo_fx8 // 256) % 256
    z = (coo_fx8) % 256
    coords = torch.cat([x, y, z], dim=-1)
    # if name is not None:
    #     # torch.save(f'exp/{name}/')
    #     torch.save((coords / 256 - 0.5), f'exp/{name}/params/marching.pt')

    verts = (cube_corners.unsqueeze(0).to(coords) + coords.unsqueeze(1)).reshape(-1, 3)
    verts_unique, indices = torch.unique(verts, dim=0, return_inverse=True)
    # indices: (N)
    if name is not None:
        # torch.save(f'exp/{name}/')
        torch.save(indices, f'{output_path}/{name}/params/marching.pt')
    # print(coords.shape, coords)
    # print(verts_unique.shape)
    return verts_unique