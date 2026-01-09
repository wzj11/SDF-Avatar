import torch
# from ...modules.sparse import SparseTensor
from easydict import EasyDict as edict
from .utils_cube import *
from .flexicubes.flexicubes import FlexiCubes


class MeshExtractResult:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None,
        res=64
    ):
        self.vertices = vertices
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.face_normal = self.comput_face_normals(vertices, faces)
        self.res = res
        self.success = (vertices.shape[0] != 0 and faces.shape[0] != 0)

        # training only
        self.tsdf_v = None
        self.tsdf_s = None
        self.reg_loss = None
        
    def comput_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        # print(face_normals.min(), face_normals.max(), face_normals.shape)
        return face_normals[:, None, :].repeat(1, 3, 1)
                
    def comput_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        return v_normals   





class SparseFeatures2Mesh:
    def __init__(self, device="cuda", res=64, use_color=True):
        '''
        a model to generate a mesh from sparse features structures using flexicube
        '''
        super().__init__()
        self.device=device
        self.res = res
        self.mesh_extractor = FlexiCubes(device=device)
        self.sdf_bias = -1.0 / res
        verts, cube = construct_dense_grid(self.res, self.device)
        self.reg_c = cube.to(self.device)
        self.reg_v = verts.to(self.device)
        self.use_color = use_color
        self._calc_layout()
    
    def _calc_layout(self):
        LAYOUTS = {
            'sdf': {'shape': (8, 1), 'size': 8},
            'deform': {'shape': (8, 3), 'size': 8 * 3},
            'weights': {'shape': (21,), 'size': 21}
        }
        if self.use_color:
            '''
            6 channel color including normal map
            '''
            LAYOUTS['color'] = {'shape': (8, 6,), 'size': 8 * 6}
        self.layouts = edict(LAYOUTS)
        start = 0
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.feats_channels = start
        
    def get_layout(self, feats : torch.Tensor, name : str):
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name]['shape'])
    
    def __call__(self, cubefeats, training=False, v_a=None, dcoords=None, dfeats=None, debug=False, debug_sdf=None, mask=None, no_other=False, return_v_a=False, n_coords=None, name=None, marching_mask=None, change_marching=False, indices=None, output_path=None):
        """
        Generates a mesh based on the specified sparse voxel structures.
        Args:
            cube_attrs [Nx21] : Sparse Tensor attrs about cube weights
            verts_attrs [Nx10] : [0:1] SDF [1:4] deform [4:7] color [7:10] normal 
        Returns:
            return the success tag and ni you loss, 
        """
        # add sdf bias to verts_attrs
        if dcoords is not None:
            coords = dcoords[:, 1:]
        else:
            coords = cubefeats.coords[:, 1:]

        # print(coords.min(), coords.max())
        # exit()
        # wzj_test = coords[:, 0] * (256**2) + coords[:, 1] * 256 + coords[:, 2]
        # boo = torch.zeros((256, 256, 256), dtype=torch.bool).to(coords.device).flatten()
        # boo[wzj_test] = True
        if dfeats is not None:
            feats = dfeats
        else:
            feats = cubefeats.feats
        # coords = cubefeats.coords[:, 1:]
        # print('cubefeats:', coords.shape)
        # print(coords.min(), coords.max())
        # torch.save(coords, '/home/wzj/project/TRELLIS/output/000/coords.pt')
        # print('save ok!')
        # feats = cubefeats.feats
        # print(feats.shape)
        # torch.save(feats, '/home/wzj/project/TRELLIS/output/000/feats.pt')
        
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        sdf = sdf + self.sdf_bias
        # print(sdf.min(), sdf.max())
        # exit()
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss, cubes = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training, wzj=True)
        if training:
            v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res+1, sdf_init=True, no_other=no_other, training=training)
        else:
            v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res+1, sdf_init=True, no_other=no_other, training=training)
        if debug:
            return v_attrs_d[..., 0]
        
        weights_d = get_dense_attrs(coords, weights, res=self.res, sdf_init=False)
        if self.use_color:
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None 
        if v_a is not None:
            trans = get_dense_attrs(n_coords, v_a, res=self.res+1, sdf_init=False)
            # m = (trans == 100)
            # print(sdf_d[m[..., 0]].max(), sdf_d[m[..., 0]].min())
            # exit()
        x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res)
        if v_a is not None:
            x_nx3 = x_nx3 + trans
        if debug_sdf is not None:
            sdf_d = debug_sdf
        if mask is not None:
            sdf_d[mask] *= -1
        if return_v_a:
            # print(v_pos.shape)
            v_pos_ = deform_wzj(sdf_d, self.reg_c, name=name, output_path=output_path)
            # print(v_pos_.dtype, ' v_pos_.dtype')
            index = v_pos_[..., 0] * (256 ** 2) + v_pos_[..., 1] * 256 + v_pos_[..., 2]
            v_pos_i = (v_pos_ / 256) - 0.5
            v_pos_i += (1 - 1e-8) / (256 * 2) * torch.tanh(deform_d[index])
            torch.save(v_pos_, f'{output_path}/{name}/params/v_pos_.pt')
            torch.save(v_pos_i, f'{output_path}/{name}/params/v_pos_i.pt')
        # exit()
        #     sdf_d = torch.where(mask, torch.full_like(sdf_d, -1.0), sdf_d)
        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=self.reg_c,
            resolution=self.res,
            beta=weights_d[:, :12],
            alpha=weights_d[:, 12:20],
            gamma_f=weights_d[:, 20],
            voxelgrid_colors=colors_d,
            training=training,
            # no_other=no_other
            marching_mask=marching_mask,
            change_marching=change_marching,
            name=name,
            indices=indices,
            output_path=output_path
            )
        
        # count = 0
        # import time
        # while count < 100:

        #     if count == 10:
        #         start = time.time()
        #     vertices, faces, L_dev, colors = self.mesh_extractor(
        #         voxelgrid_vertices=x_nx3,
        #         scalar_field=sdf_d,
        #         cube_idx=self.reg_c,
        #         resolution=self.res,
        #         beta=weights_d[:, :12],
        #         alpha=weights_d[:, 12:20],
        #         gamma_f=weights_d[:, 20],
        #         voxelgrid_colors=colors_d,
        #         training=training)
        #     count += 1
        #     if count == 60:
        #         end = time.time()

        # print('time consume:', end - start)

        mesh = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res)
        if return_v_a:
            mesh.v_p = v_pos_i
        if training:
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5
            # reg_loss += (weights[:,:20]).abs().mean() * 0.2
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
            mesh.tsdf_s = v_attrs[:, 0][cubes]
        return mesh
