import torch
import nvdiffrast.torch as dr
from easydict import EasyDict as edict

def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = - (far + near) / (far - near)
    ret[2, 3] = 2 * near * far / (near - far)
    ret[3, 2] = -1.
    return ret

def render(
            mesh,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            HW,
            return_types = ["mask", "normal", "depth"],
            v=None,
            device='cuda:0',
            normalize=False,
            rot_mvadapter=None,
            attr=None,
            only_texture=False,
            only_rast=False,
            shading=None
        ) -> edict:

        """
        Render the mesh.

        Args:
            mesh : meshmodel
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            HW: List, size of image
            return_types (list): list of return types, can be "mask", "depth", "normal_map", "normal", "color"

        Returns:
            edict based on return_types containing:
                color (torch.Tensor): [3, H, W] rendered color image
                depth (torch.Tensor): [H, W] rendered depth image
                normal (torch.Tensor): [3, H, W] rendered normal image
                normal_map (torch.Tensor): [3, H, W] rendered normal map image
                mask (torch.Tensor): [H, W] rendered mask image
        """
        H, W = HW
        resolution = 1024
        near = 0.1
        far = 100
        ssaa = 1
        
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            default_img = torch.zeros((1, resolution, resolution, 3), dtype=torch.float32, device=device)
            ret_dict = {k : default_img if k in ['normal', 'normal_map', 'color'] else default_img[..., :1] for k in return_types}
            return ret_dict
        
        if intrinsics.shape[-1] == 4:
            perspective = intrinsics
        else:
            perspective = intrinsics_to_projection(intrinsics, near, far)
        assert extrinsics.ndim <= 3, "Shape of extrinsics must be (B, 4, 4) or (4, 4)"
        if extrinsics.ndim == 2:
            RT = extrinsics.unsqueeze(0)
            full_proj = (perspective @ extrinsics).unsqueeze(0)
        else:
            RT = extrinsics
            full_proj = (perspective @ extrinsics)
        
        if v is not None:
            vertices = v.unsqueeze(0)
        else:
            vertices = mesh.vertices.unsqueeze(0)

        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        # print(vertices_homo.shape, vertices_homo.expand((RT.shape[0], -1, -1)), RT.shape)
        vertices_camera = torch.bmm(vertices_homo.expand((RT.shape[0], -1, -1)), RT.transpose(-1, -2))
        vertices_clip = torch.bmm(vertices_homo.expand((full_proj.shape[0], -1, -1)), full_proj.transpose(-1, -2))
    
        faces_int = mesh.faces.int().contiguous()
        glctx = dr.RasterizeCudaContext(device=device)
        rast, _ = dr.rasterize(
            glctx, vertices_clip, faces_int, (H, W))
        if only_rast:
            return rast
        out_dict = edict()
        for type in return_types:
            img = None
            if type == "mask":
                img = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
            elif type == "depth":
                img = dr.interpolate(-vertices_camera[..., 2:3].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "normal" :
                if extrinsics.ndim == 2:
                    img = dr.interpolate(
                        mesh.face_normal.reshape(1, -1, 3), rast,
                        torch.arange(mesh.faces.shape[0] * 3, device=device, dtype=torch.int).reshape(-1, 3)
                    )[0]
                    img = torch.nn.functional.normalize(img, dim=-1)
                    img = dr.antialias(img, rast, vertices_clip, faces_int)
                else:
                    img = dr.interpolate(
                        mesh.face_normal.reshape(1, -1, 3), rast,
                        torch.arange(mesh.faces.shape[0] * 3, device=device, dtype=torch.int).reshape(-1, 3)
                    )[0]
                    
                # normalize norm pictures
                img = (img + 1) / 2
            elif type == "normal_map" :
                img = dr.interpolate(mesh.vertex_attrs[:, 3:].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "color" :
                if attr is not None:
                    attrs = attr
                else:
                    attrs = mesh.vertex_attrs

                if shading is not None:
                    if extrinsics.ndim == 2:
                        normals = dr.interpolate(
                            mesh.face_normal.reshape(1, -1, 3), rast,
                            torch.arange(mesh.faces.shape[0] * 3, device=device, dtype=torch.int).reshape(-1, 3)
                        )[0]
                        normals = torch.nn.functional.normalize(normals, dim=-1)
                        normals = dr.antialias(normals, rast, vertices_clip, faces_int) # (H, W, 3)
                        # shading: (9, 3)
                        


                    else:
                        normals = dr.interpolate(
                            mesh.face_normal.reshape(1, -1, 3), rast,
                            torch.arange(mesh.faces.shape[0] * 3, device=device, dtype=torch.int).reshape(-1, 3)
                        )[0]
                    normals = torch.nn.functional.normalize(normals, dim=-1).detach()
                    x = normals[..., 0]
                    y = normals[..., 1]
                    z = normals[..., 2]
                    c0 = 0.28209479177387814
                    c1 = 0.4886025119029199
                    c2 = 1.0925484305920792
                    c3 = 0.31539156525252005
                    c4 = 0.5462742152960396
                    basis = torch.empty((*normals.shape[:-1], 9), device=normals.device, dtype=normals.dtype)
                    basis[..., 0] = 0.886227
                    basis[..., 1] = -1.023326 * y
                    basis[..., 2] =  1.023326 * z
                    basis[..., 3] = -1.023326 * x
                    basis[..., 4] =  0.858086 * x * y
                    basis[..., 5] = -0.858086 * y * z
                    basis[..., 6] =  0.247708 * (3 * z * z - 1)
                    basis[..., 7] = -0.858086 * x * z
                    basis[..., 8] =  0.429043 * (x * x - y * y)
                    # print(basis.shape)
                    # print(normals.shape)
                    irradiance = torch.matmul(basis.reshape(-1, 9), shading).reshape(*normals.shape[:-1], 3)
                    irradiance = torch.nn.functional.relu(irradiance)

                    # pix_normals = torch.nn.functional.normalize(normals, dim=-1)
                    # shade = (pix_normals @ shading[1:].unsqueeze(-1)).squeeze(-1) + shading[0]
                    # shade = torch.relu(shade).unsqueeze(-1)
                    
                
                if extrinsics.ndim == 3:
                    img = dr.interpolate(attrs[None, :, :3].contiguous(), rast, faces_int)[0]
                else:
                    img = dr.interpolate(attrs[:, :3].contiguous(), rast, faces_int)[0]
                if shading is not None:
                    img = img * irradiance
                if not only_texture:
                    img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "pos":
                # if rot_mvadapter is not None:
                attr = mesh.vertices.clone().detach()
                
                max_scale = attr.abs().max()
                attr = attr / max_scale * 0.5
                if rot_mvadapter is not None:
                    attr = attr @ rot_mvadapter.T
                    print('wzj')
                img = dr.interpolate(attr[None], rast, faces_int)[0]
                img = (img + 0.5).clamp(0, 1)
            elif type == "posi":
                # if rot_mvadapter is not None:
                attr = mesh.vertices.clone().detach()
                img = dr.interpolate(attr[None], rast, faces_int)[0]
                # img = (img + 0.5).clamp(0, 1)

            if ssaa > 1:
                img = torch.nn.functional.interpolate(img.permute(0, 3, 1, 2), (resolution, resolution), mode='bilinear', align_corners=False, antialias=True)
                img = img.squeeze()
            else:
                img = img.permute(0, 3, 1, 2).squeeze()
            out_dict[type] = img

        return out_dict


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