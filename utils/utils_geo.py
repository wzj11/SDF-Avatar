import pymeshlab
from pymeshlab import Percentage
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy import ndimage

def flood_fill(sdf, ndims=2):
    # from scipy import ndimage
    unsigned = np.abs(sdf)
    mask = (sdf >= 0)

    struct = ndimage.generate_binary_structure(ndims, 1)
    labeled, num = ndimage.label(mask, structure=struct)

    Dx, Dy, Dx = sdf.shape
    border_mask = np.zeros_like(mask, dtype=bool)
    border_mask[0, :, :] = border_mask[-1, :, :] = True
    border_mask[:, 0, :] = border_mask[:, -1, :] = True
    border_mask[:, :, 0] = border_mask[:, :, -1] = True

    border_labels = np.unique(labeled[border_mask])
    outside = np.isin(labeled, border_labels)
    cv2.imwrite('src/tutils/utils_geo_.png', ~outside[..., None].astype(np.int8) * 255.)

    fixed = unsigned.copy()
    fixed[outside] = +unsigned[outside]
    fixed[~outside] = -unsigned[~outside]

    return fixed

def densify(
    vertices,
    faces
):
    '''
    only need numpy.ndarray
    '''
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertices, faces))
    ms.apply_filter(
        "meshing_surface_subdivision_ls3_loop",
        iterations=2,
        threshold=Percentage(0),
    )
    v = ms.current_mesh().vertex_matrix()
    f = ms.current_mesh().face_matrix()
    return v, f

def quick_check(sdf, eps=1e-6):
    pos = (sdf >  eps).sum()
    neg = (sdf < -eps).sum()
    zero = (np.abs(sdf) <= eps).sum()
    print(f">0: {pos}, <0: {neg}, ~0: {zero}")


def fix_sdf_sign_via_propagation(
    sdf, eps=1e-6, connectivity=1, close_iters=0, outside_is_positive=True,
    ensure_seeds=True
):
    """
    outside_is_positive: True 表示外部为正；False 表示外部为负。
    ensure_seeds: 没有边界种子时，自动用“虚拟外部”兜底（见下）。
    """
    # 选择“外部候选掩码”
    if outside_is_positive:
        outside_mask = sdf > eps
    else:
        outside_mask = sdf < -eps

    # 可选：轻微闭运算，封掉针孔（别用太大）
    structure = ndi.generate_binary_structure(3, connectivity)
    if close_iters > 0:
        outside_mask = ndi.binary_closing(outside_mask, structure=structure, iterations=close_iters)

    Z, Y, X = sdf.shape

    # ——构造边界种子——
    seed = np.zeros_like(outside_mask, bool)
    seed[0,:,:]  = outside_mask[0,:,:]
    seed[-1,:,:] = outside_mask[-1,:,:]
    seed[:,0,:]  = outside_mask[:,0,:]
    seed[:,-1,:] = outside_mask[:,-1,:]
    seed[:,:,0]  = outside_mask[:,:,0]
    seed[:,:,-1] = outside_mask[:,:,-1]

    # 若边界没有任何种子，用“虚拟外部”兜底：在 padding 体积的最外层强制放种子
    if ensure_seeds and seed.sum() == 0:
        pad = np.pad(outside_mask, 1, constant_values=False)
        seedp = np.zeros_like(pad, bool)
        seedp[[0, -1], :, :] = True
        seedp[:, [0, -1], :] = True
        seedp[:, :, [0, -1]] = True
        outside_p = ndi.binary_propagation(seedp, mask=pad, structure=structure)
        outside = outside_p[1:-1, 1:-1, 1:-1]
    else:
        outside = ndi.binary_propagation(seed, mask=outside_mask, structure=structure)

    # 依据外部符号约定生成最终符号
    if outside_is_positive:
        sign = np.where(outside, +1.0, -1.0)
    else:
        sign = np.where(outside, -1.0, +1.0)

    return sign * np.abs(sdf)



def fix_sdf_sign_narrowband_only(
    sdf, 
    band_tau=99,           # 窄带阈值: |sdf| <= band_tau 被认为是可靠区域
    eps=0,               # 正负判断的容差
    seal_iters=2,           # 给窄带加厚几层以封住缺口
    connectivity=1,         # 18邻接能更好封小洞
    outside_is_positive=True  # 你的符号约定：外正内负（如相反改为 False）
):
    """
    只修正符号，不重建距离。对窄带外的错误正值进行符号反转。
    """
    # 1. 确定窄带（可靠区域），以及正/负区域
    known_band = np.abs(sdf) <= band_tau
    for i, l in enumerate(known_band):
        print(l[0, 0])
        cv2.imwrite(f'src/tutils/fix_sdf/fix_sdf_{i}.png', l.astype(np.int8) * 255.)
    
    print(known_band.sum(), 'wzj                       jjj')
    known_pos  = (sdf >  eps) & known_band
    
    known_neg  = (sdf < -eps) & known_band
    for i, l in enumerate(known_neg):
        print(l[0, 0])
        cv2.imwrite(f'src/tutils/fix_sdf/fix_sdf_pos_{i}.png', l.astype(np.int8) * 255.)
    # 2. 生成“墙”——窄带及可靠负区（外部传播不能穿过）
    struct = ndi.generate_binary_structure(3, connectivity)
    wall = ndi.binary_dilation(known_neg, structure=struct, iterations=seal_iters)
    # wall |= known_neg  # 防止外部传播穿过已知内部负区

    unknown = ~known_band
    allowed = unknown & (~wall)
      # 允许传播的区域

    # 3. 以边界正体素为种子，从外向内传播（外部区域）
    seed = np.zeros_like(allowed, dtype=bool)
    seed[0,:,:] = seed[-1,:,:] = True
    seed[:,0,:] = seed[:,-1,:] = True
    seed[:,:,0] = seed[:,:,-1] = True
    seed &= allowed

    structure = ndi.generate_binary_structure(3, connectivity)
    outside_unknown = ndi.binary_propagation(seed, mask=allowed, structure=structure)
    inside_unknown = unknown & (~outside_unknown)

    # 4. 生成符号场（只修符号）
    sign = np.zeros_like(sdf, dtype=float)

    if outside_is_positive:
        sign[outside_unknown] = +1.0
        sign[inside_unknown]  = -1.0
        sign[known_pos] = +1.0
        sign[known_neg] = -1.0
    else:
        sign[outside_unknown] = -1.0
        sign[inside_unknown]  = +1.0
        sign[known_pos] = -1.0
        sign[known_neg] = +1.0
    
    unset = (sign == 0.0)
    sign[unset] = np.sign(np.where(outside_is_positive, +1, -1))
    # 5. 用已知窄带和可靠区域的符号覆写（保持原来的准确信息）
    # sign[known_pos] = +1.0 if outside_is_positive else -1.0
    # sign[known_neg] = -1.0 if outside_is_positive else +1.0
    for i, l in enumerate(sign):

        cv2.imwrite(f'src/tutils/fix_sdf/fix_sdf_debug_{i}.png', l.astype(np.int8) * 255.)
    
    # 6. 返回只修符号的 SDF
    sdf_fixed = sign * np.abs(sdf)
    return sdf_fixed


def fix_sign_two_fronts_biased(
    sdf,
    band_tau=99,         # 窄带 |sdf|<=band_tau 认为是“已知”，不改数值，只用来放种子
    eps=0,             # 判断正/负的容差（仅窄带内）
    connectivity=1,       # 6邻接，保守
    erode_inside_seeds=1, # 腐蚀内部种子，避免内部过强；0 表示不腐蚀
    outside_bias_margin=0.5, # 外部偏置 m：判外部条件 d_out <= d_in + m
    outer_guard_band=2,   # 外部保护带半径（体素）：距外边界<=R 的未知区直接归外部；0 关闭
):
    # 1) 窄带内可信正/负
    known_band = np.abs(sdf) <= band_tau
    known_pos  = (sdf >  eps) & known_band
    known_neg  = (sdf < -eps) & known_band

    # 未知区（只在这里做归类）
    unknown = ~known_neg

    # 2) 内部/外部种子
    # 外部种子：体素外边界 ∩ 未知
    seed_out = np.zeros_like(unknown, bool)
    seed_out[0,:,:]=seed_out[-1,:,:]=True
    seed_out[:,0,:]=seed_out[:,-1,:]=True
    seed_out[:,:,0]=seed_out[:,:,-1]=True
    seed_out &= unknown

    # 内部种子：窄带内的可信负
    seed_in = np.zeros_like(unknown, bool)
    seed_in[128, 128, 128] = True
    # seed_in = known_neg.copy()
    if erode_inside_seeds > 0 and seed_in.any():
        struct = ndi.generate_binary_structure(3, connectivity)
        seed_in = ndi.binary_erosion(seed_in, structure=struct, iterations=erode_inside_seeds)

    # 3) 外部保护带（防止外部起步慢被吃掉）
    if outer_guard_band > 0:
        # 计算到外边界的“曼哈顿”近似距离：用反向 EDT 技巧
        border = np.zeros_like(unknown, bool)
        border[0,:,:]=border[-1,:,:]=True
        border[:,0,:]=border[:,-1,:]=True
        border[:,:,0]=border[:,:,-1]=True
        # 到边界的距离：把“非边界”当 True 做 EDT
        d2border = ndi.distance_transform_edt(~border)
        guard = (d2border <= outer_guard_band) & unknown
    else:
        guard = np.zeros_like(unknown, bool)

    # 4) 计算到外部/内部种子的距离（只在 unknown 中）
    #   使用 EDT 是为了欧氏距离更平滑；不重建 SDF 幅值，仅用于分类
    #   trick: 到种子距离 = EDT(~seed)
    d_out = ndi.distance_transform_edt(~seed_out)
    d_in  = ndi.distance_transform_edt(~seed_in) if seed_in.any() else np.full_like(d_out, np.inf, dtype=float)

    # 5) 分类规则（带外部偏置 + 外部保护带）
    outside_unknown = unknown & ((d_out <= d_in + outside_bias_margin) | guard)
    inside_unknown  = unknown & (~outside_unknown)

    # 6) 生成符号：未知区用竞争结果；窄带内保持原可信符号
    sign = np.zeros_like(sdf, float)
    sign[outside_unknown] = +1.0
    sign[inside_unknown]  = -1.0
    sign[known_pos] = +1.0
    sign[known_neg] = -1.0

    # 可能还有极少未被覆盖的位置（比如没有任何种子时），做一个外部为正的兜底
    unset = (sign == 0.0)
    if unset.any():
        sign[unset] = +1.0
    for i, l in enumerate(sign):

        cv2.imwrite(f'src/tutils/fix_sdf/fix_sdf_debug_{i}.png', l.astype(np.int8) * 255.)
    # 7) 只改符号，不改幅值
    return sign * np.abs(sdf)


def wzj_fix_sdf(sdf):
    neg = sdf < 0
    for i in range(neg.shape[0]):
        for j in range(neg.shape[2]):
            vec = neg[i, :, j]
            # print(vec.shape)
            start = np.argmax(vec)
            end = -np.argmax(vec[::-1])
            # print(start, end, sep=' ')
            if start == end == 0:
                continue
            c = np.cumsum(~vec[start:end], axis=-1, dtype=np.int32)
            z = c - np.maximum.accumulate(np.where(~vec[start:end], 0, c), axis=-1)

            length = z.max(axis=-1)
            end1 = z.argmax(axis=-1)
            start1 = end - length + 1

            vec[start:end][start1:end1] = True

    
    for i, l in enumerate(neg):

        cv2.imwrite(f'src/tutils/fix_sdf/fix_sdf_pro_{i}.png', l.astype(np.int8) * 255.)
    sign = np.zeros_like(sdf)
    sign[neg] = -1
    sign[~neg] = 1
    return np.abs(sdf) * sign


def cad_style_fill_sign_only(
    sdf,
    band_tau=99,             # 你的“已知窄带”半宽 τ（|sdf|<=τ）
    gap_tol_vox=2.0,          # 缺口容差 r（体素为单位）；若有实际尺度，用 gap_tol_mm / spacing
    spacing=(1.0,1.0,1.0),    # 体素物理尺寸 (dz, dy, dx)，用来把毫米换成体素
    outside_is_positive=True, # 你的符号约定：外正内负；若相反改 False
    connectivity=1            # 6 邻接：防过度连通
):
    # 1) 已知窄带 & 仅窄带内可信符号
    known_band = np.abs(sdf) <= band_tau
    known_pos  = (sdf >  0) & known_band
    known_neg  = (sdf <  0) & known_band

    # 2) 用 EDT 做“球形半径 r 的厚化 + 封洞”（CAD 的 offset/closing 等价物）
    #    wall = 距离窄带 <= r 的区域（相当于 band 膨胀 r）；再把 wall 的“内空洞”填掉。
    r_phys = gap_tol_vox * min(spacing)  # 以最小体素边长换算
    # 到窄带的欧氏距离
    d_to_band = ndi.distance_transform_edt(~known_neg, sampling=spacing)
    wall = d_to_band <= r_phys
    # 填掉 wall 内与外边界不连通的空腔（真正的“洞”）
    space = ~wall
    struct = ndi.generate_binary_structure(3, connectivity)
    labels, _ = ndi.label(space, structure=struct)
    border = np.zeros_like(space, bool)
    border[0,:,:]=border[-1,:,:]=border[:,0,:]=border[:,-1,:]=border[:,:,0]=border[:,:,-1]=True
    outside_lbls = np.unique(labels[border])
    holes_mask = (labels != 0) & (~np.isin(labels, outside_lbls))
    wall[holes_mask] = True

    # 3) 只在 ~wall（厚壳之外的空域）里，从体素外边界做外部传播
    allowed = ~wall
    seed = np.zeros_like(allowed, bool)
    seed[0,:,:]=seed[-1,:,:]=seed[:,0,:]=seed[:,-1,:]=seed[:,:,0]=seed[:,:,-1]=True
    seed &= allowed
    outside_unknown = ndi.binary_propagation(seed, mask=allowed, structure=struct)
    inside_unknown  = allowed & (~outside_unknown)  # 被厚壳围住的都视为内部

    # 4) 组合符号：窄带内保留原符号；窄带外按 inside/outside 指派
    sign = np.zeros_like(sdf, float)
    if outside_is_positive:
        sign[outside_unknown] = +1.0
        sign[inside_unknown]  = -1.0
        sign[known_pos] = +1.0
        sign[known_neg] = -1.0
    else:
        sign[outside_unknown] = -1.0
        sign[inside_unknown]  = +1.0
        sign[known_pos] = -1.0
        sign[known_neg] = +1.0

    # 5) 只改符号，不改幅值
    sdf_fixed = sign * np.abs(sdf)
    test = sdf_fixed > 0
    for i, l in enumerate(test):

        cv2.imwrite(f'src/tutils/fix_sdf/fix_sdf_pro_{i}.png', l.astype(np.int8) * 255.)
    return sdf_fixed


def fill_enclosed_2d(
    wall: np.ndarray,
    r: int = 4,                 # 缺口容差（像素）：能“补上”<= r 的洞
    connectivity: int = 1,      # 4 邻接(1) 或 8 邻接(2)
    fill_small_only: bool = False,
    min_hole_area: int = 0
):
    """
    参数:
      wall: 2D bool，True 表示墙（不可穿越的边界带）
      r: 膨胀次数，用于虚拟补洞（相当于 CAD 里的公差/刀具半径）
      connectivity: 1=4 邻接，2=8 邻接
      fill_small_only: 仅把“面积 <= min_hole_area”的洞填上（更保守的选项）
      min_hole_area: 面积阈值（像素）
    返回:
      filled: 2D bool，True 表示“墙 + 被墙包围并填充”的区域
      inside: 2D bool，仅被墙包围的内部（不含墙本身）
    """
    assert wall.ndim == 2 and wall.dtype == bool
    print(wall.shape)
    struct = ndi.generate_binary_structure(2, connectivity)

    # 1) 把墙“加厚” r 像素，封住 <= r 的缺口
    thick = ndi.binary_dilation(wall, structure=struct, iterations=r)

    # 2) 在“非厚墙区域”中找连通域，并识别与边界连通的“外部”
    space = ~thick
    labels, n = ndi.label(space, structure=struct)

    border = np.zeros_like(space, dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True

    outside_labels = np.unique(labels[border])
    outside = np.isin(labels, outside_labels)

    # 3) 候选“内部洞” = 不是外部的空域
    holes = space & (~outside)

    if fill_small_only and n > 0:
        # 只填小洞：按面积阈值筛选
        areas = np.bincount(labels.ravel())
        hole_ids = np.where((np.arange(n + 1) > 0) & (areas <= min_hole_area))[0]
        holes = holes & np.isin(labels, hole_ids)

    # 4) 输出
    inside = holes            # 被墙包围的区域
    filled = wall | inside    # 墙 + 内部
    return filled, inside


def fill_enclosed_2d(wall: np.ndarray, r: int = 2, connectivity: int = 1):
    assert wall.ndim == 2 and wall.dtype == bool
    struct = ndi.generate_binary_structure(2, connectivity)

    # 1) 仅用于传播的“厚墙”（封 ≤r 的洞）
    thick = ndi.binary_dilation(wall, structure=struct, iterations=r)

    # 2) 在 ~thick 里标外部
    space = ~thick
    labels, _ = ndi.label(space, structure=struct)

    border = np.zeros_like(space, bool)
    border[0,:] = border[-1,:] = True
    border[:,0] = border[:,-1] = True

    outside_space_labels = np.unique(labels[border])
    outside_space = np.isin(labels, outside_space_labels)

    # ★ 关键：把 outside 扩展回全图坐标
    outside_full = np.zeros_like(wall, bool)
    outside_full[space] = outside_space     # thick 区域保持 False（非外部）

    # 3) 内部 = 既不是外部，也不属于原始墙
    inside = (~outside_full) & (~wall)

    # 4) 最终填充 = 原始墙 ∪ 内部（不会留“护城河”）
    filled = wall | inside
    cv2.imwrite('src/tutils/utils_geo_.png', filled[..., None].astype(np.int8) * 255.)

    return filled, inside

def fill_enclosed_2d(wall: np.ndarray, r: int = 0, connectivity: int = 1):
    assert wall.ndim == 2 and wall.dtype == bool
    struct = ndi.generate_binary_structure(2, connectivity)

    # 1) 仅用于传播的“厚墙”（封 ≤r 像素的缺口）
    thick = ndi.binary_dilation(wall, structure=struct, iterations=r)

    # 2) 在 ~thick 里标记外部
    space = ~thick
    labels, _ = ndi.label(space, structure=struct)

    border = np.zeros_like(space, bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True

    outside_lbls = np.unique(labels[border])
    outside_lbls = outside_lbls[outside_lbls != 0]     # 去掉背景0
    outside_space = np.isin(labels, outside_lbls)

    # ★ 修正处：把 outside 限制/还原到全图
    outside_full = outside_space & space                # 或用 outside_full[space] = outside_space[space]

    # 3) 内部 = 非外部 且 非原始墙
    inside = (~outside_full) & (~wall)

    # 4) 最终填充
    filled = wall | inside
    cv2.imwrite('src/tutils/utils_geo_.png', filled[..., None].astype(np.int8) * 255.)

    return filled, inside

def fill_enclosed_2d_no_expand(wall: np.ndarray, r: int = 8, connectivity: int = 1):
    """
    wall: 2D bool, True=墙/边界带
    r: 补洞强度（迭代次数/大致像素半径），越大越能跨过更宽的缺口
    connectivity: 1=4邻接, 2=8邻接（传播与闭运算都用同一连通性）
    返回:
      filled: 墙 ∪ 内部
      inside: 仅内部区域（不含墙）
    """
    assert wall.ndim == 2 and wall.dtype == bool
    struct = ndi.generate_binary_structure(2, connectivity)

    # 1) 在“可穿越区域”里补洞：对 ~wall 做 closing（先膨胀再腐蚀）
    comp = ~wall
    comp_closed = ndi.binary_closing(comp, structure=struct, iterations=r)

    # 2) 在补洞后的可穿越区域里找“外部”（与图像边界连通的部分）
    labels, _ = ndi.label(comp_closed, structure=struct)
    border = np.zeros_like(comp_closed, bool)
    border[0,:]=border[-1,:]=True; border[:,0]=border[:,-1]=True
    outside_lbls = np.unique(labels[border])
    outside_closed = np.isin(labels, outside_lbls)

    # 3) 把“外部”限制回原始可穿越区域 comp，避免向外扩到 wall 上
    outside = outside_closed & comp

    # 4) 内部 = 原始 comp 中 不是外部 的部分；最终结果 = 墙 ∪ 内部
    inside = comp & (~outside)
    filled = wall | inside
    cv2.imwrite('src/tutils/utils_geo_.png', filled[..., None].astype(np.int8) * 255.)

    return filled, inside


def flood_fill_ndimage(mask, seed_point, connectivity=1):
    """
    简洁版 flood fill：
    在 mask=True 的区域内，从 seed_point 出发做连通传播。

    参数:
      mask : 2D bool 数组，True 表示允许传播的区域
      seed_point : (y, x) 起始点坐标
      connectivity : 1=4 邻接, 2=8 邻接
    返回:
      filled : 2D bool 数组，True 表示从 seed 可达的区域
    """
    filled = np.zeros_like(mask, dtype=bool)
    filled[seed_point] = True

    struct = ndi.generate_binary_structure(2, connectivity)
    filled = ndi.binary_propagation(filled, mask=mask, structure=struct)
    l = ~filled
    out = ((mask) & (l > 0))
    return out

def wzj_final(sdf):
    mask = sdf >= 0
    xy = mask.copy()
    yz = mask.copy()
    xz = mask.copy()
    seed = [0, 0]
    for i in range(257):
        xy[i] = flood_fill_ndimage(xy[i], seed)
        yz[:, i] = flood_fill_ndimage(yz[:, i], seed)
        xz[..., i] = flood_fill_ndimage(xz[..., i], seed)
    out = xy | yz | xz
    return out
    # for i, img in enumerate(out):
    #     cv2.imwrite(f'src/tutils/fix_sdf/fix_sdf_pro_{i}.png', img.astype(np.int8) * 255.)

def smplx_test():
    import torch
    from models.smplx.smplx import SMPLX

    device = 'cuda:0'
    def to_device(*args, **kwargs):
        func = lambda x: (x.to(device) if isinstance(x, torch.Tensor) else torch.from_numpy(x).float().to(device) if isinstance(x, np.ndarray) else x)
        # print(kwargs)
        # exit()
        if kwargs:
            return list(map(func, kwargs.values()))
    num_betas = 300
    num_expression_coeffs = 100
    model_path = 'models/smplx/SMPLX2020'
    smplx = SMPLX(num_betas=num_betas, num_expression_coeffs=num_expression_coeffs, model_path=model_path)
    smplx.to(device)
    mname = 'test_pink'
    track_path = f'/home/wzj/project/TRELLIS/dataset/{mname}/body_track/smplx_track.pth'
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
    import trimesh
    from glob import glob
    ll = glob('src/geo_utils/*.npy')
    ll = [np.load(l) for l in ll]
    sub_tris = np.concatenate(ll).astype(np.int32)
    v, f = densify(smplx_v[0], smplx.faces)
    m = trimesh.Trimesh(vertices=v, faces=f)
    sub = m.submesh([sub_tris])[0]
    sub.export('src/geo_utils/region.obj')
    # v, f = densify(smplx_v[0], smplx.faces)

def flood_fill_ndimage(mask, seed_point, connectivity=1):
    """
    简洁版 flood fill：
    在 mask=True 的区域内，从 seed_point 出发做连通传播。

    参数:
      mask : 2D bool 数组，True 表示允许传播的区域
      seed_point : (y, x) 起始点坐标
      connectivity : 1=4 邻接, 2=8 邻接
    返回:
      filled : 2D bool 数组，True 表示从 seed 可达的区域
    """
    filled = np.zeros_like(mask, dtype=bool)
    filled[seed_point] = True

    struct = ndi.generate_binary_structure(2, connectivity)
    filled = ndi.binary_propagation(filled, mask=mask, structure=struct)
    l = ~filled
    out = ((mask) & (l > 0))
    return out

def wzj_final(sdf):
    mask = sdf >= 0
    xy = mask.copy()
    yz = mask.copy()
    xz = mask.copy()
    seed = [0, 0]
    for i in range(257):
        xy[i] = flood_fill_ndimage(xy[i], seed)
        yz[:, i] = flood_fill_ndimage(yz[:, i], seed)
        xz[..., i] = flood_fill_ndimage(xz[..., i], seed)
    out = xy | yz | xz
    return out

def myicp(name):
    source = f'outputs/{name}/objects/sample_outer_filtered.obj'
    target = f'outputs/{name}/objects/smplx.obj'

    import trimesh

    source_mesh = trimesh.load(source)
    target_mesh = trimesh.load(target)

    

if __name__ == '__main__':
    smplx_test()
    exit()
    img = cv2.imread('/home/wzj/project/TRELLIS/src/tutils/fix_sdf/fix_sdf_pos_87.png')
    if img.shape[-1] < 5:
        wall = img[..., 0] > 0
        mask = img[..., 0] == 0
    filled = flood_fill_ndimage(mask, [0, 0])
    print(filled.sum())
    cv2.imwrite('src/tutils/utils_geo_.png', filled[..., None].astype(np.int8) * 255.)

    # ff, i = fill_enclosed_2d(wall)
    # m = ~filled
    # m[wall] = False
    # cv2.imwrite('src/tutils/utils_geo_1.png', m[..., None].astype(np.int8) * 255.)
