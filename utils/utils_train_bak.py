from math import exp
import torch
from torch.autograd import Variable
from lpips import LPIPS
import torch.nn.functional as F
# from lpips import LPIPS
import numpy as np

loss_fn_vgg = None

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    import torch.nn.functional as F
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
        
def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def l1_loss(network_output, gt, type=None):
    if type == 'color':
        # sRGB -> linear
        # lin_pred.clamp_(0, 1)
        network_output.clamp_(0, 1)
        lin_gt = torch.where(gt <= 0.04045, gt/12.92, ((gt+0.055)/1.055)**2.4)
        lin_pred = torch.where(network_output <= 0.04045, network_output / 12.92, ((network_output + 0.055) / 1.055) ** 2.4)
    else:
        lin_pred = network_output
        lin_gt = gt
        delta_value = 0.1
        huber_loss = torch.nn.functional.huber_loss(
            input=lin_pred, 
            target=lin_gt, 
            reduction='mean', 
            delta=delta_value
        )
    return huber_loss + 0.3 * (1 - ssim(gt, pred))


def lpips(img1, img2, value_range=(0, 1)):
    global loss_fn_vgg
    if loss_fn_vgg is None:
        loss_fn_vgg = LPIPS(net='vgg').cuda().eval()
    # normalize to [-1, 1]
    img1 = (img1 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    img2 = (img2 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    return loss_fn_vgg(img1, img2).mean()

def mesh2smplx(mname, mesh=None, return_=False, device='cuda:0'):
    from utils3d.torch import quaternion_to_matrix

    trans = np.load(f'outputs_track/{mname}/params/trans.npz')
    kt = np.load(f'outputs_track/{mname}/params/kt.npz')
    k, t = list(kt.values())
    s, R, T = list(trans.values())
    s = torch.from_numpy(s).to(device)
    R = quaternion_to_matrix(torch.from_numpy(R).to(device))
    T = torch.from_numpy(T).to(device)
    k = torch.from_numpy(k).to(device)
    t = torch.from_numpy(t).to(device)
    if mesh is not None:
        mesh.vertices = (mesh.vertices - T) @ R[0] / s[0]
        mesh.vertices = (mesh.vertices - t) / k
        mesh.face_normal = torch.matmul(mesh.face_normal, R)


    if return_:
        return s, R, T, k, t
    

def loss_recon(gt, pred):
    return l1_loss(gt, pred) + 0.3 * (1 - ssim(gt, pred))


def dice_loss(pred, target, eps=1e-6):
    """
    pred: (N, H, W)  raw logits
    target:      (N, H, W)  binary mask 0/1
    """
    # pred = torch.sigmoid(pred_logits)
    
    intersection = (pred * target).sum(dim=(-1,-2))
    union = pred.sum(dim=(-1,-2)) + target.sum(dim=(-1,-2))
    
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()