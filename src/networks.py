import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from trellis.modules import sparse as sp
import torch
import torch.nn as nn
from trellis.modules.norm import LayerNorm32
from typing import List
import torch.nn.functional as F
import numpy as np
from trellis.modules.transformer import AbsolutePositionEmbedder
from trellis.modules.sparse.transformer import ModulatedSparseTransformerCrossBlockNoT
from trellis.modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
import time



# def zero_module(module):
#     """
#     Zero out the parameters of a module and return it.
#     """
#     for p in module.parameters():
#         p.detach().zero_()
#     return module

class SparseSubdivideBlock3d(nn.Module):
    """
    A 3D subdivide block that can subdivide the sparse tensor.

    Args:
        channels: channels in the inputs and outputs.
        out_channels: if specified, the number of output channels.
        num_groups: the number of groups for the group norm.
    """
    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: None,
        # num_groups: int = 32,
        num_groups: int = 8
    ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.out_resolution = resolution * 2
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels),
            sp.SparseSiLU()
        )
        
        self.sub = sp.SparseSubdivide()
        
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}"),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}")),
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(channels, self.out_channels, 1, indice_key=f"res_{self.out_resolution}")
        
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: an [N x C x ...] Tensor of features.
        Returns:
            an [N x C x ...] Tensor of outputs.
        """
        h = self.act_layers(x)
        h = self.sub(h)
        x = self.sub(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)
        return h

class SparseResBlock3d(nn.Module):
    def __init__(
            self,
            channels,
            out_channels,
            downsample: bool = False,
            upsample: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.upsample = upsample
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = None
        if self.downsample:
            self.updown = sp.SparseDownsample(2)
        elif self.upsample:
            self.updown = sp.SparseUpsample(2)
    
    def _updown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.updown is not None:
            x = self.updown(x)
        return x

    def forward(self, x):
        x = self._updown(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)

        return h

class myNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 8,
            model_channels: int = 16,
            cond_channels: int = 512,
            zero_output=True
    ):
        self.zero_output = zero_output
        super().__init__()
        self.num_heads = 8
        self.mlp_ratio = 4
        self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        self.input_layer = sp.SparseLinear(in_channels, model_channels)

        self.input_blocks = nn.ModuleList([
            SparseResBlock3d(
                        model_channels,
                        # model_channels,
                        out_channels=model_channels,
                    ),
            SparseResBlock3d(
                model_channels,
                out_channels=model_channels,
                downsample=True,
            )
        ])
        # self.blocks = nn.ModuleList([
        #     ModulatedSparseTransformerCrossBlockNoT(
        #         model_channels,
        #         cond_channels,
        #         num_heads=self.num_heads,
        #         mlp_ratio=self.mlp_ratio,
        #         attn_mode='full',
        #         use_checkpoint=False,
        #         use_rope=False,
        #         qk_rms_norm=True,
        #         qk_rms_norm_cross=True,
        #     )
        #     for _ in range(1)
        # ])
        self.out_blocks = nn.ModuleList([
            SparseResBlock3d(
                model_channels * 2,
                out_channels=model_channels,
                upsample=True,
            ),
            SparseResBlock3d(
                model_channels * 2,
                # model_channels,
                out_channels=model_channels,
            )
        ])
        # self.out_layer = sp.SparseLinear(model_channels, 101)
        self.upsample = nn.ModuleList([
            SparseSubdivideBlock3d(
                channels=model_channels,
                resolution=64,
                out_channels=model_channels
            ),
            SparseSubdivideBlock3d(
                channels=model_channels,
                resolution=128,
                out_channels=model_channels
            )
        ])
        self.out_layer = sp.SparseLinear(model_channels, 32)
        self.dtype = torch.float16
        self.initialize_weights()

        self.convert_to_fp16()
        self.zero_last()

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        # self.input_blocks.apply(convert_module_to_f16)
        # self.blocks.apply(convert_module_to_f16)
        # self.out_blocks.apply(convert_module_to_f16)
        self.upsample.apply(convert_module_to_f16)
        # self.out_layer.apply(convert_module_to_f16)
    
    def zero_last(self):
        # ...
        if self.zero_output:
            self.out_layer = zero_module(self.out_layer)


    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        # nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        # if self.share_mod:
        #     nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        # else:
        #     for block in self.blocks:
        #         nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #         nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        # nn.init.constant_(self.out_layer.weight, 0)
        # nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x, cond=None):
        if cond is not None:
            cond = cond.reshape((1, 1, cond.shape[-1]))
            batch_size = x.shape[0]
            N = x.feats.shape[0]
            cond = cond.expand((batch_size, N, cond.shape[-1]))
            # cond = F.layer_norm(cond, cond.shape[-1:])
            # h = x.type(self.dtype)
            cond = cond.type(self.dtype)
        h = self.input_layer(x)
        skips = []
        # pack with input blocks
        for block in self.input_blocks:
            h = block(h)
            skips.append(h.feats)
        h = h + self.pos_embedder(h.coords[:, 1:])
        h = h.type(self.dtype)
        # temp = torch.zeros((1, 6, 1)).to(device)
        # temp[:, 2, :] = 1.
        # temp[:, 5, :] = 1.
        # for block in self.blocks:
        #     h = block(h, cond)

        # h = h.type(x.dtype)
        for block, skip in zip(self.out_blocks, reversed(skips)):
            h = block(h.replace(torch.cat([h.feats, skip], dim=1)))

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:])).type(self.dtype)
        # h = self.out_layer(h.type(x.dtype))
        for block in self.upsample:
            h = block(h)
        h = h.type(x.dtype)
        h = self.out_layer(h)
        return h

device = 'cuda:0'

if __name__ == '__main__':
    x = torch.randint(0, 256, (10000, ))
    y = torch.randint(0, 256, (10000, ))
    z = torch.randint(0, 256, (10000, ))
    coords = torch.stack([x, y, z], dim=-1)
    coords = torch.cat([torch.zeros((coords.shape[0], 1)), coords], dim=-1)
    coords = coords.to(device).int()
    feats = torch.randn(10000, 8).to(device, torch.float32)
    print(coords.shape)
    print(feats.shape)
    input = sp.SparseTensor(
        coords = coords,
        feats = feats
    ).to(device)
    # the dimension of input is (N, 8), is the coarse features of trellis
    test = myNet(in_channels=8)
    cond = torch.randn(1, 10000, 512).to(device)
    test.to(device)
    print(test(input).feats.shape)

    while True:
        time.sleep(1)
    #TODO
    # input is coarse features, and output is delta of fine feature
    # so need a upsample block
    # and add the final out to fine features 



    # print(test(input).feats.shape)
    # print(test(input).data)
    # print(test(input).data.shape)

