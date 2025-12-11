from dataclasses import dataclass, fields
from typing import Literal, Optional, List
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from einops import rearrange
import numpy as np
from scipy.spatial.transform import Rotation

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .visualization.encoder_visualizer_depthsplat_cfg import EncoderVisualizerDepthSplatCfg
from .unimatch.mv_unimatch import MultiViewUniMatch
from .unimatch.dpt_head import DPTHead

@dataclass
class EncoderDepthSplatCfg:
    name: Literal["depthsplat"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerDepthSplatCfg
    gaussian_adapter: GaussianAdapterCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]

    # mv_unimatch
    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool

    # depthsplat color branch
    large_gaussian_head: bool
    color_large_unet: bool
    init_sh_input_img: bool
    feature_upsampler_channels: int
    gaussian_regressor_channels: int

    # loss config
    supervise_intermediate_depth: bool
    return_depth: bool

    # only depth
    train_depth_only: bool

    # monodepth config
    monodepth_vit_type: str

    # multi-view matching
    local_mv_match: int


class EncoderDepthSplat(Encoder[EncoderDepthSplatCfg]):
    def __init__(self, cfg: EncoderDepthSplatCfg) -> None:
        super().__init__(cfg)

        self.depth_predictor = MultiViewUniMatch(
            num_scales=cfg.num_scales,
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            vit_type=cfg.monodepth_vit_type,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
        )

        if self.cfg.train_depth_only:
            return

        # upsample features to the original resolution
        model_configs = {
            'vits': {'in_channels': 384, 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'in_channels': 768, 'features': 96, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'in_channels': 1024, 'features': 128, 'out_channels': [128, 256, 512, 1024]},
        }

        self.feature_upsampler = DPTHead(**model_configs[cfg.monodepth_vit_type],
                                        downsample_factor=cfg.upsample_factor,
                                        return_feature=True,
                                        num_scales=cfg.num_scales,
                                        )
        feature_upsampler_channels = model_configs[cfg.monodepth_vit_type]["features"]
        
        # gaussians adapter
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # concat(img, depth, match_prob, features)
        in_channels = 3 + 1 + 1 + feature_upsampler_channels
        channels = self.cfg.gaussian_regressor_channels

        # conv regressor
        modules = [
                    nn.Conv2d(in_channels, channels, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, 1, 1),
                ]

        self.gaussian_regressor = nn.Sequential(*modules)

        # predict gaussian parameters: scale, q, sh, offset, opacity
        num_gaussian_parameters = self.gaussian_adapter.d_in + 2 + 1

        # concat(img, features, regressor_out, match_prob)
        in_channels = 3 + feature_upsampler_channels + channels + 1
        self.gaussian_head = nn.Sequential(
                nn.Conv2d(in_channels, num_gaussian_parameters,
                          3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters,
                          num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
            )

        if self.cfg.init_sh_input_img:
            nn.init.zeros_(self.gaussian_head[-1].weight[10:])
            nn.init.zeros_(self.gaussian_head[-1].bias[10:])

        # init scale
        # first 3: opacity, offset_xy
        nn.init.zeros_(self.gaussian_head[-1].weight[3:6])
        nn.init.zeros_(self.gaussian_head[-1].bias[3:6])

    def predict(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        output_view_idx: int = None
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        
        # [B, V, C, H, W]
        
        if v > 3:
            with torch.no_grad():
                xyzs = context["extrinsics"][:, :, :3, -1].detach()
                cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                cameras_dist_index = torch.argsort(cameras_dist_matrix)

                cameras_dist_index = cameras_dist_index[:, :, :(self.cfg.local_mv_match + 1)]
        else:
            cameras_dist_index = None

        # depth prediction
        results_dict = self.depth_predictor(
            context["image"],
            attn_splits_list=[2],
            min_depth=1. / context["far"],
            max_depth=1. / context["near"],
            intrinsics=context["intrinsics"],
            extrinsics=context["extrinsics"],
            nn_matrix=cameras_dist_index,
        )

        depth_preds = results_dict['depth_preds']
        depth = depth_preds[-1] # [B, V, H, W]

        # === Early Return for Depth Only Training ===
        if self.cfg.train_depth_only:
            depths = rearrange(depth, "b v h w -> b v (h w) () ()")
            depths = rearrange(depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w).squeeze(-1).squeeze(-1)
            return {"gaussians": None, "depths": depths}

        # features
        features = self.feature_upsampler(results_dict["features_mono_intermediate"],
                                          cnn_features=results_dict["features_cnn_all_scales"][::-1],
                                          mv_features=results_dict["features_mv"][
                                          0] if self.cfg.num_scales == 1 else results_dict["features_mv"][::-1]
                                          )

        match_prob = results_dict['match_probs'][-1]
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[0]
        match_prob = F.interpolate(match_prob, size=depth.shape[-2:], mode='nearest')

        # unet input
        concat = torch.cat((
            rearrange(context["image"], "b v c h w -> (b v) c h w"),
            rearrange(depth, "b v h w -> (b v) () h w"),
            match_prob,
            features,
        ), dim=1)

        out = self.gaussian_regressor(concat)

        concat = [out, rearrange(context["image"], "b v c h w -> (b v) c h w"), features, match_prob]
        out = torch.cat(concat, dim=1)

        gaussians = self.gaussian_head(out)  
        # [BV, C, H, W] -> [B, V, C, H, W]
        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)
        
        # Prepare adapter input
        depths = rearrange(depth, "b v h w -> b v (h w) () ()")
        raw_gaussians = rearrange(gaussians, "b v c h w -> b v (h w) c")
        opacities = raw_gaussians[..., :1].sigmoid().unsqueeze(-1)
        raw_gaussians = raw_gaussians[..., 1:]
        
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians_reshaped = rearrange(raw_gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
        offset_xy = gaussians_reshaped[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        sh_input_images = context["image"]

        # Gaussian Adapter Forward -> Returns tensors with shape [B, V, N, C]
        gaussians_phys = self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            opacities,
            rearrange(gaussians_reshaped[..., 2:], "b v r srf c -> b v r srf () c"),
            (h, w),
            input_images=sh_input_images if self.cfg.init_sh_input_img else None,
        )

        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w)

        def get_tensor(name):
            val = getattr(gaussians_phys, name)
            if output_view_idx is not None:
                # [B, 1, N, C]
                val = val[:, output_view_idx:output_view_idx+1]
            return val

        # === Flatten Batch and View dimensions ===
        # Input: [B, V, N, C] -> Output: [B*V, N, C]
        gaussians_separated = Gaussians(
            rearrange(get_tensor('means'), "b v r srf spp xyz -> (b v) (r srf spp) xyz"),
            rearrange(get_tensor('covariances'), "b v r srf spp i j -> (b v) (r srf spp) i j"),
            rearrange(get_tensor('harmonics'), "b v r srf spp c d_sh -> (b v) (r srf spp) c d_sh"),
            rearrange(get_tensor('opacities'), "b v r srf spp -> (b v) (r srf spp)"),
            rearrange(get_tensor('scales'), "b v r srf spp xyz -> (b v) (r srf spp) xyz"),
            rearrange(get_tensor('rotations'), "b v r srf spp xyzw -> (b v) (r srf spp) xyzw"),
        )
        
        depths_ret = rearrange(depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w).squeeze(-1).squeeze(-1)
        
        return {"gaussians": gaussians_separated, "depths": depths_ret}

    def forward(self, context, global_step, deterministic=False, visualization_dump=None, scene_names=None, **kwargs):
        # Clean forward, just calls predict
        res = self.predict(context, global_step, deterministic, visualization_dump, scene_names)
        if self.cfg.return_depth:
            return res
        return res["gaussians"]

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(batch, patch_size=self.cfg.shim_patch_size * self.cfg.downscale_factor)
            return batch
        return data_shim

    @property
    def sampler(self):
        return None
