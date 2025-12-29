import os
import sys
from dataclasses import dataclass, fields
from typing import Literal, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation
from ...dataset.types import BatchedExample
from ..types import Gaussians
from .encoder_depthsplat import EncoderDepthSplat, EncoderDepthSplatCfg
from src.global_cfg import get_cfg
from ...misc.image_io import save_image

from .common.gaussians import build_covariance


class GeometricEvaluator:
    def __init__(self, device):
        self.device = device

    def filter_outliers_by_box(self, pred_pts, gt_pts, margin=0.1):
        if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
            return pred_pts
        gt_min = gt_pts.min(dim=0)[0]
        gt_max = gt_pts.max(dim=0)[0]
        span = gt_max - gt_min
        box_min = gt_min - span * margin
        box_max = gt_max + span * margin
        in_box_mask = (pred_pts >= box_min) & (pred_pts <= box_max)
        in_box_mask = in_box_mask.all(dim=1)
        filtered_pred = pred_pts[in_box_mask]
        if filtered_pred.shape[0] < 100:
            return pred_pts
        return filtered_pred

    def robust_chamfer_distance(self, p1, p2):
        """
        p1: Prediction [N, 3]
        p2: Ground Truth [M, 3]
        """
        N_max = 5000
        if p1.shape[0] > N_max: p1 = p1[torch.randperm(p1.shape[0])[:N_max]]
        if p2.shape[0] > N_max: p2 = p2[torch.randperm(p2.shape[0])[:N_max]]
        
        # 计算距离矩阵 (Euclidean)
        dists = torch.cdist(p1, p2, p=2) 
        
        min_dist_pred, _ = torch.min(dists, dim=1) 
        min_dist_gt, _ = torch.min(dists, dim=0)
        
        chamfer_dist = torch.mean(min_dist_pred) + torch.mean(min_dist_gt)
        
        # [改进 2] 动态计算 F-Score 阈值
        # 策略：计算 GT 点云的包围盒对角线长度，取其 1% 或 5% 作为阈值
        # 这能适应任何尺度的场景 (无论是 1.0 还是 100.0)
        gt_span = p2.max(dim=0)[0] - p2.min(dim=0)[0]
        scene_scale = torch.norm(gt_span).item() # 对角线长度
        
        # 常用标准：Scene Scale 的 1% (严格) 或 5% (宽松)
        # 鉴于 Unimatch 是稠密预测，我们用 2% 作为一个中肯标准
        threshold = 0.01 * scene_scale 
        
        precision = (min_dist_pred < threshold).float().mean()
        recall = (min_dist_gt < threshold).float().mean()
        f_score = 2 * precision * recall / (precision + recall + 1e-8)
        
        # 返回 Threshold 方便 debug
        return chamfer_dist.item(), f_score.item(), threshold

    def evaluate(self, step_name, pred_depth, gt_depth, pred_means, gt_means, valid_mask):
        metrics = {}
        
        # --- 2D Metrics ---
        # 增加极小值保护，防止 AbsRel 算除法时爆炸
        valid_mask = valid_mask & (gt_depth > 1e-3) & (pred_depth > 1e-3)
        
        p = pred_depth[valid_mask]
        g = gt_depth[valid_mask]
        
        if len(p) > 100:
            abs_rel = (torch.abs(p - g) / g).mean().item()
            thresh = torch.max((g / p), (p / g))
            acc_125 = (thresh < 1.25).float().mean().item()
            metrics['AbsRel'] = abs_rel
            metrics['Acc1.25'] = acc_125
        else:
            metrics['AbsRel'] = 1.0
            metrics['Acc1.25'] = 0.0

        # --- 3D Metrics ---
        clean_pred_means = self.filter_outliers_by_box(pred_means, gt_means)
        
        # 获取 metrics 和 动态计算的 threshold
        cd, f_score, adaptive_thresh = self.robust_chamfer_distance(clean_pred_means, gt_means)
        metrics['Chamfer'] = cd
        metrics['F-Score'] = f_score

        print(f"  [EVAL @ {step_name}] AbsRel: {metrics['AbsRel']:.3f} | Acc1.25: {metrics['Acc1.25']:.3f} | "
              f"CD: {metrics['Chamfer']:.3f} | F-Score(@{adaptive_thresh:.2f}): {metrics['F-Score']:.3f}")
        
        return metrics

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

from sklearn.linear_model import RANSACRegressor

def align_depth_ransac_with_anchor(mono, sparse, mask):
    # Flatten arrays
    X = mono[mask].cpu().numpy().reshape(-1, 1)
    y = sparse[mask].cpu().numpy().reshape(-1, 1)
    
    if len(X) < 50: return mono, 1.0, 0.0
    
    ransac = RANSACRegressor(min_samples=50, residual_threshold=0.5)
    ransac.fit(X, y)
    
    scale = ransac.estimator_.coef_[0][0]
    shift = ransac.estimator_.intercept_[0]

    sparse_med = torch.median(sparse)
    bias = 0.02 * sparse_med 
    shift += bias  # 深度值变大 = 离相机更远
    
    # 截断。防止小基线导致误差
    if scale < 0.80 or scale > 1.20:
        print(f"  [Align Warning] Scale {scale:.4f} out of bounds (0.85-1.15). Fallback to Identity.")
        return mono, 1.0, 0.0 
        
    return mono * scale + shift, scale, shift

def align_depth_median(mono_depth, sparse_depth, valid_mask):
    """
    使用 MAD (Median Absolute Deviation) 进行鲁棒的深度对齐 (Scale & Shift)。
    公式: Depth_aligned = scale * Depth_mono + shift
    """
    eps = 1e-6
    # 1. 过滤无效值 (确保深度为正)
    valid_mask = valid_mask & (sparse_depth > eps) & (mono_depth > eps)
    t_mono = mono_depth[valid_mask]
    t_sparse = sparse_depth[valid_mask]

    # 如果有效点太少，直接返回原图
    if t_mono.shape[0] < 50: 
        return mono_depth, 1.0, 0.0

    # 2. 计算中位数 (Center)
    mono_med = torch.median(t_mono)
    sparse_med = torch.median(t_sparse)

    # 3. 计算绝对中位差 MAD (Spread/Contrast)
    # MAD = Median(|X - Median(X)|)
    mono_mad = torch.median(torch.abs(t_mono - mono_med))
    sparse_mad = torch.median(torch.abs(t_sparse - sparse_med))

    # 4. 计算 Scale (基于分布的离散程度比率)
    if mono_mad < eps:
        scale = 1.0
    else:
        scale = (sparse_mad / mono_mad).item()

    # 5. 计算 Shift (基于缩放后的中心对齐)
    # Shift = Median_GT - Scale * Median_Mono
    shift = (sparse_med - scale * mono_med).item()

    # bias = 0.02 * sparse_med 
    # shift += bias  # 深度值变大 = 离相机更远

    # 6. 应用变换 (直接在深度空间对齐)
    aligned_depth = mono_depth * scale + shift

    # 7. 防止出现负数或极端值
    # aligned_depth = torch.clamp(aligned_depth, min=0.01, max=100.0)

    return aligned_depth, scale, shift

def visualize_depth(depth_tensor, cmap_name='magma'):
    if depth_tensor.dim() == 3: depth_tensor = depth_tensor.squeeze(0)
    depth_np = depth_tensor.detach().cpu().numpy()
    valid_mask = depth_np > 0.01
    if valid_mask.sum() > 0:
        d_min = depth_np[valid_mask].min()
        d_max = depth_np[valid_mask].max()
        depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-6)
        depth_norm[~valid_mask] = 0
    else: depth_norm = depth_np
    cmap = cm.get_cmap(cmap_name)
    depth_colored = cmap(depth_norm)[:, :, :3] 
    return torch.from_numpy(depth_colored).permute(2, 0, 1).float()

# -----------------------------------------------------------------------------
# Gaussian TTO Optimizer (Multi-Stage Strategy)
# -----------------------------------------------------------------------------

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def inverse_sigmoid(x):
    return torch.log(x / (1 - x + 1e-6))

class GaussianTTOOptimizer(nn.Module):
    def __init__(self, 
                 steps_stage1=50,
                 steps_stage2=30, 
                 lr_opa_s2=0.08,
                 steps_stage3=50,
                 lr_pos_s3=0.00016,         
                 lr_scale_s3=0.001,
                 lr_opa_s3=0.08,
                 lr_rot_s3=0.001,
                 max_scale=0.015,
                 context_view_sample_k=4
                 ):
        super().__init__()
        self.steps_stage1 = steps_stage1
        self.steps_stage2 = steps_stage2
        self.steps_stage3 = steps_stage3
        self.lr_opa_s2 = lr_opa_s2
        self.lr_pos_s3 = lr_pos_s3
        self.lr_opa_s3 = lr_opa_s3
        self.lr_scale_s3 = lr_scale_s3
        self.lr_rot_s3 = lr_rot_s3
        self.max_scale = max_scale
        self.context_view_sample_k = context_view_sample_k

    def forward(self,
                decoder,
                target_gaussians: Gaussians, 
                context_gaussians: Gaussians, 
                context_render_pkg: dict,
                target_render_pkg: dict,
                hole_tags: torch.Tensor = None,
                stereo_anchor_idx: int = None,
                monster_indices: torch.Tensor = None):
        
        device = target_gaussians.means.device
        
        # ... (Evaluator 和 GT 准备代码，保持不变) ...
        evaluator = GeometricEvaluator(device)
        with torch.no_grad():
            flat_context_gaussians = self.flatten_gaussians(context_gaussians)
            gt_render_out = decoder(context_gaussians, target_render_pkg['extrinsics'], target_render_pkg['intrinsics'], target_render_pkg['near'], target_render_pkg['far'], (target_render_pkg['h'], target_render_pkg['w']), depth_mode="depth")
            gt_scene_depth = gt_render_out.depth[0].squeeze(1) 
            gt_valid_mask = gt_render_out.alpha[0].squeeze(0) > 0.5 
            gt_scene_means = context_gaussians.means.squeeze(0)

        def run_eval(step_label, current_gaussians):
            with torch.no_grad():
                curr_out = decoder(current_gaussians, target_render_pkg['extrinsics'], target_render_pkg['intrinsics'], target_render_pkg['near'], target_render_pkg['far'], (target_render_pkg['h'], target_render_pkg['w']), depth_mode="depth")
                evaluator.evaluate(step_label, curr_out.depth[0].squeeze(1), gt_scene_depth, current_gaussians.means.squeeze(0), gt_scene_means, gt_valid_mask)

        # --------------------------------------------------------------------------
        # Eval-Before
        # --------------------------------------------------------------------------
        run_eval("Pre-Stage1 (Raw Encoder)", target_gaussians)

        # --------------------------------------------------------------------------
        # 准备 Stage 1 变量
        # --------------------------------------------------------------------------
        with torch.no_grad():
            target_c2w = target_render_pkg['extrinsics'][0, 0].to(device)
            cam_center = target_c2w[:3, 3]
            c2w_rot = target_c2w[:3, :3]
            
            init_means = target_gaussians.means.clone().detach().squeeze(0) 
            ray_vectors = init_means - cam_center 
            init_dists_norm = torch.norm(ray_vectors, dim=-1, keepdim=True) 
            ray_dirs = ray_vectors / (init_dists_norm + 1e-8)
            
            init_scales = torch.clamp(target_gaussians.scales.clone().detach(), max=self.max_scale)
            if init_scales.dim() == 3: init_scales = init_scales.squeeze(0)
            opt_log_scales = torch.log(torch.clamp(init_scales, min=1e-6)).requires_grad_(False)
            
            opt_rotations = target_gaussians.rotations.clone().detach()
            if opt_rotations.dim() == 3: opt_rotations = opt_rotations.squeeze(0)
            opt_rotations.requires_grad_(False)
            
            opt_opacities = inverse_sigmoid(torch.clamp(torch.sigmoid(target_gaussians.opacities), min=0.1, max=0.99)).detach()
            if opt_opacities.dim() == 2: opt_opacities = opt_opacities.squeeze(0)
            opt_opacities.requires_grad_(False)
            
            opt_harmonics = target_gaussians.harmonics.clone().detach()
            if opt_harmonics.dim() == 4: opt_harmonics = opt_harmonics.squeeze(0)
            opt_harmonics.requires_grad_(False)

            # [新增] 准备 Stereo Anchor 数据
            anchor_gt_depth = None
            anchor_valid_mask = None
            anchor_extrinsics = None
            anchor_intrinsics = None
            anchor_near = None
            anchor_far = None
            
            if stereo_anchor_idx is not None:
                print(f"  [TTO Stage 1] Enabling Stereo Reprojection Loss (Anchor: {stereo_anchor_idx})")
                anchor_gt_depth = context_render_pkg['depth'][:, stereo_anchor_idx].to(device) # [1, H, W]
                anchor_extrinsics = context_render_pkg['extrinsics'][:, stereo_anchor_idx:stereo_anchor_idx+1]
                anchor_intrinsics = context_render_pkg['intrinsics'][:, stereo_anchor_idx:stereo_anchor_idx+1]
                anchor_near = context_render_pkg['near'][:, stereo_anchor_idx:stereo_anchor_idx+1]
                anchor_far = context_render_pkg['far'][:, stereo_anchor_idx:stereo_anchor_idx+1]
                anchor_valid_mask = anchor_gt_depth > 1e-4

        # --------------------------------------------------------------------------
        # STAGE 1: Geometric Surface Alignment
        # --------------------------------------------------------------------------
        print("  [TTO] Stage 1 (Ray-Depth Optimization)...")
        
        with torch.no_grad():
            # 初始化距离
            aligned_z_depth = target_render_pkg['depth'].view(-1).to(device)
            view_dir = c2w_rot[:, 2] 
            cos_theta = (ray_dirs * view_dir).sum(dim=-1)
            if cos_theta.mean() < 0: cos_theta = -cos_theta
            cos_theta = torch.clamp(cos_theta, min=0.01)
            init_dists = (aligned_z_depth / cos_theta).unsqueeze(-1)
        
        opt_dists = init_dists.clone().detach().requires_grad_(True)
        gt_sparse_depth = target_render_pkg.get('sparse_gt_depth', None).to(device)
        gt_mask = target_render_pkg.get('sparse_valid_mask', None).to(device) > 0.5
        gt_image = target_render_pkg['image'].to(device)

        optimizer_s1 = torch.optim.Adam([opt_dists], lr=0.01) 

        with torch.enable_grad():
            for step in range(self.steps_stage1):
                optimizer_s1.zero_grad()
                
                curr_means = cam_center + opt_dists * ray_dirs
                curr_target_gaussians = self._build_gaussians(curr_means, opt_log_scales, opt_rotations, opt_opacities, opt_harmonics, c2w_rot)
                merged_gaussians = self.merge_gaussians_for_render(flat_context_gaussians, curr_target_gaussians)
                
                # Loss A: Target Depth
                render_out = decoder(merged_gaussians, target_render_pkg['extrinsics'], target_render_pkg['intrinsics'], target_render_pkg['near'], target_render_pkg['far'], (target_render_pkg['h'], target_render_pkg['w']), depth_mode="depth")
                
                loss_depth = (torch.abs(render_out.depth.squeeze(1) - gt_sparse_depth) * gt_mask).sum() / (gt_mask.sum() + 1e-6)
                loss_rgb = l1_loss(render_out.color[0], gt_image[0])
                
                total_loss = 1.0 * loss_depth + 0.1 * loss_rgb
                
                if step % 10 == 0:
                    print(f"    Step {step} | Target align: Depth Loss: {loss_depth.item():.4f}, RGB Loss: {loss_rgb.item():.4f}")
                
                # Loss B: Stereo Reprojection (Occlusion Aware)
                if stereo_anchor_idx is not None:
                    stereo_out = decoder(curr_target_gaussians, anchor_extrinsics, anchor_intrinsics, anchor_near, anchor_far, (target_render_pkg['h'], target_render_pkg['w']), depth_mode="depth")
                    stereo_depth = stereo_out.depth[0]
                    stereo_alpha = stereo_out.alpha[0]
                    
                    # # Filter: Only punish if point is BEHIND or ON surface (allow occlusion, disallow flying in front)
                    # epsilon = 0.05
                    # depth_consistency_mask = stereo_depth >= (anchor_gt_depth - epsilon)
                    # valid_stereo_mask = (stereo_alpha > 0.5) & anchor_valid_mask & depth_consistency_mask
                    valid_stereo_mask = (stereo_alpha > 0.5) & anchor_valid_mask

                    if valid_stereo_mask.sum() > 100:
                        loss_stereo = torch.abs(stereo_depth - anchor_gt_depth)
                        loss_stereo = (loss_stereo * valid_stereo_mask).sum() / (valid_stereo_mask.sum() + 1e-6)
                        total_loss += 1.0 * loss_stereo
                        
                        if step % 10 == 0:
                            print(f"    Step {step} | Stereo Loss: {loss_stereo.item():.4f}")

                total_loss.backward()
                optimizer_s1.step()
                with torch.no_grad(): 
                    opt_dists.data = torch.clamp(opt_dists.data, min=0.1)

        # --------------------------------------------------------------------------
        # Eval-After
        # --------------------------------------------------------------------------
        with torch.no_grad():
            final_means_s1 = cam_center + opt_dists.detach() * ray_dirs 
            target_gaussians.means = final_means_s1.unsqueeze(0).detach()
            
            run_eval("Post-Stage1 (Stereo Optimized)", target_gaussians)

        # --------------------------------------------------------------------------
        # STAGE 2: Hard Hole Filter
        # --------------------------------------------------------------------------
        print("  [TTO] Stage 2 Hole Filter...")
        with torch.no_grad():
            if hole_tags is not None: keep_mask = hole_tags.to(device)
            else: keep_mask = torch.ones(final_means_s1.shape[0], dtype=torch.bool, device=device)
            
            print(f"  [TTO Stage 2-Filter] Points before: {len(final_means_s1)} -> after: {keep_mask.sum()}")
            if keep_mask.sum() == 0: return target_gaussians

            valid_means = final_means_s1[keep_mask]
            valid_log_scales = opt_log_scales[keep_mask]
            valid_rotations = opt_rotations[keep_mask]
            valid_opacities = opt_opacities[keep_mask]
            valid_harmonics = opt_harmonics[keep_mask]

        # # --------------------------------------------------------------------------
        # # STAGE 2: Hard Hole Filter with dilation
        # # --------------------------------------------------------------------------
        # print("  [TTO] Stage 2 Hole Filter...")
        # assert hole_tags is not None, "Hole mask must be here."
        # H, W = target_render_pkg['h'], target_render_pkg['w']
        # hole_mask_2d = hole_tags.view(1, 1, H, W).float()
        # dilation_radius = 5 # 向外扩张5个像素
        # kernel_size = 2 * dilation_radius + 1
        # padding = dilation_radius
        # dilated_hole_mask = F.max_pool2d(hole_mask_2d, kernel_size=kernel_size, stride=1, padding=padding)
        # keep_mask = (dilated_hole_mask.view(-1) > 0.5)
        # print(f"  [TTO Stage 2-Filter Mask Dilation] Original Hole: {hole_tags.sum()} -> Dilated Hole (+{dilation_radius}px): {keep_mask.sum()} pixels.")
       
        # with torch.no_grad():
        #     print(f"  [TTO] Stage 2-Filter Points before: {len(final_means_s1)} -> after: {keep_mask.sum()}")
        #     if keep_mask.sum() == 0: return target_gaussians

        #     valid_means = final_means_s1[keep_mask]
        #     valid_log_scales = opt_log_scales[keep_mask]
        #     valid_rotations = opt_rotations[keep_mask]
        #     valid_opacities = opt_opacities[keep_mask]
        #     valid_harmonics = opt_harmonics[keep_mask]

        # --------------------------------------------------------------------------
        # STAGE 2.5: Context Monster targeted optimization
        # 核心逻辑：仅开启 opacity 梯度，通过 RGB 和 Coverage 损失优化怪物高斯的透明度
        # --------------------------------------------------------------------------
        # has_monsters = monster_indices is not None and monster_indices.numel() > 0
        # if has_monsters:
        #     print(f"  [TTO Stage 2.5] Optimizing {monster_indices.numel()} Monster Opacities...")
            
        #     # 1. 提取怪物参数：仅开启 opacities 的梯度
        #     m_means = flat_context_gaussians.means[:, monster_indices].squeeze(0)
        #     m_log_scales = torch.log(flat_context_gaussians.scales[:, monster_indices].squeeze(0).clamp(min=1e-10))
        #     m_rots = flat_context_gaussians.rotations[:, monster_indices].squeeze(0)
        #     m_harms = flat_context_gaussians.harmonics[:, monster_indices].squeeze(0)
            
        #     m_opas = flat_context_gaussians.opacities[:, monster_indices].squeeze(0).detach().requires_grad_(True)

        #     # 2. 构建静态背景 (剔除怪物)
        #     all_indices = torch.arange(flat_context_gaussians.means.shape[1], device=device)
        #     monster_mask = torch.ones(len(all_indices), dtype=torch.bool, device=device)
        #     monster_mask[monster_indices] = False
            
        #     bg_data = {
        #         field.name: getattr(flat_context_gaussians, field.name)[:, monster_mask].detach()
        #         for field in fields(Gaussians) if getattr(flat_context_gaussians, field.name) is not None
        #     }
        #     bg_context_gaussians = Gaussians(**bg_data)

        #     # 3. 优化器配置：仅优化不透明度
        #     optimizer_monster = torch.optim.Adam([{'params': [m_opas], 'lr': self.lr_opa_s2}], eps=1e-15)

        #     # 4. 优化循环
        #     with torch.enable_grad():
        #         for step in range(self.steps_stage2):
        #             optimizer_monster.zero_grad()
        #             curr_m = self._build_gaussians(m_means, m_log_scales, m_rots, m_opas, m_harms, c2w_rot)
        #             temp_tgt = self._build_gaussians(valid_means, valid_log_scales, valid_rotations, valid_opacities, valid_harmonics, c2w_rot)

        #             render_pkg = self.merge_gaussians_for_render(bg_context_gaussians, curr_m)
        #             render_pkg = self.merge_gaussians_for_render(render_pkg, temp_tgt)
                    
        #             out = decoder(
        #                 render_pkg, target_render_pkg['extrinsics'], target_render_pkg['intrinsics'], 
        #                 target_render_pkg['near'], target_render_pkg['far'], 
        #                 (target_render_pkg['h'], target_render_pkg['w']), depth_mode=None
        #             )

        #             loss_rgb = l1_loss(out.color[0], gt_image[0])
        #             loss_rgb.backward()
        #             optimizer_monster.step()

        #             if step % 10 == 0:
        #                 print(f"  [TTO Stage 2] Step {step}: RGB Loss: {loss_rgb.item():.4f}")

        #     # 5. 写回全局 Context 并更新协方差
        #     with torch.no_grad():
        #         flat_context_gaussians.opacities[:, monster_indices] = m_opas.unsqueeze(0)
        #         updated_m = self._build_gaussians(m_means, m_log_scales, m_rots, m_opas, m_harms, c2w_rot)
        #         flat_context_gaussians.covariances[:, monster_indices] = updated_m.covariances

        # self.steps_stage2_baking = 50
        # print(f"  [TTO] Stage 2: Opacity Baking ({self.steps_stage2_baking} steps)...")
        # opt_means = valid_means.clone().detach().requires_grad_(False)
        # opt_log_scales = valid_log_scales.clone().detach().requires_grad_(False)
        # opt_rotations = valid_rotations.clone().detach().requires_grad_(False)
        # opt_opacities = valid_opacities.clone().detach().requires_grad_(True)
        # opt_harmonics = valid_harmonics.clone().detach().requires_grad_(False) 
        
        # # High LR for Baking
        # self.lr_opa_baking = 0.2
        # optimizer_baking = torch.optim.Adam([opt_opacities], lr=self.lr_opa_baking)

        # # Opacity-baking
        # with torch.enable_grad():
        #     for step in range(self.steps_stage2_baking):
        #         optimizer_baking.zero_grad()
        #         curr_tgt_active = self._build_gaussians(opt_means, opt_log_scales, opt_rotations, opt_opacities, opt_harmonics, c2w_rot)
        #         merged_gaussians = self.merge_gaussians_for_render(flat_context_gaussians, curr_tgt_active)
        #         out = decoder(merged_gaussians, target_render_pkg['extrinsics'], target_render_pkg['intrinsics'], target_render_pkg['near'], target_render_pkg['far'], (target_render_pkg['h'], target_render_pkg['w']), depth_mode=None)
        #         loss_rgb = l1_loss(out.color[0], gt_image[0])
        #         loss_rgb.backward()
        #         optimizer_baking.step()
        #         if step % 10 == 0:
        #             print(f"    Step {step} | Baking L1: {loss_rgb.item():.4f}")

        # # STAGE 2-Pruning: (Post Stage 2)
        # with torch.no_grad():
        #     mask_keep = torch.sigmoid(opt_opacities) > 0.2
        #     print(f"  [Filter Stage-2] Points before: {len(opt_means)} -> after: {mask_keep.sum()}")
        #     if mask_keep.sum() == 0: return target_gaussians
        #     valid_means = opt_means[mask_keep]
        #     valid_log_scales = opt_log_scales[mask_keep]
        #     valid_rotations = opt_rotations[mask_keep]
        #     valid_opacities = opt_opacities[mask_keep]
        #     valid_harmonics = opt_harmonics[mask_keep]

        # --------------------------------------------------------------------------
        # STAGE 3: Refinement (随机采样 K 个 Context 视角)
        # --------------------------------------------------------------------------
        print("  [TTO] Stage 3 (Unified RGB Refinement - Random K Views)...")
        
        opt_means = valid_means.clone().detach().requires_grad_(True)
        opt_log_scales = valid_log_scales.clone().detach().requires_grad_(True)
        opt_rotations = valid_rotations.clone().detach().requires_grad_(True)
        opt_opacities = valid_opacities.clone().detach().requires_grad_(True)
        opt_harmonics = valid_harmonics.clone().detach().requires_grad_(False) 

        optimizer_s2 = torch.optim.Adam([
            {'params': [opt_means], 'lr': self.lr_pos_s3}, 
            {'params': [opt_log_scales], 'lr': self.lr_scale_s3},
            {'params': [opt_rotations], 'lr': self.lr_rot_s3},
            {'params': [opt_opacities], 'lr': self.lr_opa_s3},
        ], eps=1e-15)

        num_ctx_views = context_render_pkg['image'].shape[1]
        
        with torch.enable_grad():
            for step in range(self.steps_stage3):
                optimizer_s2.zero_grad()
                with torch.no_grad():
                    opt_log_scales.data = torch.clamp(opt_log_scales.data, max=torch.log(torch.tensor(self.max_scale, device=device)))

                curr_tgt_gaussians = self._build_gaussians(opt_means, opt_log_scales, opt_rotations, opt_opacities, opt_harmonics, c2w_rot)
                merged_gaussians = self.merge_gaussians_for_render(flat_context_gaussians, curr_tgt_gaussians)

                total_loss = 0.0
                views_count = 0
                
                # --- A. Target View RGB ---
                tgt_out = decoder(merged_gaussians, target_render_pkg['extrinsics'], target_render_pkg['intrinsics'], target_render_pkg['near'], target_render_pkg['far'], (target_render_pkg['h'], target_render_pkg['w']), depth_mode=None)
                total_loss += l1_loss(tgt_out.color[0], gt_image[0])
                views_count += 1

                # --- B. Context Views RGB (Random K) ---
                if num_ctx_views > 0:
                    K = self.context_view_sample_k
                    if num_ctx_views <= K: selected_indices = torch.arange(num_ctx_views, device=device)
                    else: selected_indices = torch.randperm(num_ctx_views, device=device)[:K]
                    
                    for v_idx in selected_indices:
                        ctx_extrinsics = context_render_pkg['extrinsics'][:, v_idx : v_idx+1]
                        ctx_intrinsics = context_render_pkg['intrinsics'][:, v_idx : v_idx+1]
                        ctx_near = context_render_pkg['near'][:, v_idx : v_idx+1]
                        ctx_far = context_render_pkg['far'][:, v_idx : v_idx+1]
                        ctx_image_gt = context_render_pkg['image'][:, v_idx]
                        
                        ctx_out = decoder(merged_gaussians, ctx_extrinsics, ctx_intrinsics, ctx_near, ctx_far, (context_render_pkg['h'], context_render_pkg['w']), depth_mode=None)
                        total_loss += l1_loss(ctx_out.color[0], ctx_image_gt[0])
                        views_count += 1
                
                final_loss = total_loss / views_count
                final_loss.backward()
                optimizer_s2.step()
                
                if step % 20 == 0:
                    print(f"  [TTO Stage-3] Step {step}: AvgRGB={final_loss.item():.4f} (Views: {views_count})")

        # Pruning & Return
        with torch.no_grad():
            final_scales = torch.exp(opt_log_scales)
            mask_keep = torch.sigmoid(opt_opacities) > 0.05 
            print(f"  [Pruning] Final Keep: {mask_keep.sum()} / {len(mask_keep)}")

            valid_means = opt_means[mask_keep]
            valid_scales = final_scales[mask_keep]
            valid_rotations = opt_rotations[mask_keep]
            valid_rotations = valid_rotations / (valid_rotations.norm(dim=-1, keepdim=True) + 1e-9)
            valid_opacities = opt_opacities[mask_keep]
            valid_harmonics = opt_harmonics[mask_keep]

            final_local_covs = build_covariance(valid_scales, valid_rotations)
            R_c2w_expanded = c2w_rot.unsqueeze(0).expand(final_local_covs.shape[0], -1, -1)
            final_world_covs = R_c2w_expanded @ final_local_covs @ R_c2w_expanded.transpose(-2, -1)

            target_gaussians.means = valid_means.unsqueeze(0).detach()
            target_gaussians.scales = valid_scales.unsqueeze(0).detach()
            target_gaussians.rotations = valid_rotations.unsqueeze(0).detach()
            target_gaussians.covariances = final_world_covs.unsqueeze(0).detach()
            target_gaussians.opacities = valid_opacities.unsqueeze(0).detach()
            target_gaussians.harmonics = valid_harmonics.unsqueeze(0).detach()

        return target_gaussians

    def _build_gaussians(self, means, log_scales, rotations, opacities, harmonics, c2w_rot):
        if means.dim() == 3: means_sq = means.squeeze(0)
        else: means_sq = means
        if log_scales.dim() == 3: log_scales_sq = log_scales.squeeze(0)
        else: log_scales_sq = log_scales
        if rotations.dim() == 3: rotations_sq = rotations.squeeze(0)
        else: rotations_sq = rotations
        if opacities.dim() == 2: opacities_sq = opacities.squeeze(0)
        else: opacities_sq = opacities
        if harmonics.dim() == 4: harmonics_sq = harmonics.squeeze(0)
        else: harmonics_sq = harmonics
        scales = torch.exp(log_scales_sq)
        rot = rotations_sq / (rotations_sq.norm(dim=-1, keepdim=True) + 1e-9)
        local_covs = build_covariance(scales, rot)
        R_c2w_expanded = c2w_rot.unsqueeze(0).expand(local_covs.shape[0], -1, -1)
        world_covs = R_c2w_expanded @ local_covs @ R_c2w_expanded.transpose(-2, -1)
        return Gaussians(means=means_sq.unsqueeze(0), covariances=world_covs.unsqueeze(0), scales=scales.unsqueeze(0), rotations=rot.unsqueeze(0), harmonics=harmonics_sq.unsqueeze(0), opacities=opacities_sq.unsqueeze(0))

    def merge_gaussians_for_render(self, g1: Gaussians, g2: Gaussians) -> Gaussians:
        merged = {}
        for field in fields(Gaussians):
            v1 = getattr(g1, field.name)
            v2 = getattr(g2, field.name)
            if v1 is not None and v2 is not None:
                if v1.dim() == v2.dim(): merged[field.name] = torch.cat([v1, v2], dim=1)
                else: merged[field.name] = v1 
        return Gaussians(**merged)
    
    def flatten_gaussians(self, g: Gaussians) -> Gaussians:
        flat = {}
        for field in fields(Gaussians):
            val = getattr(g, field.name)
            if val is not None: flat[field.name] = val.reshape(1, -1, *val.shape[2:])
        return Gaussians(**flat)

# -----------------------------------------------------------------------------
# 3. Encoder Depth Splat Refiner (Main Class)
# -----------------------------------------------------------------------------
class EncoderDepthSplatRefiner(EncoderDepthSplat):
    def __init__(self, cfg: EncoderDepthSplatCfg) -> None:
        super().__init__(cfg)

        self.rotation_weight = 1.0
        self.translation_weight = 1.0
        
        # 1. Using Positional-Encoding Field
        print("Loading PE Pipeline (from ~/code/pe_field)...")
        external_pe_root = os.path.expanduser("~/gaoao/code")
        if external_pe_root not in sys.path:
            sys.path.insert(0, external_pe_root)
            
        import pe_field
        from pe_field.pipeline_pe_field import PEFieldPipeline
        # 初始化 PE-Field Pipeline
        self.pe_field = PEFieldPipeline()

        # 2. Init TTO Optimizer
        self.tto_optimizer = GaussianTTOOptimizer()

    
    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        extra_context: Optional[dict] = None,
        decoder = None, 
    ):
        _, _, _, h, w = context["image"].shape

        # ... (Predict Context and Merge) ...
        with torch.no_grad():
            res_context = self.predict(context, global_step, deterministic, visualization_dump, scene_names)
        
        if self.cfg.train_depth_only:
            return res_context
            
        original_context_gaussians = res_context["gaussians"]
        context_depths = res_context["depths"]
        
        flat_context_gaussians = self._flatten_and_merge_gaussians([original_context_gaussians], batch_size=1)
        final_output_list = []

        # [DEBUG] save context image
        output_dir = get_cfg().get("output_dir", "experiments/debug")
        scene_name = scene_names[0][:-len('/nerfstudio')] if scene_names else "scene"
        vis_dir = os.path.join(output_dir, "before", scene_name)
        self._save_images(decoder, flat_context_gaussians, context, h, w, vis_dir)
        
        # extra_context = None
        if extra_context is not None:

            assignments = self.find_nearest_assignments(context["extrinsics"][0], extra_context["extrinsics"][0])
            stereo_assignments = self.find_stereo_assignments(context["extrinsics"][0], extra_context["extrinsics"][0])

            num_targets = extra_context["image"].shape[1]
            
            output_dir = get_cfg().get("output_dir", "experiments/debug")
            scene_name = scene_names[0][:-len('/nerfstudio')] if scene_names else "scene"
            debug_dir = os.path.join(output_dir, "debug", scene_name)
            os.makedirs(debug_dir, exist_ok=True)

            context_render_pkg = {
                "extrinsics": context["extrinsics"],
                "intrinsics": context["intrinsics"],
                "near": context["near"],
                "far": context["far"],
                "image": context["image"],
                "depth": context_depths,
                "h": h, "w": w,
                "index": context["index"]
            }

            for i in range(num_targets):
                difix_anchor_idx = int(assignments[i])
                print(f"--- Processing Target View {i} ---")
                print(f"Fixing Image Using Anchor: {difix_anchor_idx} ")

                # Step A: Check Coverage (Render current Gaussians)
                with torch.no_grad():
                    tgt_out = decoder(flat_context_gaussians, extra_context["extrinsics"][:, i:i+1], extra_context["intrinsics"][:, i:i+1], extra_context["near"][:, i:i+1], extra_context["far"][:, i:i+1], (h, w), depth_mode="depth")
                    
                    # ================= [Online Get mask: Based on Radii] =================
                    current_radii = tgt_out.radii.view(-1) # [N]
                    prune_threshold = 20 
                    mask_to_prune = current_radii > prune_threshold
                    # scales = flat_context_gaussians.scales
                    # big_points_ws = scales.max(dim=1).values > 3.0
                    # mask_to_prune = torch.logical_or(big_points_vs, big_points_ws)
                    if mask_to_prune.sum() > 0:
                        print(f"  [Radii Pruning] Detected {mask_to_prune.sum()} artifacts (Radii > {prune_threshold}). Removing...")
                        mask_keep = ~mask_to_prune

                        # 更新 context
                        new_data = {}
                        for field in fields(Gaussians):
                            val = getattr(flat_context_gaussians, field.name)
                            if val is not None:
                                new_data[field.name] = val[:, mask_keep]
                        temp_flat_context_gaussians = Gaussians(**new_data)
                        
                        # Re-Render
                        tgt_out = decoder(
                            temp_flat_context_gaussians, 
                            extra_context["extrinsics"][:, i:i+1], 
                            extra_context["intrinsics"][:, i:i+1], 
                            extra_context["near"][:, i:i+1], 
                            extra_context["far"][:, i:i+1], 
                            (h, w), 
                            depth_mode="depth"
                        )
                    else:
                        print("  [Radii Pruning] No large artifacts detected. Skipping re-render.")
                    # ================= [Online Get mask: Based on Radii] =================

                    final_output_list.append(self._clone_gaussians_data(flat_context_gaussians))
                    tgt_rendered_rgb = tgt_out.color[0]
                    tgt_opacity_mask = tgt_out.alpha[0]
                    tgt_hole_mask = tgt_opacity_mask < 0.5

                    # 2. [TRICK] 向外膨胀1个像素
                    dilation_radius = 1
                    hole_mask_float = tgt_hole_mask.float().view(1, 1, h, w)
                    kernel_size = 2 * dilation_radius + 1
                    padding = dilation_radius
                    dilated_hole_mask = F.max_pool2d(
                        hole_mask_float,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding
                    )
                    tgt_hole_mask = (dilated_hole_mask > 0).view(1, h, w)
                    tgt_valid_mask = ~tgt_hole_mask

                    # coverage_ratio = (tgt_opacity_mask > 0.99).float().mean().item()
                    # should_pass = coverage_ratio < 0.1
                    # if should_pass:
                    #     print(f"Coverage {coverage_ratio:.3f} < 0.1, skipping view {i}")
                    #     continue

                    tgt_sparse_depth = tgt_out.depth[0] * tgt_valid_mask.float()
                    # tgt_hole_mask = ~tgt_valid_mask
                
                # # ================= [Monster Index Tracking] =================
                # current_radii = tgt_out.radii.view(-1).float()
                # # hard_limit = 200.0
                # hard_limit = torch.quantile(current_radii, 0.99)
                # is_monster = current_radii > hard_limit
                # monster_indices = torch.where(is_monster)[0]
                # num_monsters = monster_indices.numel()
                # if num_monsters > 0:
                #     print(f"  [Monster Tracking] Found {num_monsters} monsters in Context. Passing indices to TTO Stage 2.")
                # # ================= [Monster Index Tracking] =================
                # final_output_list.append(self._clone_gaussians_data(flat_context_gaussians))
                # tgt_rendered_rgb = tgt_out.color[0]
                # tgt_opacity_mask = tgt_out.alpha[0]
                # tgt_valid_mask = tgt_opacity_mask > 0.5

                # # coverage_ratio = (tgt_opacity_mask > 0.99).float().mean().item()
                # # should_pass = coverage_ratio < 0.1
                # # if should_pass:
                # #     print(f"Coverage {coverage_ratio:.3f} < 0.1, skipping view {i}")
                # #     continue

                # tgt_sparse_depth = tgt_out.depth[0] * tgt_valid_mask.float()
                # tgt_hole_mask = ~tgt_valid_mask

                # --- [Modified] Step B: Generate with PE-Field (Replacing DiFix) ---
                anchor_image = context["image"][:, difix_anchor_idx] 
                anchor_extrinsic = context["extrinsics"][:, difix_anchor_idx]
                anchor_depth = context_depths[:, difix_anchor_idx]
                target_extrinsic = extra_context["extrinsics"][:, i]

                pe_ctx = {
                    "image": anchor_image,
                    "extrinsics": anchor_extrinsic
                }
                pe_tgt = {
                    "extrinsics": target_extrinsic
                }
                print(f"Generating view {i} using PE-Field (with GT/Pred depth)...")
                fixed_image_tensor, phi, theta, warped_image_tensor = self.pe_field(
                    context=pe_ctx, 
                    target=pe_tgt, 
                    num_inference_steps=10,
                )

                print(f"Calculate the spherical offset: {phi}, {theta}")

                # [DEBUG] Visualization
                with torch.no_grad():
                    save_image(tgt_rendered_rgb, os.path.join(debug_dir, f"view_{i}_00_rgb_before.png"))
                    save_image(fixed_image_tensor[0], os.path.join(debug_dir, f"view_{i}_00_rgb_after_pe_field.png"))
                    save_image(warped_image_tensor[0], os.path.join(debug_dir, f"view_{i}_00_rgb_warped_pe_field.png"))
                    anchor_depth_vis = visualize_depth(anchor_depth[0], cmap_name='Spectral')
                    save_image(anchor_depth_vis, os.path.join(debug_dir, f"view_{i}_01_anchor_depth.png"))
                    sparse_depth_vis = visualize_depth(tgt_sparse_depth[0], cmap_name='Spectral')
                    save_image(sparse_depth_vis, os.path.join(debug_dir, f"view_{i}_01_sparse_depth.png"))

                # Step B: Predict Target Gaussians on Paired Views
                
                # stereo_anchor_idx = int(assignments[i])
                stereo_anchor_idx = int(stereo_assignments[i])
                print(f"Calculate Gaussian Using Anchor: {stereo_anchor_idx} ")
                with torch.no_grad():
                    pair = {}
                    for key in extra_context.keys():
                        if key not in context: continue
                        val_anchor = context[key][:, stereo_anchor_idx : stereo_anchor_idx + 1]
                        if key == "image": val_target = fixed_image_tensor.unsqueeze(0)
                        else: val_target = extra_context[key][:, i : i + 1]
                        pair[key] = torch.cat([val_anchor, val_target], dim=1)
                    res_pair = self.predict(pair, global_step, deterministic)

                # ---------------- align using anchor ---------------------------------
                # 1. 获取 Target 的原始预测 (用于生成高斯)
                target_gaussians = self._slice_gaussians_views(res_pair["gaussians"], [1])
                target_gaussians = self._flatten_and_merge_gaussians([target_gaussians], batch_size=1)
                
                # 2. 获取深度图
                # res_pair["depths"] shape: [1, 2, H, W]. Index 0 is Anchor, Index 1 is Target.
                anchor_pred_depth = res_pair["depths"][:, 0] # [1, H, W]
                target_pred_depth = res_pair["depths"][:, 1] # [1, H, W]

                # 3. 获取 Anchor 的 GT 深度 (Ground Truth)
                # context["depth"] shape: [1, V, H, W]
                anchor_gt_depth = context_depths[:, stereo_anchor_idx] # [1, H, W]
                
                # 4. 计算对齐参数 (Compute Scale & Shift on Anchor)
                # Mask: 仅在 GT 有效的区域计算 (GT > 0)
                anchor_valid_mask = anchor_gt_depth > 1e-4
                
                # 这里我们不需要返回对齐后的 anchor depth，只需要 scale 和 shift
                # 注意：align_depth_median 函数需要支持 batch 输入或 squeeze 后输入
                # 假设 align_depth_median 接收 (H, W) 或 (1, H, W)
                _, scale, shift = align_depth_ransac_with_anchor(
                    anchor_pred_depth.squeeze(0), 
                    anchor_gt_depth.squeeze(0), 
                    anchor_valid_mask.squeeze(0)
                )
                
                print(f"  [Depth Align via Anchor] Scale: {scale:.4f}, Shift: {shift:.4f}")
                
                # 5. 将参数应用到 Target (Apply to Target)
                # 公式: Depth_aligned = scale * Depth_mono + shift
                aligned_dense_depth = target_pred_depth * scale + shift
                
                # 安全截断，防止数值爆炸
                # aligned_target_depth = torch.clamp(aligned_target_depth, min=0.01, max=100.0)

                # [DEBUG] Visualization
                dense_depth_vis = visualize_depth(aligned_dense_depth, cmap_name='Spectral')
                save_image(dense_depth_vis, os.path.join(debug_dir, f"view_{i}_02_tgt_dense_depth.png"))
                mask_vis = tgt_valid_mask.float().cpu().squeeze()  # [H, W]
                if mask_vis.dim() == 2:
                    mask_vis = mask_vis.unsqueeze(0).repeat(3, 1, 1) # [3, H, W]
                save_image(mask_vis, os.path.join(debug_dir, f"view_{i}_03_tgt_align_mask.png"))
                # # ---------------------------------------------------------------------

                # # ---------------- align using target ---------------------------------
                
                # # 1. 获取 Target 的原始预测 (用于生成高斯)
                # target_gaussians = self._slice_gaussians_views(res_pair["gaussians"], [1])
                # target_gaussians = self._flatten_and_merge_gaussians([target_gaussians], batch_size=1)
                
                # # 2. 获取 Target 的单目预测深度 (Monocular Predicted)
                # # res_pair["depths"] shape: [1, 2, H, W]. Index 1 is Target.
                # target_pred_depth = res_pair["depths"][:, 1] # [1, H, W]

                # # 3. 获取 Target 的稀疏真值 (Sparse GT from Context Projection)
                # # 这些变量在 Step A 中已经计算过了：
                # # tgt_sparse_depth: [1, H, W] (Context 渲染出的深度)
                # # tgt_valid_mask:   [1, H, W] (Context 有效覆盖的区域 alpha > 0.8)
                
                # # 4. 执行对齐 (RANSAC)
                # # 注意：我们squeeze掉batch维度进行计算，最后再unsqueeze回来或者保持形状
                # print(f"  [Depth Align] Using Target Sparse GT (Valid Ratio: {tgt_valid_mask.float().mean():.2%})")
                
                # aligned_dense_depth, scale, shift = align_depth_ransac_with_target(
                #     target_pred_depth.squeeze(0),   # Input: Mono
                #     tgt_sparse_depth.squeeze(0),    # Target: GT
                #     tgt_valid_mask.squeeze(0)       # Mask: Only valid pixels
                # )
                
                # # 确保维度恢复为 [1, H, W]
                # if aligned_dense_depth.dim() == 2:
                #     aligned_dense_depth = aligned_dense_depth.unsqueeze(0)

                # print(f"  [Depth Align Result] Scale: {scale:.4f}, Shift: {shift:.4f}")

                # # [DEBUG] Visualization
                # # 保存对齐后的稠密深度图
                # vis_dir = os.path.join(output_dir, "debug", scene_name) # 确保路径存在
                # dense_depth_vis = visualize_depth(aligned_dense_depth, cmap_name='Spectral')
                # save_image(dense_depth_vis, os.path.join(vis_dir, f"view_{i}_02_dense_depth_aligned.png"))
                
                # mask_vis = tgt_valid_mask.float().cpu().squeeze()  # [H, W]
                # if mask_vis.dim() == 2:
                #     mask_vis = mask_vis.unsqueeze(0).repeat(3, 1, 1) # [3, H, W]

                # save_image(mask_vis, os.path.join(vis_dir, f"view_{i}_03_align_mask.png"))
                # # # ---------------------------------------------------------------------

                if target_gaussians is not None:
                    # target_gaussians.means shape: [1, N, 3]
                    num_gaussians = target_gaussians.means.shape[1]
                    
                    # tgt_hole_mask shape: [1, 1, H, W]
                    assert num_gaussians == (h * w), f"Mask shape mismatch. G:{num_gaussians}, H*W:{h*w}"
                    hole_tags = tgt_hole_mask.view(-1).bool() # [N]

                    # 获取当前 Target 视角的参数
                    tgt_extrinsics = extra_context["extrinsics"][:, i:i+1]
                    tgt_intrinsics = extra_context["intrinsics"][:, i:i+1]
                    tgt_near = extra_context["near"][:, i:i+1]
                    tgt_far = extra_context["far"][:, i:i+1]
                    tgt_index = extra_context["index"][:, i:i+1]

                    target_render_pkg = {
                        "extrinsics": tgt_extrinsics, 
                        "intrinsics": tgt_intrinsics,
                        "near": tgt_near, 
                        "far": tgt_far,
                        "image": fixed_image_tensor,
                        "depth": aligned_dense_depth.unsqueeze(1),
                        "hole_mask": tgt_hole_mask[0, 0],
                        "sparse_gt_depth": tgt_sparse_depth,
                        "sparse_valid_mask": tgt_valid_mask,
                        "h": h, "w": w
                    }
                    refined_target_gaussians = self.tto_optimizer(
                        decoder=decoder, 
                        target_gaussians=target_gaussians, 
                        context_gaussians=flat_context_gaussians, 
                        context_render_pkg=context_render_pkg, 
                        target_render_pkg=target_render_pkg,
                        hole_tags=hole_tags,
                        stereo_anchor_idx=stereo_anchor_idx,
                        monster_indices=None,
                        # monster_indices=monster_indices
                    )
                    # 3. 收集结果
                    final_output_list.append(self._clone_gaussians_data(refined_target_gaussians))
                    
                    # 4. [关键更新 A] 将优化后的 Target Gaussians 合并进 Context Gaussians
                    # 这样下一个 Target 就能看到当前 Target 产生的几何信息
                    flat_context_gaussians = self._flatten_and_merge_gaussians(
                        [flat_context_gaussians, refined_target_gaussians], 
                        batch_size=1
                    )
                    
                    # 5. [关键更新 B] 将当前 Target 的渲染信息追加到 Context Render Package 中
                    # 这样下一个 Target 在 Stage 2 训练时，可以将当前 Target 视角的 RGB 作为 Context 监督
                    
                    # 将 [B, N_old, ...] 扩展成 [B, N_old + 1, ...]
                    context_render_pkg['extrinsics'] = torch.cat([context_render_pkg['extrinsics'], tgt_extrinsics], dim=1)
                    context_render_pkg['intrinsics'] = torch.cat([context_render_pkg['intrinsics'], tgt_intrinsics], dim=1)
                    context_render_pkg['near'] = torch.cat([context_render_pkg['near'], tgt_near], dim=1)
                    context_render_pkg['far'] = torch.cat([context_render_pkg['far'], tgt_far], dim=1)
                    context_render_pkg['index'] = torch.cat([context_render_pkg['index'], tgt_index], dim=1)
                    
                    # fixed_image_tensor 应该是 [1, 1, H, W, 3] 或 [1, 1, H, W, C]
                    # context_render_pkg['image'] 是 [B, N_old, H, W, C]，需要将 fixed_image_tensor 扩展到 [1, 1, H, W, C]
                    # 假设 fixed_image_tensor 已经是 [1, 1, H, W, 3]，确保维度匹配
                    if fixed_image_tensor.dim() == 5 and fixed_image_tensor.shape[1] == 1:
                        context_render_pkg['image'] = torch.cat([context_render_pkg['image'], fixed_image_tensor], dim=1)
                    elif fixed_image_tensor.dim() == 4:
                        context_render_pkg['image'] = torch.cat([context_render_pkg['image'], fixed_image_tensor.unsqueeze(1)], dim=1)
                    else:
                        raise ValueError("Unexpected fixed_image_tensor dimension for context update.")

                    num_current_context = context_render_pkg['image'].shape[1]
                    num_context_gaussians = flat_context_gaussians.means.shape[1]
                    
                    print(f"  [Context Update] View {i} promoted to Context. ")
                    print(f"  -> Total Context Views: {num_current_context}")
                    print(f"  -> Total Context Gaussians: {num_context_gaussians}")
            # [DEBUG] save context image
            vis_dir = os.path.join(output_dir, "after", scene_name)
            self._save_images(decoder, flat_context_gaussians, context_render_pkg, h, w, vis_dir)
        else:
            final_output_list.append(self._clone_gaussians_data(original_context_gaussians))
        final_gaussians = self._flatten_and_merge_gaussians(final_output_list, batch_size=1)
        
        if self.cfg.return_depth: 
            return {
                    "gaussians": final_gaussians, 
                    "depths": context_depths
                }
        return final_gaussians
    
    def _save_images(self, decoder, gaussians, context, h, w, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        output = decoder(
            gaussians,
            context["extrinsics"],
            context["intrinsics"],
            context["near"],
            context["far"],
            (h, w),
            depth_mode=None,
        )
        images_prob_input = output.color[0]
        path = f"{output_dir}/color"
        for index, color in zip(context["index"][0], images_prob_input):
            save_image(color, f"{path}/{index:0>6}.png")
    

    def _flatten_and_merge_gaussians(self, gaussians_list: List[Gaussians], batch_size: int) -> Gaussians:
        merged_data = {}
        for field in fields(Gaussians):
            tensors_to_cat = []
            for g in gaussians_list:
                val = getattr(g, field.name)
                if val is not None and val.shape[1] > 0:
                    val = rearrange(val, "(b v) n ... -> b (v n) ...", b=batch_size)
                    tensors_to_cat.append(val)
            if tensors_to_cat:
                merged_data[field.name] = torch.cat(tensors_to_cat, dim=1)
            else:
                device = gaussians_list[0].means.device if gaussians_list else torch.device('cuda')
                merged_data[field.name] = torch.empty(0).to(device)
        return Gaussians(**merged_data)

    def _clone_gaussians_data(self, g: Gaussians) -> Gaussians:
        data = {}
        for field in fields(Gaussians):
            val = getattr(g, field.name)
            data[field.name] = val.clone() if val is not None else None
        return Gaussians(**data)

    def _slice_gaussians_views(self, gaussians: Gaussians, view_indices: List[int]) -> Gaussians:
        new_data = {}
        idx_tensor = torch.tensor(view_indices, device=gaussians.means.device, dtype=torch.long)
        for field in fields(Gaussians):
            val = getattr(gaussians, field.name)
            new_data[field.name] = val.index_select(0, idx_tensor)
        return Gaussians(**new_data)
    
    def compute_pose_distance(self, pose1, pose2):
        t1, t2 = pose1[:3, 3], pose2[:3, 3]
        translation_dist = np.linalg.norm(t1 - t2)
        R1 = Rotation.from_matrix(pose1[:3, :3])
        R2 = Rotation.from_matrix(pose2[:3, :3])
        q1, q2 = R1.as_quat(), R2.as_quat()
        if np.dot(q1, q2) < 0: q2 = -q2
        dot_product = np.clip(2 * np.dot(q1, q2)**2 - 1, -1.0, 1.0)
        rotation_dist = np.arccos(dot_product)
        return (self.translation_weight * translation_dist + self.rotation_weight * rotation_dist)

    def find_nearest_assignments(self, training_poses, testing_poses):
        if isinstance(training_poses, torch.Tensor): training_poses = training_poses.detach().cpu().numpy()
        if isinstance(testing_poses, torch.Tensor): testing_poses = testing_poses.detach().cpu().numpy()
        assignments = []
        for j in range(len(testing_poses)):
            distances = [self.compute_pose_distance(tp, testing_poses[j]) for tp in training_poses]
            assignments.append(np.argmin(distances))
        return assignments
    
    def find_stereo_assignments(self, training_poses, testing_poses, max_angle_deg=45.0):
        """
        寻找最佳的立体匹配 Anchor。
        策略：
        1. 过滤掉旋转角度差异过大（> max_angle_deg）的视角，因为它们重叠率可能太低。
        2. 在剩余的候选者中，选择物理距离（Translation）最大的那个。
        
        这能有效避免选择距离太近的帧（窄基线）导致 Unimatch 输出尺度漂移。
        """
        if isinstance(training_poses, torch.Tensor): training_poses = training_poses.detach().cpu().numpy()
        if isinstance(testing_poses, torch.Tensor): testing_poses = testing_poses.detach().cpu().numpy()
        
        assignments = []
        
        for j in range(len(testing_poses)):
            test_pose = testing_poses[j]
            test_R = test_pose[:3, :3]
            test_t = test_pose[:3, 3]
            
            candidates = []
            
            for i, train_pose in enumerate(training_poses):
                train_R = train_pose[:3, :3]
                train_t = train_pose[:3, 3]
                
                # 1. 计算物理距离 (Baseline)
                dist = np.linalg.norm(test_t - train_t)
                
                # 2. 计算视角夹角 (Rotation Angle)
                # Trace method for rotation distance
                R_diff = train_R @ test_R.T
                trace = np.trace(R_diff)
                # clip to safety range [-1, 3] for trace of rotation matrix
                trace = np.clip(trace, -1.0, 3.0) 
                # theta = arccos((tr(R) - 1) / 2)
                angle_rad = np.arccos((trace - 1) / 2)
                angle_deg = np.degrees(angle_rad)
                
                candidates.append({
                    "idx": i,
                    "dist": dist,
                    "angle": angle_deg
                })
            
            # --- 筛选逻辑 ---
            
            # Step A: 过滤掉角度差异太大的 (保证重叠率)
            valid_candidates = [c for c in candidates if c["angle"] < max_angle_deg]
            
            if len(valid_candidates) > 0:
                # Step B: 在有效候选者中，选基线最宽的 (保证几何精度，锁死尺度)
                # 按距离降序排列
                valid_candidates.sort(key=lambda x: x["dist"], reverse=True)
                best_match = valid_candidates[0]
                assignments.append(best_match["idx"])
                
                # [DEBUG LOG] 方便你观察选了谁
                # print(f"  Target {j} -> Anchor {best_match['idx']} (Dist: {best_match['dist']:.3f}, Angle: {best_match['angle']:.1f}°)")
            else:
                # Step C: 如果所有视角角度都很大（极其罕见），回退到最近邻策略
                # 这种情况下通常选角度最小的
                candidates.sort(key=lambda x: x["angle"])
                assignments.append(candidates[0]["idx"])
                print(f"  [Warning] Target {j} has no overlapping context (Min Angle: {candidates[0]['angle']:.1f}°). Fallback to min angle.")

        return assignments