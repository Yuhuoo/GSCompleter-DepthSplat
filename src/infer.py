import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import warnings
import shutil
import math
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from colorama import Fore
from jaxtyping import install_import_hook
from PIL import Image
from einops import rearrange, repeat

from torchmetrics.image.fid import FrechetInceptionDistance

# 强制使用严格类型检查和 hook
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule, get_data_shim
    from src.global_cfg import set_cfg, get_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder.encoder_refinedsplat import EncoderDepthSplatRefiner
    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
    from src.misc.step_tracker import StepTracker
    from src.misc.benchmarker import Benchmarker
    from src.misc.image_io import save_image, save_video
    from src.visualization.vis_depth import viz_depth_tensor
    from src.model.ply_export import save_gaussian_ply

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

import os
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
from PIL import Image

# ================= 新增功能：生成环绕虚拟视角位姿 =================
def generate_orbit_poses(c2w: torch.Tensor, radius: float = 0.15, n_frames: int = 40):
    """
    围绕给定的 c2w 位姿构造一圈虚拟视角
    """
    device = c2w.device
    R_orig = c2w[:3, :3]  # [3, 3]
    t_orig = c2w[:3, 3]   # [3]
    angles = torch.linspace(0, 2 * math.pi, n_frames, device=device)
    
    # 在相机局部坐标系的 XY 平面生成圆环偏移
    dx = torch.cos(angles) * radius
    dy = torch.sin(angles) * radius
    offsets_local = torch.stack([dx, dy, torch.zeros_like(angles)], dim=-1) # [N, 3]
    
    # 转换到世界坐标系
    offsets_world = torch.einsum('ij, nj -> ni', R_orig, offsets_local) 
    
    # 构造位姿序列
    virtual_c2ws = repeat(c2w, 'i j -> n i j', n=n_frames).clone()
    virtual_c2ws[:, :3, 3] = t_orig + offsets_world
    return virtual_c2ws

def save_colmap_structure(batch, extra_context, sparse_dir, height, width, gaussians=None):
    """
    导出 COLMAP 数据结构，确保 input.ply 包含 positions, colors 和 normals
    以兼容 fetchPly 读取逻辑。
    """
    sparse_dir = Path(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # images 文件夹位于 sparse 上两级: output/colmap/scene/images
    root_dir = sparse_dir.parent.parent 
    images_dir = root_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cameras_file = sparse_dir / "cameras.txt"
    images_file = sparse_dir / "images.txt"
    # points3d_file = sparse_dir / "points3D.txt"
    ply_file = sparse_dir / "points3D.ply"

    print(f"Saving COLMAP data to {sparse_dir}...")

    # ================= 1. 重写的点云保存逻辑 (兼容 fetchPly) =================
    if gaussians is not None:
        # 1. 位置 (Positions) [N, 3]
        xyz = gaussians.means[0].detach().cpu().numpy()
        
        # 2. 法线 (Normals) [N, 3]
        # 高斯点云通常没有显式法线，为了兼容读取结构，填充为 0
        normals = np.zeros_like(xyz)
        
        # 3. 颜色 (Colors) [N, 3] -> uint8
        # 尝试从 SH 直流分量提取颜色
        if hasattr(gaussians, 'harmonics'):
            SH_C0 = 0.28209479177387814
            shs = gaussians.harmonics[0].detach().cpu().numpy()
            dc = shs[:, :, 0]
            # SH -> RGB 转换公式
            rgb = (dc * SH_C0 + 0.5)
            # 截断到 [0, 1] 并转为 uint8 [0, 255]
            rgb = np.clip(rgb, 0, 1) * 255
            rgb = rgb.astype(np.uint8)
        else:
            # 默认白色
            rgb = np.ones_like(xyz, dtype=np.uint8) * 255

        # 4. 构建符合 fetchPly 要求的结构化数组
        # 必须包含: x, y, z, nx, ny, nz, red, green, blue
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        
        elements = np.empty(xyz.shape[0], dtype=dtype)
        
        # 填充数据
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]
        
        elements['nx'] = normals[:, 0]
        elements['ny'] = normals[:, 1]
        elements['nz'] = normals[:, 2]
        
        elements['red'] = rgb[:, 0]
        elements['green'] = rgb[:, 1]
        elements['blue'] = rgb[:, 2]
        
        # 写入文件
        PlyData([PlyElement.describe(elements, 'vertex')]).write(str(ply_file))
        print(f"Saved input.ply with {xyz.shape[0]} points (with normals/colors).")
    
    # ================= 2. 处理图像与相机 (保持不变) =================
    all_views = []

    def process_and_save_views(data_dict, source_name):
        c2ws = data_dict["extrinsics"][0].detach().cpu().numpy()
        ks = data_dict["intrinsics"][0].detach().cpu().numpy()
        indices = data_dict["index"][0].detach().cpu().numpy()
        images_tensor = data_dict["image"][0].detach().cpu()

        for i in range(len(indices)):
            idx = int(indices[i])
            img_name = f"{idx:06d}.png"
            
            if any(v['id'] == idx for v in all_views):
                continue
            
            # 保存图片
            img_np = images_tensor[i].permute(1, 2, 0).numpy()
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img_np).save(images_dir / img_name)

            all_views.append({
                'c2w': c2ws[i], 'K': ks[i], 'id': idx, 
                'name': img_name, 'source': source_name
            })

    if "context" in batch: process_and_save_views(batch["context"], "context")
    if extra_context is not None: process_and_save_views(extra_context, "target")
    all_views.sort(key=lambda x: x['id'])

    # --- 写入 cameras.txt ---
    with open(cameras_file, "w") as f:
        f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        f.write(f"# Number of cameras: {len(all_views)}\n")
        
        for view in all_views:
            k = view['K']
            
            # 原始读取的值 (归一化的)
            fx_norm, fy_norm = k[0, 0], k[1, 1]
            cx_norm, cy_norm = k[0, 2], k[1, 2]
            
            # === 修正步骤：反归一化 ===
            if fx_norm < 10.0 and cx_norm < 1.0: 
                fx = fx_norm * width
                fy = fy_norm * height
                cx = cx_norm * width
                cy = cy_norm * height
            else:
                # 已经是像素单位了
                fx, fy, cx, cy = fx_norm, fy_norm, cx_norm, cy_norm

            f.write(f"{view['id']} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    # 写入 images.txt
    with open(images_file, "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        f.write(f"# Number of images: {len(all_views)}\n")
        for view in all_views:
            try:
                w2c = np.linalg.inv(view['c2w'])
                quat = R.from_matrix(w2c[:3, :3]).as_quat() # (x, y, z, w)
                t = w2c[:3, 3]
                f.write(f"{view['id']} {quat[3]} {quat[0]} {quat[1]} {quat[2]} {t[0]} {t[1]} {t[2]} {view['id']} {view['name']}\n\n")
            except: pass

    print("COLMAP export finished.")

@hydra.main(version_base=None, config_path="../config", config_name="main")
def inference(cfg_dict: DictConfig):
    # 1. 环境与配置
    cfg_dict["mode"] = "test"
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ablation study
    using_stereo_anchor = cfg.test.using_stereo_anchor
    using_gscompleter = cfg.test.using_gscompleter
    using_depth_alignment = cfg.test.using_depth_alignment
    using_rc_register = cfg.test.using_rc_register
    using_mv_refinement = cfg.test.using_mv_refinement
    using_target_align = cfg.test.using_target_align
    angle_deg = cfg.test.angle_deg

    output_dir = Path(cfg_dict.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 初始化核心组件
    step_tracker = StepTracker()
    benchmarker = Benchmarker()
    
    print(cyan("Initializing Encoder & Decoder..."))
    encoder = EncoderDepthSplatRefiner(cfg.model.encoder).to(device)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset).to(device)
    
    # === 新增：初始化 FID 指标计算器 ===
    # normalize=True 表示我们稍后会传入 uint8 [0,255]，它内部会自动处理到 inception 需要的格式
    # reset_real_features=False 允许跨 batch 累积特征
    print(cyan("Initializing FID Metric..."))
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # 数据格式转换器 (Data Shim)
    data_shim = get_data_shim(encoder)
    
    # 指标容器
    test_step_outputs = {"psnr": [], "ssim": [], "lpips": []}
    time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    # 3. 加载权重
    if cfg.checkpointing.pretrained_model is not None:
        print(cyan(f"Loading weights: {cfg.checkpointing.pretrained_model}"))
        ckpt = torch.load(cfg.checkpointing.pretrained_model, map_location=device)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        # 移除 state_dict 中的 "encoder." 或 "decoder." 前缀
        encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}, strict=False)
        decoder.load_state_dict({k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}, strict=False)

    # 4. 准备数据
    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    # 5. 推理循环
    encoder.eval()
    decoder.eval()
    
    print(cyan(f"\nStarting Inference Loop ({len(test_loader)} samples)..."))
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
        # if batch_idx > 10:
        #     break
        torch.manual_seed(42) # 固定种子
        np.random.seed(42)

        # Data Shim & Move to Device
        batch = data_shim(batch)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        for k in ["context", "target"]:
            batch[k] = {nk: nv.to(device) if isinstance(nv, torch.Tensor) else nv for nk, nv in batch[k].items()}

        (scene,) = batch["scene"]



        b, v, _, h, w = batch["target"]["image"].shape
        visualization_dump = {} if (cfg.test.save_depth or cfg.test.save_gaussian) else None

        if cfg.test.using_refinement:
            extra_context = {
                "extrinsics": batch["target"]["extrinsics"], 
                "intrinsics": batch["target"]["intrinsics"], 
                "image": batch["target"]["image"],           
                "near": batch["target"]["near"],             
                "far": batch["target"]["far"],               
                "index": batch["target"]["index"]            
            }
        else:
            extra_context = None

        # --- Encoder Step ---
        with benchmarker.time("encoder"):
            gaussians = encoder(
                batch["context"],
                global_step=0, 
                deterministic=False,
                visualization_dump=visualization_dump,
                extra_context=extra_context,
                scene_names=batch['scene'],
                decoder=decoder.forward,
                using_stereo_anchor=using_stereo_anchor,
                using_gscompleter=using_gscompleter,
                using_depth_alignment=using_depth_alignment,
                using_rc_register=using_rc_register,
                using_mv_refinement=using_mv_refinement,
                using_target_align=using_target_align,
                angle_deg=angle_deg,
            )
            if isinstance(gaussians, dict):
                pred_depths = gaussians["depths"]
                gaussians = gaussians["gaussians"]

        # --- Save Gaussian PLY ---
        if cfg.test.save_gaussian:
            save_path = output_dir / 'gaussians' / f"{scene}.ply"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_gaussian_ply(gaussians, visualization_dump, batch, save_path)

        # --- Decoder Step (Rendering) ---
        if not cfg.train.forward_depth_only:
            with benchmarker.time("decoder", num_calls=v):
                target_output = decoder.forward(
                    gaussians, batch["target"]["extrinsics"], batch["target"]["intrinsics"],
                    batch["target"]["near"], batch["target"]["far"], (h, w), depth_mode=None
                )
                images_prob = target_output.color

            # --- Save Results & Compute Metrics ---
            rgb_gt = batch["target"]["image"][0] # [V, 3, H, W]

            if cfg.test.compute_scores:

                test_step_outputs["psnr"].append(compute_psnr(rgb_gt, images_prob[0]).mean().item())
                test_step_outputs["ssim"].append(compute_ssim(rgb_gt, images_prob[0]).mean().item())
                test_step_outputs["lpips"].append(compute_lpips(rgb_gt, images_prob[0]).mean().item())

                fake_imgs = torch.clamp(images_prob[0].detach(), 0, 1)
                real_imgs = torch.clamp(rgb_gt.detach(), 0, 1)
                
                fake_imgs_uint8 = (fake_imgs * 255).to(dtype=torch.uint8)
                real_imgs_uint8 = (real_imgs * 255).to(dtype=torch.uint8)

                fid_metric.update(real_imgs_uint8, real=True)
                fid_metric.update(fake_imgs_uint8, real=False)

                if cfg.test.render_context:
                    with benchmarker.time("decoder_context"):
                        context_output = decoder.forward(
                            gaussians, batch["context"]["extrinsics"], batch["context"]["intrinsics"],
                            batch["context"]["near"], batch["context"]["far"], (h, w), depth_mode=None
                        )
                        context_images_prob = context_output.color
                        rgb_context_gt = batch["context"]["image"][0]

                    test_step_outputs.update({"ctx_psnr": [], "ctx_ssim": [], "ctx_lpips": []})
                    test_step_outputs["ctx_psnr"].append(compute_psnr(rgb_context_gt, context_images_prob[0]).mean().item())
                    test_step_outputs["ctx_ssim"].append(compute_ssim(rgb_context_gt, context_images_prob[0]).mean().item())
                    test_step_outputs["ctx_lpips"].append(compute_lpips(rgb_context_gt, context_images_prob[0]).mean().item())

            if cfg.test.save_virtual_video:

                base_c2w = batch["target"]["extrinsics"][0, 0]
                base_k = batch["target"]["intrinsics"][0, 0]
                orbit_c2ws = generate_orbit_poses(base_c2w, radius=0.15, n_frames=40)
                
                orbit_frames = []
                for i in range(len(orbit_c2ws)):
                    curr_c2w = orbit_c2ws[i:i+1].unsqueeze(0) # [1, 1, 4, 4]
                    v_out = decoder.forward(
                        gaussians, curr_c2w, base_k.view(1, 1, 3, 3), 
                        batch["target"]["near"][:, :1], batch["target"]["far"][:, :1], (h, w)
                    )
                    orbit_frames.append(v_out.color[0, 0].clamp(0, 1).cpu())

                scene_vid_dir = output_dir / "videos-frames" / scene / "color"
                scene_vid_dir.mkdir(parents=True, exist_ok=True)
                for idx, color in enumerate(orbit_frames):
                    save_image(color, scene_vid_dir / f"{idx:0>6}.png")

            if cfg.test.save_image:
                for idx, color in zip(batch["target"]["index"][0], images_prob[0]):
                    save_image(color, output_dir / "images" / scene / f"color/{idx:0>6}.png")
                    if cfg.test.save_gt_image:
                        save_image(rgb_gt[idx if isinstance(idx, int) else 0], output_dir / "images" / scene / f"color/{idx:0>6}_gt.png")

                if cfg.test.render_context:
                    ctx_dir = output_dir / "images" / scene / "context_color"
                    ctx_dir.mkdir(parents=True, exist_ok=True)
                    for idx, color in zip(batch["context"]["index"][0], context_images_prob[0]):
                        save_image(color, ctx_dir / f"{idx:0>6}.png")

            if cfg.test.save_colmap:
                colmap_sparse_dir = output_dir / "colmap_data" / scene / "sparse" / "0"
                h, w = batch["target"]["image"].shape[-2:]
                save_colmap_structure(batch, extra_context, colmap_sparse_dir, h, w, gaussians)

        # --- Depth Export ---
        if cfg.test.save_depth:
            depth = pred_depths[0].detach().cpu() if cfg.train.forward_depth_only else \
                    visualization_dump["depth"][0, :, :, :, 0, 0].detach().cpu()
            for idx, d_i in zip(batch["context"]["index"][0], depth):
                depth_viz = viz_depth_tensor(1.0 / d_i, return_numpy=True)
                save_path = output_dir / "images" / scene / "depth" / f"{idx:0>6}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(depth_viz).save(save_path)

    print(cyan("\nFinalizing Metrics..."))

    tto_opt = getattr(encoder, "tto_optimizer", None)
    geometry_summary = ""
    if tto_opt and hasattr(tto_opt, "evaluator"):
        geometry_summary = tto_opt.evaluator.summarize(save_path=str(output_dir / "geometry_metrics.txt"))

    final_results = {}
    for metric, scores in test_step_outputs.items():
        if scores:
            avg_score = np.mean(scores)
            final_results[metric] = avg_score
            print(f"{metric.upper()}: {avg_score:.4f}")

    if cfg.test.compute_scores:
        try:
            print("Computing FID... (This may take a while depending on VRAM/Dataset size)")

            fid_score = fid_metric.compute()
            fid_value = fid_score.item()
            
            print(f"FID: {fid_value:.4f}")

            final_results["fid"] = fid_value

            fid_metric.reset()
        except Exception as e:
            print(f"{Fore.RED}Failed to compute FID (possibly too few samples or OOM): {e}{Fore.RESET}")
            final_results["fid"] = -1.0 

    print(f"Fallback_cnt:{encoder._cnt_fallback}")

    benchmarker.dump(metrics_dir / "benchmark.json")

    summary_path = output_dir / "metrics_summary.txt"
    print(f"Saving metrics summary to {summary_path}")
    
    with open(summary_path, "w") as f:
        f.write("Final Report\n")
        f.write("====================\n")

        if geometry_summary:
            f.write(f"{geometry_summary}\n")

        for k, v in final_results.items():
            f.write(f"{k}: {v:.4f}\n")

        f.write(f"Fallback_cnt:{encoder._cnt_fallback}")

    backup_dir = output_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    if os.path.exists("src/model/encoder/encoder_refinedsplat.py"):
        shutil.copyfile("src/model/encoder/encoder_refinedsplat.py", os.path.join(backup_dir, "encoder_refinedsplat.py"))

    print(cyan(f"Done. Outputs saved to {output_dir}"))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')
    inference()
