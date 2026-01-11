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
from einops import rearrange

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

@hydra.main(version_base=None, config_path="../config", config_name="main")
def inference(cfg_dict: DictConfig):
    # 1. 环境与配置
    cfg_dict["mode"] = "test"
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ablation study
    using_stereo_anchor = cfg.test.using_stereo_anchor
    using_depth_alignment = cfg.test.using_depth_alignment
    using_rc_register = cfg.test.using_rc_register
    using_mv_refinement = cfg.test.using_mv_refinement

    output_dir = Path(cfg_dict.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 初始化核心组件 (跳过 ModelWrapper)
    step_tracker = StepTracker()
    benchmarker = Benchmarker()
    
    print(cyan("Initializing Encoder & Decoder..."))
    encoder = EncoderDepthSplatRefiner(cfg.model.encoder).to(device)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset).to(device)
    
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
        # 移除 state_dict 中的 "encoder." 或 "decoder." 前缀以匹配直接定义的模型
        # 或者使用 model_wrapper 结构兼容加载
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
        # if batch_idx > 0:
        #     break
        torch.manual_seed(42) # 固定种子
        np.random.seed(42)

        # Data Shim & Move to Device
        batch = data_shim(batch)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # 特殊处理 context/target 嵌套字典
        for k in ["context", "target"]:
            batch[k] = {nk: nv.to(device) if isinstance(nv, torch.Tensor) else nv for nk, nv in batch[k].items()}

        (scene,) = batch["scene"]
        b, v, _, h, w = batch["target"]["image"].shape
        visualization_dump = {} if (cfg.test.save_depth or cfg.test.save_gaussian) else None

        if cfg.test.using_refinement:
            extra_context = {
                "extrinsics": batch["target"]["extrinsics"], # [1, V, 4, 4]
                "intrinsics": batch["target"]["intrinsics"], # [1, V, 3, 3]
                "image": batch["target"]["image"],           # [1, V, 3, H, W]
                "near": batch["target"]["near"],             # [1, V]
                "far": batch["target"]["far"],               # [1, V]
                "index": batch["target"]["index"]            # [1, V]
            }
        else:
            extra_context = None

        # --- Encoder Step ---
        with benchmarker.time("encoder"):
            gaussians = encoder(
                batch["context"],
                global_step=0, # 推理时 step 设为 0
                deterministic=False,
                visualization_dump=visualization_dump,
                extra_context=extra_context,
                scene_names=batch['scene'],
                decoder=decoder.forward,
                using_stereo_anchor=using_stereo_anchor,
                using_depth_alignment=using_depth_alignment,
                using_rc_register=using_rc_register,
                using_mv_refinement=using_mv_refinement,
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
            rgb_gt = batch["target"]["image"][0]

            if cfg.test.compute_scores:
                # Target 视角
                test_step_outputs["psnr"].append(compute_psnr(rgb_gt, images_prob[0]).mean().item())
                test_step_outputs["ssim"].append(compute_ssim(rgb_gt, images_prob[0]).mean().item())
                test_step_outputs["lpips"].append(compute_lpips(rgb_gt, images_prob[0]).mean().item())

                # Context 视角
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

            # 保存图像
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

        # --- Depth Export ---
        if cfg.test.save_depth:
            depth = pred_depths[0].detach().cpu() if cfg.train.forward_depth_only else \
                    visualization_dump["depth"][0, :, :, :, 0, 0].detach().cpu()
            for idx, d_i in zip(batch["context"]["index"][0], depth):
                depth_viz = viz_depth_tensor(1.0 / d_i, return_numpy=True)
                save_path = output_dir / "images" / scene / "depth" / f"{idx:0>6}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(depth_viz).save(save_path)

    # 6. 汇总与备份 (提取自 on_test_end)
    print(cyan("\nFinalizing Metrics..."))
    
    # 几何指标汇总
    tto_opt = getattr(encoder, "tto_optimizer", None)
    geometry_summary = ""
    if tto_opt and hasattr(tto_opt, "evaluator"):
        geometry_summary = tto_opt.evaluator.summarize(save_path=str(output_dir / "geometry_metrics.txt"))

    # 图像指标汇总
    final_results = {}
    for metric, scores in test_step_outputs.items():
        if scores:
            final_results[metric] = np.mean(scores)
            print(f"{metric.upper()}: {final_results[metric]:.4f}")

    # 保存 Benchmark
    benchmarker.dump(metrics_dir / "benchmark.json")
    
    # 保存汇总 TXT
    with open(output_dir / "tto_metrics_summary.txt", "w") as f:
        f.write(f"Final Report\n{geometry_summary}\n")
        for k, v in final_results.items(): f.write(f"{k}: {v:.4f}\n")

    # 源代码备份
    backup_dir = output_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    shutil.copyfile("src/model/encoder/encoder_refinedsplat.py", os.path.join(backup_dir, "encoder_refinedsplat.py"))

    print(cyan(f"Done. Outputs saved to {output_dir}"))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')
    inference()