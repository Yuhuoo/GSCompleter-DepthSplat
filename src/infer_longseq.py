import os
import json
import torch
import numpy as np
import hydra
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
from colorama import Fore
from jaxtyping import install_import_hook
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader
import cv2

# [RegGS 依赖]
from roma import unitquat_to_rotmat 

# ==============================================================================
# 1. 核心模块导入
# ==============================================================================
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder.encoder_refinedsplat import EncoderDepthSplatRefiner
    from src.evaluation.metrics import compute_psnr, compute_ssim, compute_lpips
    from src.misc.image_io import save_image

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

# ==============================================================================
# 2. 数据集定义 (Re10K)
# ==============================================================================
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_config: dict):
        self.dataset_path = Path(dataset_config["input_path"])
        self.height = dataset_config["H"]
        self.width = dataset_config["W"]

class Re10KDataset(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.scene_name = dataset_config["scene_name"]
        input_path = Path(self.dataset_path) / self.scene_name
        
        if not input_path.exists():
            raise FileNotFoundError(f"Scene path not found: {input_path}")
        
        self.load_dataset(input_path)

    def load_dataset(self, input_path: Path):
        image_dir = input_path / "images"
        if not image_dir.exists(): raise FileNotFoundError(f"Image dir not found: {image_dir}")
        self.images_paths = sorted(list(image_dir.glob("*.png")), key=lambda x: x.stem)

        intrinsics_path = input_path / "intrinsics.json"
        with intrinsics_path.open("r") as f: intrinsics = json.load(f)

        self.intrinsics_norm = np.eye(3, dtype=np.float32)
        self.intrinsics_norm[0, 0] = intrinsics["fx"]
        self.intrinsics_norm[1, 1] = intrinsics["fy"]
        self.intrinsics_norm[0, 2] = intrinsics["cx"]
        self.intrinsics_norm[1, 2] = intrinsics["cy"]

        cameras_path = input_path / "cameras.json"
        with cameras_path.open("r") as f: cameras = json.load(f)

        cam_quat = torch.tensor([cam["cam_quat"] for cam in cameras], dtype=torch.float32)
        cam_trans = torch.tensor([cam["cam_trans"] for cam in cameras], dtype=torch.float32)
        n = cam_quat.shape[0]

        rot_mat = unitquat_to_rotmat(cam_quat)
        c2ws = torch.stack([torch.eye(4, dtype=torch.float32)] * n)
        c2ws[..., :3, :3] = rot_mat
        c2ws[..., :3, 3] = cam_trans
        
        c2ws = torch.inverse(c2ws[0]) @ c2ws 
        self.poses = c2ws.numpy()

    def _load_image(self, index):
        image_path = self.images_paths[index]
        color_data = cv2.imread(str(image_path))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(color_data).float() / 255.0

    def __len__(self): return self.poses.shape[0]

    def get_frame_data(self, index, device):
        img = self._load_image(index) 
        img = rearrange(img, "h w c -> 1 1 c h w").to(device) 
        c2w = torch.from_numpy(self.poses[index]).float() 
        extrinsics = c2w.view(1, 1, 4, 4).to(device)
        intrinsics = torch.from_numpy(self.intrinsics_norm).float() 
        intrinsics = intrinsics.view(1, 1, 3, 3).to(device)
        near = torch.tensor(0.5).view(1, 1).to(device)
        far = torch.tensor(100.0).view(1, 1).to(device)
        return img, extrinsics, intrinsics, near, far

def images_to_video(image_dir: Path, output_video_path: Path, fps: int = 30):
    """
    读取指定目录下的所有 PNG 图片并合成 MP4 视频
    """
    image_paths = sorted(list(image_dir.glob("*.png")))
    if not image_paths:
        print(f"Warning: No images found in {image_dir}, skipping video generation.")
        return

    # 读取第一帧获取尺寸
    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        print("Error reading the first frame.")
        return
        
    height, width, layers = first_frame.shape
    
    # 初始化视频写入器 (mp4v 编码兼容性较好)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    print(f"[Video] Assembling {len(image_paths)} frames into {output_video_path.name}...")
    for img_path in tqdm(image_paths, desc="Encoding Video"):
        frame = cv2.imread(str(img_path))
        video.write(frame)

    video.release()
    print(f"[Video] Saved to {output_video_path}")

# ==============================================================================
# 3. 全轨迹评测函数 (已修改支持视频导出)
# ==============================================================================
@torch.no_grad()
def evaluate_trajectory(stage_name, decoder, global_gaussians, dataset, seq_indices, device, output_dir):
    print(f"\n{'-'*60}")
    print(f"[Evaluation] Rendering Full Trajectory ({stage_name})...")
    print(f"Eval Sequence (Stride=10): {seq_indices}")
    print(f"{'-'*60}")

    H, W = dataset.height, dataset.width
    save_dir = output_dir / stage_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    psnr_list = []
    full_results = []

    # 1. 渲染循环
    for idx in tqdm(seq_indices, desc=f"Rendering {stage_name}"):
        gt_img, extrinsics, intrinsics, near, far = dataset.get_frame_data(idx, device)
        
        render_out = decoder(
            global_gaussians, 
            extrinsics, intrinsics, near, far, 
            (H, W), depth_mode=None
        )
        pred_img = render_out.color 
        
        gt_in = gt_img.reshape(-1, 3, H, W)
        pred_in = pred_img.reshape(-1, 3, H, W)
        
        psnr = compute_psnr(gt_in, pred_in).mean().item()
        psnr_list.append(psnr)
        
        full_results.append({
            "frame": int(idx),
            "psnr": float(psnr)
        })

        # 保存图片，文件名格式为 0001.png，方便排序
        save_image(pred_in[0], save_dir / f"{idx:04d}.png")
        save_image(gt_in[0], save_dir / f"{idx:04d}_gt.png")


    # 2. 计算平均指标
    avg_psnr = np.mean(psnr_list) if psnr_list else 0.0
    print(f"[{stage_name}] Avg PSNR: {cyan(f'{avg_psnr:.4f}')} dB")
    
    # 3. 保存 JSON 数据
    json_path = output_dir / f"{stage_name}_metrics.json"
    save_data = {
        "avg_psnr": float(avg_psnr),
        "trajectory_details": full_results
    }
    # import pdb; pdb.set_trace()
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=4)
    print(f"[IO] Saved metrics to {json_path}")

    # =========================================================
    # 4. [新增] 视频组装逻辑
    # =========================================================
    video_output_path = output_dir / f"{stage_name}.mp4"
    images_to_video(save_dir, video_output_path, fps=30)
    
    return avg_psnr

# ==============================================================================
# 4. 推理主程序
# ==============================================================================
@hydra.main(version_base=None, config_path="../config", config_name="main")
def inference(cfg_dict: DictConfig):
    cfg_dict["mode"] = "test"
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = Path(cfg_dict.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(cyan("Initializing Encoder & Decoder..."))
    encoder = EncoderDepthSplatRefiner(cfg.model.encoder).to(device)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset).to(device)
    encoder.eval()
    decoder.eval()

    if cfg.checkpointing.pretrained_model is not None:
        print(cyan(f"Loading weights: {cfg.checkpointing.pretrained_model}"))
        ckpt = torch.load(cfg.checkpointing.pretrained_model, map_location=device)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}, strict=False)
        decoder.load_state_dict({k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}, strict=False)

    root = Path(cfg.dataset.roots[0])
    if cfg.dataset.overfit_to_scene:
        scene_name = cfg.dataset.overfit_to_scene
    else:
        scene_name = sorted([p.name for p in root.iterdir() if p.is_dir()])[0]

    dataset_config = {
        "input_path": str(root), "scene_name": scene_name,
        "H": cfg.dataset.image_shape[0], "W": cfg.dataset.image_shape[1], 
        "fx": 0, "fy": 0, "cx": 0, "cy": 0, "depth_scale": 1.0, 
    }
    
    print(cyan(f"Loading Scene: {scene_name}"))
    dataset = Re10KDataset(dataset_config)
    # total_frames = len(dataset)
    total_frames = 100
    
    # # 1. 设定输入帧：第0帧和第5帧
    # ctx_indices = [0, 5]
    
    # # 2. 设定补全帧：剩下的每隔20帧 (从输入的最后一帧10开始往后算，即从30开始)
    # completion_stride = 20
    # start_frame = ctx_indices[-1] + completion_stride
    # tgt_indices = list(range(start_frame, total_frames, completion_stride))
    
    # # 检查序列长度是否足够
    # if not tgt_indices and len(ctx_indices) < 2:
    #     print("Sequence too short!")
    #     return

    # # 3. 设定测试视角
    # # 截止到最后一个补全帧 (如果没有补全帧，则截止到最后一个输入帧)
    # max_frame = tgt_indices[-1] if len(tgt_indices) > 0 else ctx_indices[-1]
    # all_frames_in_range = list(range(0, max_frame + 1))
    # exclude_set = set(ctx_indices + tgt_indices)
    # eval_indices = [idx for idx in all_frames_in_range if idx not in exclude_set]

    # 初始化训练帧与测试帧
    sample_rate = 8
    n_views = 12
    n_frames = total_frames
    frame_ids = np.arange(n_frames)

    test_frame_ids = frame_ids[int(sample_rate/2)::sample_rate]
    remain_frame_ids = np.array(
        [i for i in frame_ids if i not in test_frame_ids])
    train_frame_ids = remain_frame_ids[np.linspace(
        0, remain_frame_ids.shape[0] - 1, n_views).astype(int)]

    max_frame = n_frames
    eval_indices = test_frame_ids
    ctx_indices = [train_frame_ids[0], train_frame_ids[1]]
    tgt_indices = train_frame_ids[2:]

    print(f"Max frame used: {max_frame}")              
    print(f"Evaluation Sequence:   {eval_indices}")
    print(f"Train Sequence:        {train_frame_ids}")
    print(f"Initial Context:       {ctx_indices}")
    print(f"Online Targets:        {tgt_indices}")

    # ==========================================================================
    # Phase 1: Initialization
    # ==========================================================================
    print(cyan("\n[Phase 1] Initialization with Context Frames..."))
    
    ctx_imgs, ctx_exts, ctx_ints, ctx_nears, ctx_fars = [], [], [], [], []
    for idx in ctx_indices:
        i, e, intr, n, f = dataset.get_frame_data(idx, device)
        ctx_imgs.append(i); ctx_exts.append(e); ctx_ints.append(intr); ctx_nears.append(n); ctx_fars.append(f)
    
    context_batch = {
        "image": torch.cat(ctx_imgs, dim=1),
        "extrinsics": torch.cat(ctx_exts, dim=1),
        "intrinsics": torch.cat(ctx_ints, dim=1),
        "near": torch.cat(ctx_nears, dim=1),
        "far": torch.cat(ctx_fars, dim=1),
        "index": torch.tensor([ctx_indices]).to(device)
    }

    with torch.no_grad():
        global_gaussians = encoder(
            context=context_batch,
            global_step=0,
            deterministic=False,
            scene_names=[scene_name],
            extra_context=None, 
            decoder=decoder.forward
        )
        if isinstance(global_gaussians, dict): global_gaussians = global_gaussians["gaussians"]

    evaluate_trajectory(
        "eval_initial", decoder, global_gaussians, dataset, eval_indices, device, output_dir
    )

    # ==========================================================================
    # Phase 2: Compute Baseline (Pre-PSNR)
    # ==========================================================================
    print(cyan("\n[Phase 2] Computing Baseline (Pre-PSNR) for Optimization Targets..."))
    
    pre_psnr_dict = {}
    H, W = cfg.dataset.image_shape
    
    for t_idx in tqdm(train_frame_ids, desc="Computing Pre-PSNR"):
        tgt_img, tgt_ext, tgt_int, tgt_near, tgt_far = dataset.get_frame_data(t_idx, device)
        
        with torch.no_grad():
            render_pre = decoder(global_gaussians, tgt_ext, tgt_int, tgt_near, tgt_far, (H, W), depth_mode=None)
            psnr_pre = compute_psnr(
                tgt_img.reshape(-1, 3, H, W), 
                render_pre.color.reshape(-1, 3, H, W)
            ).mean().item()
            pre_psnr_dict[t_idx] = psnr_pre

    # ==========================================================================
    # Phase 3: Batch Inference
    # ==========================================================================
    print(cyan("\n[Phase 3] Running Encoder on Target Sequence..."))

    tgt_imgs, tgt_exts, tgt_ints, tgt_nears, tgt_fars, tgt_idxs_list = [], [], [], [], [], []
    
    for t_idx in tgt_indices:
        i, e, intr, n, f = dataset.get_frame_data(t_idx, device)
        tgt_imgs.append(i); tgt_exts.append(e); tgt_ints.append(intr); 
        tgt_nears.append(n); tgt_fars.append(f)
        tgt_idxs_list.append(t_idx)

    extra_context_batch = {
        "image": torch.cat(tgt_imgs, dim=1),
        "extrinsics": torch.cat(tgt_exts, dim=1),
        "intrinsics": torch.cat(tgt_ints, dim=1),
        "near": torch.cat(tgt_nears, dim=1),
        "far": torch.cat(tgt_fars, dim=1),
        "index": torch.tensor([tgt_idxs_list]).to(device)
    }
    
    with torch.no_grad():
        return_time = True
        result_dict = encoder(
            context=context_batch,    
            global_step=0,
            deterministic=False,
            scene_names=[scene_name],
            extra_context=extra_context_batch, 
            decoder=decoder.forward,
            using_stereo_anchor=cfg.test.using_stereo_anchor,
            using_depth_alignment=cfg.test.using_depth_alignment,
            using_rc_register=cfg.test.using_rc_register,
            using_mv_refinement=cfg.test.using_mv_refinement,
            using_target_align=cfg.test.using_target_align,
            return_time=return_time
        )
        
        if isinstance(result_dict, dict): 
            global_gaussians = result_dict["gaussians"]
            cost_time_list = result_dict.get("cost_time_list", [])
        else:
            global_gaussians = result_dict
            cost_time_list = []

    # ==========================================================================
    # Phase 4: Compute Refined (Post-PSNR)
    # ==========================================================================
    print(cyan("\n[Phase 4] Computing Refined (Post-PSNR) & Assembling Metrics..."))
    
    online_metrics = []
    
    if len(cost_time_list) != len(tgt_indices):
        print(f"Warning: Cost time list length ({len(cost_time_list)}) != Target length ({len(tgt_indices)})")
        cost_time_list += [0.0] * (len(tgt_indices) - len(cost_time_list))
    
    for i, t_idx in enumerate(tqdm(train_frame_ids, desc="Computing Post-PSNR")):
        tgt_img, tgt_ext, tgt_int, tgt_near, tgt_far = dataset.get_frame_data(t_idx, device)
        
        with torch.no_grad():
            render_post = decoder(global_gaussians, tgt_ext, tgt_int, tgt_near, tgt_far, (H, W), depth_mode=None)
            psnr_post = compute_psnr(
                tgt_img.reshape(-1, 3, H, W), 
                render_post.color.reshape(-1, 3, H, W)
            ).mean().item()
            
            save_image(render_post.color[0, 0], output_dir / f"online_step_{t_idx}_post.png")

        post_str = cyan(f"{psnr_post:.2f}")
        print(f"  Frame {t_idx} | Pre: {pre_psnr_dict[t_idx]:.2f} | Post: {post_str} | Time: {cost_time_list[i]:.4f}")

        metric_entry = {
            "frame": int(t_idx),
            "pre": pre_psnr_dict[t_idx],
            "post": psnr_post,
            "time": cost_time_list[i]
        }
        online_metrics.append(metric_entry)

    with open(output_dir / "online_metrics.json", "w") as f:
        json.dump(online_metrics, f, indent=4)

    # ==========================================================================
    # [Evaluation 2] Final Trajectory
    # ==========================================================================
    evaluate_trajectory(
        "eval_final", decoder, global_gaussians, dataset, eval_indices, device, output_dir
    )

    print(cyan(f"\nDone. All metrics and images saved to {output_dir}"))

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    inference()