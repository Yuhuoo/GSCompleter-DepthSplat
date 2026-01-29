import os
import json
import torch
import argparse
import numpy as np
import imageio
from glob import glob
from tqdm import tqdm
from pathlib import Path
from io import BytesIO
from PIL import Image
from scipy.spatial.transform import Rotation as R

# --- 核心辅助函数 ---

def get_c2w(cam_params):
    """提取外参 (W2C -> C2W)"""
    w2c_flat = cam_params[6:]
    w2c_mat = torch.eye(4, dtype=torch.float32)
    w2c_mat[:3] = w2c_flat.view(3, 4)
    return torch.inverse(w2c_mat).numpy()

def process_resize_and_crop(pil_img, intrinsics_norm, target_res=256):
    """
    复刻 Re10K 代码中的 rescale_and_crop 逻辑
    输出: PIL Image (256x256), (fx_new, fy_new, cx, cy)
    """
    w_in, h_in = pil_img.size
    fx, fy, cx, cy = intrinsics_norm
    
    # 1. 计算缩放因子
    scale_factor = max(target_res / h_in, target_res / w_in)
    
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    
    # 2. 缩放图片 (LANCZOS)
    img_scaled = pil_img.resize((w_scaled, h_scaled), Image.LANCZOS)
    
    # 3. 中心裁剪
    h_start = (h_scaled - target_res) // 2
    w_start = (w_scaled - target_res) // 2
    
    img_cropped = img_scaled.crop((
        w_start, 
        h_start, 
        w_start + target_res, 
        h_start + target_res
    ))
    
    # 4. 更新内参
    fx_new = fx * (w_scaled / target_res)
    fy_new = fy * (h_scaled / target_res)
    
    return img_cropped, (fx_new, fy_new, cx, cy)

def process_single_scene(example, output_root, target_res=256):
    """
    处理单个场景数据：解码、缩放、提取位姿、保存图片和视频
    """
    scene_key = example["key"]
    print(f"\n>>> Processing Scene: {scene_key}")
    
    # 准备目录
    scene_dir = output_root / f"{scene_key}"
    images_dir = scene_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    images_raw = example["images"]
    cameras_raw = example["cameras"]
    timestamps = example.get("timestamps", [int(i * 4e4) for i in range(len(images_raw))]) 

    # 获取第一帧的原始内参
    fx_0, fy_0, cx_0, cy_0 = cameras_raw[0, :4].tolist()
    
    cameras_out = []
    final_intrinsics = None

    # --- 1. 处理并保存所有单帧图片 ---
    for idx, (img_byte_tensor, cam_param) in enumerate(tqdm(zip(images_raw, cameras_raw), total=len(images_raw), desc=f"Saving frames ({scene_key})")):
        img_name = f"{idx:05d}.png"
        
        # A. 解码
        try:
            img_bytes = img_byte_tensor.numpy().tobytes()
            pil_img = Image.open(BytesIO(img_bytes))
        except Exception as e:
            print(f"Error decoding image {idx}: {e}")
            continue

        # B. 缩放 & 裁剪
        img_final, (fx_new, fy_new, cx_new, cy_new) = process_resize_and_crop(
            pil_img, 
            (fx_0, fy_0, cx_0, cy_0), 
            target_res=target_res
        )
        
        img_final.save(images_dir / img_name)
        
        if final_intrinsics is None:
            final_intrinsics = {
                "cx": cx_new, "cy": cy_new, "fx": fx_new, "fy": fy_new,
                "h": target_res, "w": target_res
            }

        # C. 外参计算
        c2w_mat = get_c2w(cam_param)
        r = R.from_matrix(c2w_mat[:3, :3])
        quat = r.as_quat().tolist()
        trans_vec = c2w_mat[:3, 3].tolist()
        
        cam_entry = {
            "cam_id": idx,
            "cam_quat": quat,
            "cam_trans": trans_vec,
            "cx": cx_new, "cy": cy_new, "fx": fx_new, "fy": fy_new,
            "image_name": img_name,
            "timestamp": int(timestamps[idx]) if idx < len(timestamps) else 0
        }
        cameras_out.append(cam_entry)

    # 保存 JSON
    with open(scene_dir / "intrinsics.json", "w") as f:
        json.dump(final_intrinsics, f, indent=4)
    with open(scene_dir / "cameras.json", "w") as f:
        json.dump(cameras_out, f, indent=4)

    # --- 2. 将生成的图片合成视频 ---
    video_path = images_dir / "video.mp4"
    frame_paths = sorted(images_dir.glob("*.png"))
    
    if len(frame_paths) > 0:
        try:
            with imageio.get_writer(video_path, fps=30, codec='libx264', quality=8) as writer:
                for frame_path in frame_paths: # 这里就不显示进度条了，避免太乱
                    image = imageio.imread(frame_path)
                    writer.append_data(image)
            print(f"Video saved to: {video_path}")
        except Exception as e:
             print(f"Error creating video: {e}")
    else:
        print("No frames found, skipping video creation.")

def extract_scenes(args):
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_dir)
    
    # 目标列表 (Hardcoded list as requested)
    target_keys_list = [
        # "7c7bc5285126e6ad",
        # "2c9018ef57c6b061",
        # "4b85062505816744",
        # "5c909c802eb6edb1",
        # "1bf49251fdd23cc0",
        # "6fdd15be262854f1",
        "3aae131a319acd17",
        "0f25241e37e16f56",
        # "2b43428fa1cf1a7e",
        # "3d8d29bf0d9f24a4",
    ]
    
    # 如果命令行传入了 keys，则覆盖默认列表
    if args.target_keys:
        target_keys_list = args.target_keys

    # 转为集合，方便快速查找和移除
    targets_remaining = set(target_keys_list)
    print(f"Looking for {len(targets_remaining)} scenes: {targets_remaining}")
    
    chunk_paths = sorted(data_dir.glob("*.torch"))
    
    # 遍历所有数据块
    for chunk_path in tqdm(chunk_paths, desc="Scanning chunks"):
        if not targets_remaining:
            print("All target scenes found! Exiting early.")
            break
            
        try:
            # 加载 chunk
            chunk = torch.load(chunk_path, map_location="cpu")
        except Exception as e:
            print(f"Skipping broken chunk {chunk_path}: {e}")
            continue
        
        # 遍历 chunk 里的所有 example
        for example in chunk:
            key = example["key"]
            if key in targets_remaining:
                # 找到目标！处理它
                process_single_scene(example, output_root, target_res=256)
                
                # 从待办列表中移除
                targets_remaining.remove(key)
                
                if not targets_remaining:
                    break

    if targets_remaining:
        print(f"\nWarning: The following scenes were NOT found: {targets_remaining}")
    else:
        print("\nAll Done! All scenes extracted successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Re10K .torch chunks directory")
    parser.add_argument("--output_dir", type=str, default="extracted_scenes", help="Output directory")
    parser.add_argument("--target_keys", nargs="+", help="Override the default list of keys")
    
    args = parser.parse_args()
    extract_scenes(args)

# 7c7bc5285126e6ad
# 2c9018ef57c6b061
# 4b85062505816744
# 5c909c802eb6edb1
# 1bf49251fdd23cc0
# 6fdd15be262854f1

# python -m src.scripts.extract_scene --data_dir datasets/re10k/test --output_dir sample_data_3
