import os
import json
import torch
import argparse
from glob import glob
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from dataclasses import asdict, dataclass
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

# 导入项目内部依赖
from ..dataset.view_sampler.view_sampler_bounded_v2 import farthest_point_sample

@dataclass
class IndexEntry:
    context: list[int]
    target: list[int]

def convert_poses(
    poses: Float[Tensor, "batch 18"],
) -> Float[Tensor, "batch 4 4"]:
    """将原始相机参数转换为 OpenCV 风格的外参矩阵 (W2C -> C2W)"""
    b = poses.shape[0]
    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse()

def partition_list(lst: list, n_bins: int) -> list:
    """分桶逻辑：从列表中均匀选取 n_bins 个索引作为支撑视角"""
    if n_bins <= 0 or len(lst) < n_bins:
        return None
    bin_size = len(lst) // n_bins
    return [lst[i * bin_size] for i in range(n_bins)]

def process_scene(example, args):
    """核心采样逻辑：FPS 选 Target (外推目标)，分桶选 Context (输入视角)"""
    extrinsics = convert_poses(example["cameras"])
    n_views = extrinsics.shape[0]
    
    # 限制搜索的视角范围
    search_scope = min(n_views, args.view_selection_num) if args.view_selection_num > 0 else n_views

    # 1. 使用最远点采样 (FPS) 锁定最极端的视角作为【补全目标】
    target_indices = sorted(
        farthest_point_sample(
            extrinsics[:search_scope, :3, -1].unsqueeze(0), 
            args.num_target_views
        ).squeeze(0).tolist()
    )

    # 2. 从剩余视角中选取【输入视角】
    remaining_indices = [i for i in range(search_scope) if i not in target_indices]
    context_indices = partition_list(remaining_indices, args.num_context_views)

    if context_indices is None:
        return None
        
    return {
        "context": context_indices,
        "target": target_indices
    }

def generate_index(args):
    out_dict_all = OrderedDict()
    
    # 查找所有 .torch 数据块 (适用于 DL3DV, RE10K, ACID 的标准存储格式)
    chunk_paths = sorted(glob(os.path.join(args.data_dir, "*.torch")))
    
    for chunk_path in tqdm(chunk_paths, desc=f"Processing {args.type} datasets"):
        chunk = torch.load(chunk_path)
        for example in chunk:
            scene_key = example["key"]
            result = process_scene(example, args)
            if result:
                out_dict_all[scene_key] = result

    # 结果保存
    output_dir = Path(args.output_dir) / f"evaluation_index_extrapolation/{args.type}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = f"{args.type}_ctx{args.num_context_views}_tgt{args.num_target_views}_n{args.view_selection_num}.json"
    save_path = output_dir / file_name
    
    with open(save_path, "w") as f:
        json.dump(out_dict_all, f, indent=4)
    
    print(f"\nSuccessfully saved {len(out_dict_all)} scenes to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Evaluation Index Generator for NVS Extrapolation")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the test data chunks (.torch files)")
    parser.add_argument("--type", type=str, choices=["dl3dv", "re10k", "acid"], required=True)
    parser.add_argument("--num_context_views", type=int, default=2, help="Number of input views (context)")
    parser.add_argument("--num_target_views", type=int, default=1, help="Number of views to complete (target)")
    parser.add_argument("--view_selection_num", type=int, default=100, help="Search range for view selection")
    parser.add_argument("--output_dir", type=str, default="assets", help="Root directory for saving indices")

    args = parser.parse_args()
    generate_index(args)

# --- CLI Usage Examples ---
# python -m src.scripts.generate_extrapolation_index --data_dir=datasets/dl3dv/test --type=dl3dv --num_context_views=2 --num_target_views=1 --view_selection_num=30
# python -m src.scripts.generate_extrapolation_index --data_dir=datasets/re10k/test --type=re10k --num_context_views=2 --num_target_views=1 --view_selection_num=30
# python -m src.scripts.generate_extrapolation_index --data_dir=datasets/acid/test --type=acid --num_context_views=2 --num_target_views=1 --view_selection_num=30