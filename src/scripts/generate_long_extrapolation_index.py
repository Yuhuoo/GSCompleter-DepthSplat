import os
import json
import torch
import argparse
from glob import glob
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass

# 保持原有的 IndexEntry 定义
@dataclass
class IndexEntry:
    context: list[int]
    target: list[int]

def process_scene_fixed(example, args):
    """
    修改后的采样逻辑：
    Context: 固定为索引 [0, 20]
    Target:  从索引 40 开始，每隔 20 帧取一个 (40, 60, 80...)
    """
    # 获取总帧数 (example["cameras"] shape is [batch, 18])
    n_views = example["cameras"].shape[0]
    
    # 1. 定义 Input (Context) 视图
    # 要求：必须包含第0帧和第20帧
    context_indices = [0, 20]
    
    # 如果视频太短，甚至没有第20帧（即总长<=20），则该场景无效，跳过
    if n_views <= 20:
        return None

    # 2. 定义 Target 视图
    # 逻辑：剩下每隔20个。理解为基于第20帧之后，继续以20为步长。
    # 例如：40, 60, 80 ...
    target_indices = list(range(40, n_views, 20))

    # 可选：如果要求必须有 Target 才能训练/测试
    # 如果总帧数是 30，虽然有 Context(0, 20)，但没有 Target(40)，这里选择跳过
    if len(target_indices) == 0:
        return None
        
    return {
        "context": context_indices,
        "target": target_indices
    }

def generate_index(args):
    out_dict_all = OrderedDict()
    
    chunk_paths = sorted(glob(os.path.join(args.data_dir, "*.torch")))
    
    for chunk_path in tqdm(chunk_paths, desc=f"Processing {args.type} datasets"):
        try:
            chunk = torch.load(chunk_path)
        except Exception as e:
            print(f"Error loading {chunk_path}: {e}")
            continue

        for example in chunk:
            scene_key = example["key"]
            
            # --- 使用新的处理函数 ---
            result = process_scene_fixed(example, args)
            
            if result:
                out_dict_all[scene_key] = result

    # 结果保存路径
    output_dir = Path(args.output_dir) / f"evaluation_index_fixed/{args.type}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 修改文件名以反映新的策略
    file_name = f"{args.type}_ctx_0_20_tgt_every_20.json"
    save_path = output_dir / file_name
    
    with open(save_path, "w") as f:
        json.dump(out_dict_all, f, indent=4)
    
    print(f"\nSuccessfully saved {len(out_dict_all)} scenes to {save_path}")
    print(f"Sample entry: {list(out_dict_all.values())[0] if out_dict_all else 'None'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed Stride Evaluation Index Generator")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the test data chunks")
    parser.add_argument("--type", type=str, choices=["dl3dv", "re10k", "acid"], required=True)
    parser.add_argument("--output_dir", type=str, default="assets", help="Root directory for saving indices")
    
    # 注意：之前的 num_context_views 等参数在这个逻辑下不再需要了，但为了兼容性可以保留或移除

    args = parser.parse_args()
    generate_index(args)


# python -m src.scripts.generate_long_extrapolation_index --data_dir datasets/re10k/test --type re10k --output_dir assets/long_seq
# python -m src.scripts.generate_long_extrapolation_index --data_dir datasets/dl3dv/test --type dl3dv --output_dir assets/long_seq