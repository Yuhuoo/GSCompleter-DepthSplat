import os
from pathlib import Path
import warnings
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from jaxtyping import install_import_hook
from pytorch_lightning.plugins.environments import LightningEnvironment
from colorama import Fore
from tqdm import tqdm
import numpy as np
import time
import shutil

# Configure imports
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.model.decoder import get_decoder
    from src.model.model_wrapper import ModelWrapper
    # Import the new Refiner class
    from src.model.encoder.encoder_refinedsplat import EncoderDepthSplatRefiner

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

@hydra.main(version_base=None, config_path="../config", config_name="main")
def inference(cfg_dict: DictConfig):
    # --- 1. 初始化 (同前) ---
    cfg_dict["mode"] = "test"
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # 显式创建 metrics 目录，否则 on_test_end 可能会报错
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. 初始化模型 ---
    step_tracker = StepTracker()
    encoder = EncoderDepthSplatRefiner(cfg.model.encoder).to(device)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset).to(device)
    
    model_wrapper = ModelWrapper(
        cfg.optimizer, cfg.test, cfg.train, encoder, None,
        decoder, get_losses(cfg.loss), step_tracker, eval_data_cfg=cfg.dataset,
    ).to(device)

    # 加载权重
    if cfg.checkpointing.pretrained_model is not None:
        ckpt = torch.load(cfg.checkpointing.pretrained_model, map_location=device)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model_wrapper.load_state_dict(state_dict, strict=False)

    # --- 3. 准备数据 ---
    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    # --- 4. 手动 TTO 循环 ---
    model_wrapper.eval()
    print(cyan(f"Starting Manual TTO Loop..."))
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
        # if batch_idx > 0:
        #     break
        torch.manual_seed(42) # 固定种子
        np.random.seed(42)

        batch = model_wrapper.transfer_batch_to_device(batch, device, 0)
        model_wrapper.test_step(batch, batch_idx)

    # --- 5. 关键：手动触发汇总逻辑并保存 ---
    
    # # 定义汇总文件路径
    # summary_txt_path = output_dir / "tto_metrics_summary.txt"
    
    # A. 处理几何指标 (Geometric Metrics)
    geometry_summary = ""
    if hasattr(model_wrapper.encoder, "tto_optimizer"):
        print(cyan("\n[Geometry Summary]"))
        # 使用 str() 强制转换 Path 对象为字符串
        geometry_summary = model_wrapper.encoder.tto_optimizer.evaluator.summarize(
            save_path=str(output_dir / "geometry_metrics.txt")
        )

    # B. 调用原本的 on_test_end (保存 PSNR 等到 JSON)
    print(cyan("\n[Image Quality Summary]"))
    model_wrapper.test_cfg.output_path = metrics_dir 
    model_wrapper.on_test_end()

    # --- 6. 新增：备份源代码文件 ---
    backup_dir = output_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义源文件路径和目标路径
    src_file = Path("src/model/encoder/encoder_refinedsplat.py")
    dst_file = backup_dir / "encoder_refinedsplat.py"
    
    try:
        if src_file.exists():
            shutil.copy2(src_file, dst_file)  # copy2 会保留元数据（如修改时间）
            print(cyan(f"Source code backed up to: {dst_file}"))
        else:
            print(f"Warning: Source file {src_file} not found, skipping backup.")
    except Exception as e:
        print(f"Failed to backup source code: {e}")

if __name__ == "__main__":
    inference()