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
    # Force test mode
    cfg_dict["mode"] = "test"
    
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
    print(cyan(f"Saving inference outputs to {output_dir}."))

    logger = LocalLogger()
    step_tracker = StepTracker()

    # Important: inference_mode=False allows gradients for Test-Time Refinement (TTR)
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices=torch.cuda.device_count(),
        strategy="auto",
        enable_progress_bar=True,
        num_nodes=cfg.trainer.num_nodes,
        plugins=LightningEnvironment() if cfg.use_plugins else None,
        inference_mode=False, 
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    # --- Manually Initialize the Refiner Encoder ---
    # Instead of get_encoder(), we explicitly instantiate our new class
    # assuming cfg.model.encoder contains the config compatible with EncoderDepthSplatCfg
    print(cyan("Initializing EncoderDepthSplatRefiner..."))
    encoder = EncoderDepthSplatRefiner(cfg.model.encoder)
    
    # We don't necessarily need the visualizer for inference logic, 
    # but we can initialize it if needed via standard get_encoder logic or skip it.
    encoder_visualizer = None 

    # Initialize Model Wrapper
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker,
        eval_data_cfg=cfg.dataset,
    )

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    checkpoint_path = cfg.checkpointing.load
    strict_load = not cfg.checkpointing.no_strict_load

    # Load weights logic (similar to main.py but adapted)
    if cfg.checkpointing.pretrained_model is not None:
        print(cyan(f"Loading pretrained model from: {cfg.checkpointing.pretrained_model}"))
        pretrained_model = torch.load(cfg.checkpointing.pretrained_model, map_location='cpu')
        if 'state_dict' in pretrained_model:
            pretrained_model = pretrained_model['state_dict']
        
        # We need to handle potential key mismatches if the refiner introduces new parameters
        # that aren't in the checkpoint, or if we are loading a base checkpoint.
        missing_keys, unexpected_keys = model_wrapper.load_state_dict(pretrained_model, strict=False)
        
        if len(missing_keys) > 0:
            print(f"Missing keys (Expected if Refiner is new): {missing_keys[:5]} ...")
    
    # Run Test
    trainer.test(
        model_wrapper,
        datamodule=data_module,
        ckpt_path=checkpoint_path, # usually None if we manually loaded weights above
    )

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')
    inference()