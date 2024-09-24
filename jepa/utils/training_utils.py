import os
import torch
from pathlib import Path
import wandb
from torch import nn
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import importlib
from typing import Tuple, Dict, Any, Optional
from datetime import datetime  # New import added

def get_default_root_dir(logdir: str, resume: bool = False) -> str:
    """
    Determines the default root directory for logging and checkpoints.
    Appends a date-time stamp to the logdir to ensure uniqueness for new runs.

    Args:
        logdir (str): Base directory for logs and checkpoints.
        resume (bool): Whether to resume from the latest checkpoint.

    Returns:
        str: The determined root directory.
    """
    if (
        "SLURM_JOB_ID" in os.environ
        and "SLURM_JOB_QOS" in os.environ
        and "interactive" not in os.environ["SLURM_JOB_QOS"]
        and "jupyter" not in os.environ["SLURM_JOB_QOS"]
    ):
        return os.path.join(logdir, os.environ["SLURM_JOB_ID"])
    else:
        if resume:
            return logdir
        else:
            # Append current date-time to logdir for a unique run directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_logdir = os.path.join(logdir, timestamp)
            os.makedirs(unique_logdir, exist_ok=True)
            return unique_logdir

def find_latest_checkpoint(checkpoint_base, templates=None):
    if templates is None:
        templates = ["*.ckpt"]
    elif isinstance(templates, str):
        templates = [templates]
    checkpoint_paths = []
    for template in templates:
        checkpoint_paths = checkpoint_paths or [
            str(path) for path in Path(checkpoint_base).rglob(template)
        ]
    return max(checkpoint_paths, key=os.path.getctime) if checkpoint_paths else None


def get_trainer(config, default_root_dir):
    metric_to_monitor = config.get("metric_to_monitor", "val_loss")
    metric_mode = config.get("metric_mode", "min")

    print(f"Setting default root dir: {default_root_dir}")

    job_id = (
        os.environ["SLURM_JOB_ID"]
        if "SLURM_JOB_ID" in os.environ
        and "SLURM_JOB_QOS" in os.environ
        and "interactive" not in os.environ["SLURM_JOB_QOS"]
        and "jupyter" not in os.environ["SLURM_JOB_QOS"]
        else None
    )

    if (
        isinstance(default_root_dir, str)
        and find_latest_checkpoint(default_root_dir) is not None
    ):
        print(
            f"Found checkpoint from a previous run in {default_root_dir}, resuming from"
            f" {find_latest_checkpoint(default_root_dir)}"
        )

    print(f"Job ID: {job_id}")

    # handle wandb logging
    logger = (
        WandbLogger(
            project=config["project"],
            save_dir=config["logdir"],
            id=job_id,
            name=job_id,
            group=config.get("group"),
            resume="allow",
        )
        if wandb is not None and config.get("log_wandb", True)
        else CSVLogger(save_dir=config["logdir"])
    )

    filename_suffix = (
        str(logger.experiment.id)
        if (
            hasattr(logger, "experiment")
            and hasattr(logger.experiment, "id")
            and logger.experiment.id is not None
        )
        else ""
    )
    filename = "best-" + filename_suffix + "-{" + metric_to_monitor + ":5f}-{epoch}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["logdir"], "artifacts"),
        filename=filename,
        monitor=metric_to_monitor,
        mode=metric_mode,
        save_top_k=config.get("save_top_k", 1),
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"last-{filename_suffix}"

    return Trainer(
        accelerator=config.get("accelerator"),
        gradient_clip_val=config.get("gradient_clip_val"),
        devices=config.get("devices"),
        num_nodes=config.get("nodes"),
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback],
        logger=logger,
        limit_train_batches=config.get("train_batches"),
        limit_val_batches=config.get("val_batches"),
        strategy="ddp",
        default_root_dir=default_root_dir,
        reload_dataloaders_every_n_epochs=1
    )

def get_model(
    config: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    checkpoint_resume_dir: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, Any], str]:
    """
    Dynamically loads and returns a model based on the configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        checkpoint_path (Optional[str], optional): Path to the checkpoint file. Defaults to None.
        checkpoint_resume_dir (Optional[str], optional): Directory to resume checkpoint from. Defaults to None.

    Returns:
        Tuple[nn.Module, Dict[str, Any], str]: Instantiated model, updated configuration, and root directory.
    """
    # Determine if we are resuming
    resume = checkpoint_path is not None or checkpoint_resume_dir is not None

    # Get the default root directory
    default_root_dir = get_default_root_dir(config.get("logdir"), resume=resume)

    # Extract the full model path from config
    model_full_path = config.get("model")
    if not model_full_path:
        raise ValueError("The 'model' key must be specified in the config with the full import path.")

    try:
        # Split the full path into module and class
        module_path, class_name = model_full_path.rsplit(".", 1)
    except ValueError:
        raise ValueError("The 'model' key must be a full import path, e.g., 'jepa.modules.JEPA'.")

    try:
        # Dynamically import the module
        model_module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}': {e}")

    try:
        # Get the class from the module
        model_class = getattr(model_module, class_name)
    except AttributeError:
        raise ImportError(f"Module '{module_path}' does not have a class named '{class_name}'.")

    # Handle checkpoint resume directory
    if checkpoint_resume_dir is not None:
        if not os.path.exists(checkpoint_resume_dir):
            raise FileNotFoundError(f"Checkpoint resume directory '{checkpoint_resume_dir}' does not exist.")
        latest_checkpoint = find_latest_checkpoint(checkpoint_resume_dir, "*.ckpt")
        if not latest_checkpoint:
            raise FileNotFoundError(f"No checkpoint found in resume directory '{checkpoint_resume_dir}'.")
        default_root_dir = checkpoint_resume_dir
        checkpoint_path = latest_checkpoint

    # Use the latest checkpoint in the default_root_dir if available and not explicitly specified
    if not resume and default_root_dir and find_latest_checkpoint(default_root_dir, "*.ckpt"):
        checkpoint_path = find_latest_checkpoint(default_root_dir, "*.ckpt")
        resume = True

    # Load the checkpoint if provided
    if checkpoint_path is not None:
        model, config = load_module(checkpoint_path, model_class)
    else:
        # Instantiate the model with the remaining config parameters
        model = model_class(**{k: v for k, v in config.items() if k != "model"})
    
    return model, config, default_root_dir


def load_module(checkpoint_path, module):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    config = checkpoint["hyper_parameters"]
    stage_module = module.load_from_checkpoint(
        checkpoint_path=checkpoint_path
    )
    return stage_module, config