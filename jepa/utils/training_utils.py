import os
import torch
from pathlib import Path
import wandb
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

# local imports
from jepa.modules import Transformer


def get_default_root_dir(logdir):
    if (
        "SLURM_JOB_ID" in os.environ
        and "SLURM_JOB_QOS" in os.environ
        and "interactive" not in os.environ["SLURM_JOB_QOS"]
        and "jupyter" not in os.environ["SLURM_JOB_QOS"]
    ):
        return os.path.join(logdir, os.environ["SLURM_JOB_ID"])
    else:
        return logdir

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
    metric_to_monitor = "validation_auc"
    metric_mode = config["metric_mode"] if "metric_mode" in config else "min"

    print(f"Setting default root dir: {default_root_dir}")
    resume = "allow"

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

    print(f"Job ID: {job_id}, resume: {resume}")

    # handle wandb logging
    logger = (
        WandbLogger(
            project=config["project"],
            save_dir=config["logdir"],
            id=job_id,
            name=job_id,
            group=config.get("group"),
            resume=resume,
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
    config, checkpoint_path=None, checkpoint_resume_dir=None
):
    # get a default_root_dr
    default_root_dir = get_default_root_dir(config["logdir"])

    # get the module
    if config["model"] == "Transformer":
        module = Transformer
    else:
        raise NotImplementedError("model specified is not implemented")

    # if resume from a previous run that fails, allow to specify a checkpoint_resume_dir that must contain checkpoints from previous run
    # if checkpoint_resume_dir exists and contains a checkpoint, set as default_root_dir
    if checkpoint_resume_dir is not None:
        if not os.path.exists(checkpoint_resume_dir):
            raise Exception(
                f"Checkpoint resume directory {checkpoint_resume_dir} does not exist."
            )
        if not find_latest_checkpoint(checkpoint_resume_dir, "*.ckpt"):
            raise Exception(
                "No checkpoint found in checkpoint resume directory"
                f" {checkpoint_resume_dir}."
            )
        default_root_dir = checkpoint_resume_dir

    # if default_root_dir contains checkpoint, use latest checkpoint as starting point, ignore the input checkpoint_path
    if default_root_dir is not None and find_latest_checkpoint(
        default_root_dir, "*.ckpt"
    ):
        checkpoint_path = find_latest_checkpoint(default_root_dir, "*.ckpt")

    # Load a checkpoint if checkpoint_path is not None
    if checkpoint_path is not None:
        model, config = load_module(checkpoint_path, module)
    else:
        model = module(**config)
    return model, config, default_root_dir


def load_module(checkpoint_path, module):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    config = checkpoint["hyper_parameters"]
    stage_module = module.load_from_checkpoint(
        checkpoint_path=checkpoint_path
    )
    return stage_module, config