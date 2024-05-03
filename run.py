# Third party import
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from argparse import ArgumentParser
import torch
import yaml

# Local import
from jepa.modules import Transformer

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--cfg", type = str, required = True)
    parser.add_argument("--accelerator", type = str, default="gpu")
    parser.add_argument("--devices", type = int, default=1)
    parser.add_argument("--epochs", type = int, default=1000)
    parser.add_argument("--train-batches", type = int, default=10000)
    parser.add_argument("--val-batches", type = int, default=1000)
    parser.add_argument("--gradient-clip-val", type = float, default=0.5)
    parser.add_argument("--num-sanity-val-steps", type = int, default=0)
    parser.add_argument("--log-period", type = int, default=50)
    parser.add_argument("--log-dir", type = str, default="../cfs/set_tracking")
    parser.add_argument("--project-name", type = str, default="set_tracking")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
        
    if cfg["model"] == "Transformer":
        model = Transformer(**cfg)
    else:
        raise NotImplementedError("model specified is not implemented")
        
    torch.compile(model)
    torch.set_float32_matmul_precision('medium')
    logger = WandbLogger(
        project = args.project_name, 
        save_dir = args.log_dir
    )
    checkpoint = ModelCheckpoint(
        filename="best", 
        monitor="validation_auc", 
        save_last=True, 
        save_top_k=1, 
        mode='max', 
        auto_insert_metric_name=False,
    )
    
    trainer = Trainer(
        accelerator=args.accelerator,
        gradient_clip_val=args.gradient_clip_val,
        devices=args.devices,
        max_epochs=args.epochs,
        limit_train_batches = args.train_batches,
        limit_val_batches = args.val_batches,
        num_sanity_val_steps = args.num_sanity_val_steps,
        logger=logger,
        callbacks=[checkpoint],
        log_every_n_steps = args.log_period,
        default_root_dir = args.log_dir,
        reload_dataloaders_every_n_epochs=1
    )
    trainer.fit(model)

if __name__ == "__main__":
    main()