# Third party import
import click
import torch
import yaml
import os

# Local import
from jepa.utils.training_utils import get_model, get_trainer

@click.command()
@click.argument("cfg", type=str, required=True)
@click.option("--checkpoint-path", type=str, default=None)
@click.option("--checkpoint-resume-dir", type=str, default=None)
def main(cfg, checkpoint_path, checkpoint_resume_dir):
    with open(cfg, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(config["logdir"], exist_ok=True)
        
    model, config, default_root_dir = get_model(
        config, 
        checkpoint_path=checkpoint_path, 
        checkpoint_resume_dir=checkpoint_resume_dir
    )
    
    trainer = get_trainer(config, default_root_dir)
    
    # Add model watching if using WandB logger
    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.watch(model)
    
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model)

if __name__ == "__main__":
    main()