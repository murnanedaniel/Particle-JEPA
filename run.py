# Third party import
from argparse import ArgumentParser
import torch
import yaml
import os

# Local import
from jepa.utils.training_utils import get_model, get_trainer

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--cfg", type = str, required = True)
    parser.add_argument("--checkpoint-path", type = str, default=None)
    parser.add_argument("--checkpoint-resume-dir", type = str, default=None)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(config["logdir"], exist_ok=True)
        
    model, config, default_root_dir = get_model(
        config, 
        checkpoint_path=args.checkpoint_path, 
        checkpoint_resume_dir=args.checkpoint_resume_dir
    )
    
    trainer = get_trainer(config, default_root_dir)
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model)

if __name__ == "__main__":
    main()