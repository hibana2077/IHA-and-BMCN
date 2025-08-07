#!/usr/bin/env python3
"""
Hyperparameter sweep script for W&B sweeps
"""

import argparse
import yaml
import wandb
from pathlib import Path
from utils.config import Config, create_sweep_configs


def create_wandb_sweep_config():
    """Create W&B sweep configuration"""
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val/acc',
            'goal': 'maximize'
        },
        'parameters': {
            'kappa_init': {
                'values': [0.05, 0.1, 0.2, 0.5]
            },
            'momentum': {
                'values': [0.05, 0.1, 0.2]
            },
            'batch_size': {
                'values': [4, 8, 16]
            },
            'dataset_name': {
                'values': ['cotton80', 'soybean', 'soy_ageing_r1', 'soy_ageing_r3', 'soy_ageing_r4', 'soy_ageing_r5', 'soy_ageing_r6']
            },
            'variant': {
                'values': ['P4', 'P5']  # Focus on full proposals
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10
        }
    }
    return sweep_config


def train_with_sweep():
    """Training function for W&B sweep"""
    # Initialize wandb
    wandb.init()
    
    # Get sweep parameters
    config_dict = dict(wandb.config)
    
    # Load base config and update with sweep parameters
    base_config = Config()
    base_config.data.dataset_name = config_dict['dataset_name']
    base_config.experiment.variant = config_dict['variant']
    base_config.model.iha.kappa_init = config_dict['kappa_init']
    base_config.model.bmcn.momentum = config_dict['momentum']
    base_config.training.batch_size = config_dict['batch_size']
    
    # Update config for variant
    base_config.update_for_variant(config_dict['variant'])
    
    # Save config
    config_path = Path(f"temp_config_{wandb.run.id}.yaml")
    base_config.save_yaml(str(config_path))
    
    # Import and run training
    import subprocess
    import sys
    
    try:
        # Run training script
        cmd = [
            sys.executable, "train.py",
            "--config", str(config_path),
            "--gpu", "0"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise
    
    finally:
        # Clean up temp config
        if config_path.exists():
            config_path.unlink()


def main():
    parser = argparse.ArgumentParser(description='Run W&B hyperparameter sweep')
    parser.add_argument('--create', action='store_true', help='Create sweep and print sweep ID')
    parser.add_argument('--run', action='store_true', help='Run sweep agent')
    parser.add_argument('--sweep_id', type=str, help='Sweep ID to run agent for')
    parser.add_argument('--count', type=int, default=24, help='Number of runs for sweep')
    
    args = parser.parse_args()
    
    if args.create:
        # Create sweep
        sweep_config = create_wandb_sweep_config()
        sweep_id = wandb.sweep(sweep_config, project="IHA-BMCN-UFGVC")
        print(f"Created sweep with ID: {sweep_id}")
        print(f"Run with: python sweep.py --run --sweep_id {sweep_id}")
        
        # Save sweep config
        with open(f"sweep_config_{sweep_id}.yaml", 'w') as f:
            yaml.dump(sweep_config, f, default_flow_style=False)
    
    elif args.run:
        if not args.sweep_id:
            print("ERROR: --sweep_id required when using --run")
            return
        
        # Run sweep agent
        wandb.agent(args.sweep_id, train_with_sweep, count=args.count)
    
    else:
        print("Use --create to create a sweep or --run --sweep_id <ID> to run a sweep")


if __name__ == "__main__":
    main()
