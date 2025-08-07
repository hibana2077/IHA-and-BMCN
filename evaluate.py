#!/usr/bin/env python3
"""
Evaluation script for trained IHA+BMCN models
"""

import argparse
import torch
import yaml
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import timm

from src.iha import iha_act_layer
from src.bmcn import bmcn_norm_layer
from src.dataset.ufgvc import UFGVCDataset
from utils.metrics import MetricsCalculator
from utils.config import Config


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint['config']
    config = Config.from_dict(config_dict)
    
    # Determine number of classes from checkpoint
    state_dict = checkpoint['model_state_dict']
    classifier_key = None
    for key in state_dict.keys():
        if 'classifier' in key or 'head' in key or 'fc' in key:
            if 'weight' in key:
                classifier_key = key
                break
    
    if classifier_key:
        num_classes = state_dict[classifier_key].shape[0]
    else:
        raise ValueError("Could not determine number of classes from checkpoint")
    
    # Create model with same configuration
    model_name = config.model.name
    variant = config.experiment.variant
    
    # Setup act_layer and norm_layer
    act_layer = None
    norm_layer = None
    
    if variant in ['P1', 'P2', 'P4', 'P5', 'A1', 'A2', 'A3']:
        per_channel = variant in ['P2', 'P5', 'A3']
        kappa_init = config.model.iha.kappa_init
        if variant == 'A1':
            kappa_init = 0.0
        act_layer = iha_act_layer(kappa_init=kappa_init, per_channel=per_channel)
    
    if variant in ['P3', 'P4', 'P5', 'A1', 'A2', 'A3']:
        momentum = config.model.bmcn.momentum
        affine = True
        if variant == 'A2':
            momentum = 0.0
        elif variant == 'A3':
            affine = False
        
        is_vit = 'vit' in model_name.lower() or 'swin' in model_name.lower()
        norm_layer = bmcn_norm_layer(
            eps=config.model.bmcn.eps,
            momentum=momentum,
            affine=affine,
            channel_last=is_vit
        )
    
    # Create model
    model = timm.create_model(
        model_name,
        pretrained=False,  # Don't load pretrained weights
        num_classes=num_classes,
        act_layer=act_layer,
        norm_layer=norm_layer
    )
    
    # Load checkpoint weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    return model, config


def create_test_dataloader(config: Config, dataset_name: str = None):
    """Create test dataloader with proper transforms"""
    if dataset_name:
        config.data.dataset_name = dataset_name
    
    # Create a temporary model to get transform config
    temp_model = timm.create_model(config.model.name, pretrained=True)
    data_cfg = timm.data.resolve_data_config(temp_model.pretrained_cfg)
    test_transform = timm.data.create_transform(**data_cfg, is_training=False)
    del temp_model
    
    # Create test dataset
    test_dataset = UFGVCDataset(
        dataset_name=config.data.dataset_name,
        root=config.data.root,
        split='test',
        transform=test_transform,
        download=False
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,  # Use larger batch for evaluation
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    return test_loader, test_dataset.classes


def evaluate_model(model, test_loader, device, num_classes, save_dir=None):
    """Evaluate model and return comprehensive metrics"""
    model.eval()
    
    all_outputs = []
    all_targets = []
    all_features = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get predictions
            outputs = model(inputs)
            
            # Get features (if model supports it)
            try:
                features = model.forward_features(inputs)
                if features.dim() > 2:
                    features = features.mean(dim=list(range(2, features.dim())))
                all_features.append(features.cpu())
            except:
                # Use predictions as features if forward_features not available
                all_features.append(outputs.cpu())
            
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all results
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_features = torch.cat(all_features, dim=0)
    
    # Calculate comprehensive metrics
    metrics_calc = MetricsCalculator(num_classes=num_classes)
    metrics = metrics_calc.calculate_all_metrics(all_outputs, all_targets)
    
    # Calculate efficiency metrics
    efficiency_metrics = metrics_calc.calculate_efficiency_metrics(model)
    metrics.update(efficiency_metrics)
    
    # Save confusion matrix if save_dir provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        metrics_calc.save_confusion_matrix(
            all_outputs, all_targets, 
            save_dir / "confusion_matrix"
        )
    
    return metrics, all_outputs, all_targets, all_features


def compare_models(checkpoint_paths: list, test_datasets: list, device: torch.device):
    """Compare multiple models across multiple datasets"""
    results = {}
    
    for checkpoint_path in checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)
        model_name = checkpoint_path.stem
        
        print(f"\nEvaluating {model_name}...")
        
        # Load model
        model, config = load_model_from_checkpoint(str(checkpoint_path), device)
        
        model_results = {}
        
        for dataset_name in test_datasets:
            print(f"  Testing on {dataset_name}...")
            
            try:
                # Create test dataloader
                test_loader, class_names = create_test_dataloader(config, dataset_name)
                num_classes = len(class_names)
                
                # Evaluate
                metrics, _, _, _ = evaluate_model(model, test_loader, device, num_classes)
                
                model_results[dataset_name] = metrics
                
                print(f"    Top-1 Acc: {metrics['top1_acc']:.2f}%")
                print(f"    Top-5 Acc: {metrics['top5_acc']:.2f}%")
                print(f"    Macro F1: {metrics['macro_f1']:.4f}")
                
            except Exception as e:
                print(f"    Error: {e}")
                model_results[dataset_name] = None
        
        results[model_name] = model_results
    
    return results


def statistical_testing(results: dict, baseline_model: str = None):
    """Perform statistical testing between models"""
    import scipy.stats as stats
    
    if baseline_model is None:
        baseline_model = list(results.keys())[0]
    
    print(f"\nStatistical Testing (baseline: {baseline_model})")
    print("=" * 60)
    
    baseline_results = results[baseline_model]
    datasets = list(baseline_results.keys())
    
    for model_name, model_results in results.items():
        if model_name == baseline_model:
            continue
        
        print(f"\n{model_name} vs {baseline_model}:")
        
        baseline_accs = []
        model_accs = []
        
        for dataset in datasets:
            if (model_results[dataset] is not None and 
                baseline_results[dataset] is not None):
                baseline_accs.append(baseline_results[dataset]['top1_acc'])
                model_accs.append(model_results[dataset]['top1_acc'])
        
        if len(baseline_accs) >= 3:  # Need at least 3 datasets for meaningful test
            t_stat, p_value = stats.ttest_rel(model_accs, baseline_accs)
            mean_diff = np.mean(model_accs) - np.mean(baseline_accs)
            
            print(f"  Mean difference: {mean_diff:+.2f} pp")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained IHA+BMCN models')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--checkpoints', nargs='+', help='Paths to multiple checkpoints for comparison')
    parser.add_argument('--dataset', type=str, help='Dataset to evaluate on')
    parser.add_argument('--datasets', nargs='+', 
                       default=['cotton80', 'soybean', 'soy_ageing_r1', 'soy_ageing_r3', 'soy_ageing_r4', 'soy_ageing_r5', 'soy_ageing_r6'],
                       help='Datasets to evaluate on')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', 
                       help='Directory to save evaluation results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models')
    parser.add_argument('--statistical_test', action='store_true', help='Perform statistical testing')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.compare and args.checkpoints:
        # Compare multiple models
        results = compare_models(args.checkpoints, args.datasets, device)
        
        # Save results
        with open(output_dir / 'comparison_results.yaml', 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        # Statistical testing
        if args.statistical_test:
            statistical_testing(results)
        
        print(f"\nComparison results saved to {output_dir / 'comparison_results.yaml'}")
    
    elif args.checkpoint:
        # Evaluate single model
        print(f"Evaluating checkpoint: {args.checkpoint}")
        
        model, config = load_model_from_checkpoint(args.checkpoint, device)
        
        if args.dataset:
            datasets = [args.dataset]
        else:
            datasets = args.datasets
        
        all_results = {}
        
        for dataset_name in datasets:
            print(f"\nEvaluating on {dataset_name}...")
            
            # Create test dataloader
            test_loader, class_names = create_test_dataloader(config, dataset_name)
            num_classes = len(class_names)
            
            # Create save directory for this dataset
            dataset_save_dir = output_dir / dataset_name
            
            # Evaluate
            metrics, outputs, targets, features = evaluate_model(
                model, test_loader, device, num_classes, dataset_save_dir
            )
            
            all_results[dataset_name] = metrics
            
            print(f"Results for {dataset_name}:")
            print(f"  Top-1 Accuracy: {metrics['top1_acc']:.2f}%")
            print(f"  Top-5 Accuracy: {metrics['top5_acc']:.2f}%")
            print(f"  Macro F1: {metrics['macro_f1']:.4f}")
            print(f"  NMI: {metrics['nmi']:.4f}")
            print(f"  ARI: {metrics['ari']:.4f}")
            print(f"  ECE: {metrics['ece']:.4f}")
            print(f"  Parameters: {metrics['total_params']:,}")
            
            # Save detailed metrics
            with open(dataset_save_dir / 'metrics.yaml', 'w') as f:
                yaml.dump(metrics, f, default_flow_style=False)
        
        # Save overall results
        with open(output_dir / 'evaluation_results.yaml', 'w') as f:
            yaml.dump(all_results, f, default_flow_style=False)
        
        print(f"\nEvaluation results saved to {output_dir}")
    
    else:
        print("ERROR: Provide either --checkpoint or --checkpoints for evaluation")


if __name__ == '__main__':
    main()
