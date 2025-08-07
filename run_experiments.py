#!/usr/bin/env python3
"""
Simple launcher script for running IHA+BMCN experiments
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_single_experiment(variant: str, dataset: str, gpu: int = 0, debug: bool = False):
    """Run a single experiment"""
    cmd = [
        sys.executable, "train.py",
        "--config", f"configs/{variant}.yaml",
        "--dataset", dataset,
        "--gpu", str(gpu)
    ]
    
    if debug:
        cmd.append("--debug")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ Completed: {variant} on {dataset}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {variant} on {dataset} - {e}")
        return False


def run_all_variants(dataset: str, gpu: int = 0, debug: bool = False):
    """Run all experiment variants on a dataset"""
    variants = ["B0", "B1", "P1", "P2", "P3", "P4", "P5", "A1", "A2", "A3"]
    
    results = {}
    for variant in variants:
        success = run_single_experiment(variant, dataset, gpu, debug)
        results[variant] = success
    
    # Summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    for variant, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{variant:4s}: {status}")
    
    success_count = sum(results.values())
    print(f"\nTotal: {success_count}/{len(variants)} successful")


def run_baselines_vs_proposals(dataset: str, gpu: int = 0, debug: bool = False):
    """Run key comparisons: baselines vs proposals"""
    experiments = ["B0", "B1", "P4", "P5"]  # Key comparisons
    
    for variant in experiments:
        run_single_experiment(variant, dataset, gpu, debug)


def run_ablation_study(dataset: str, gpu: int = 0, debug: bool = False):
    """Run ablation study experiments"""
    ablations = ["P1", "P2", "P3", "A1", "A2", "A3"]
    
    for variant in ablations:
        run_single_experiment(variant, dataset, gpu, debug)


def run_cross_dataset_evaluation(variant: str = "P5", gpu: int = 0, debug: bool = False):
    """Run single variant across all datasets"""
    datasets = [
        "cotton80", "soybean", "soy_ageing_r1", 
        "soy_ageing_r3", "soy_ageing_r4", "soy_ageing_r5", "soy_ageing_r6"
    ]
    
    results = {}
    for dataset in datasets:
        success = run_single_experiment(variant, dataset, gpu, debug)
        results[dataset] = success
    
    # Summary
    print("\n" + "="*50)
    print(f"CROSS-DATASET SUMMARY ({variant})")
    print("="*50)
    for dataset, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{dataset:15s}: {status}")


def main():
    parser = argparse.ArgumentParser(description='Experiment launcher for IHA+BMCN')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--single', action='store_true', 
                           help='Run single experiment')
    mode_group.add_argument('--all-variants', action='store_true',
                           help='Run all variants on one dataset')
    mode_group.add_argument('--baselines-vs-proposals', action='store_true',
                           help='Run key comparisons (B0, B1, P4, P5)')
    mode_group.add_argument('--ablation', action='store_true',
                           help='Run ablation study (P1, P2, P3, A1, A2, A3)')
    mode_group.add_argument('--cross-dataset', action='store_true',
                           help='Run one variant across all datasets')
    
    # Parameters
    parser.add_argument('--variant', type=str, default='P5',
                       choices=['B0', 'B1', 'P1', 'P2', 'P3', 'P4', 'P5', 'A1', 'A2', 'A3'],
                       help='Experiment variant')
    parser.add_argument('--dataset', type=str, default='cotton80',
                       choices=['cotton80', 'soybean', 'soy_ageing_r1', 'soy_ageing_r3', 
                               'soy_ageing_r4', 'soy_ageing_r5', 'soy_ageing_r6'],
                       help='Dataset name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--debug', action='store_true', help='Debug mode (5 epochs)')
    
    args = parser.parse_args()
    
    # Check if config files exist
    if not Path("configs").exists():
        print("❌ Config directory not found. Run: python utils/config.py")
        return
    
    print("🚀 Starting IHA+BMCN experiments...")
    print(f"   GPU: {args.gpu}")
    print(f"   Debug mode: {args.debug}")
    print()
    
    if args.single:
        run_single_experiment(args.variant, args.dataset, args.gpu, args.debug)
    
    elif args.all_variants:
        run_all_variants(args.dataset, args.gpu, args.debug)
    
    elif args.baselines_vs_proposals:
        run_baselines_vs_proposals(args.dataset, args.gpu, args.debug)
    
    elif args.ablation:
        run_ablation_study(args.dataset, args.gpu, args.debug)
    
    elif args.cross_dataset:
        run_cross_dataset_evaluation(args.variant, args.gpu, args.debug)
    
    print("\n🎉 Experiment launcher completed!")


if __name__ == "__main__":
    main()
