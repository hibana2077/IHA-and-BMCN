#!/usr/bin/env python3
"""
Example usage demonstrations for IHA and BMCN layers
"""

import torch
import timm
from src.iha import iha_act_layer
from src.bmcn import bmcn_norm_layer


def example_1_basic_usage():
    """Example 1: Basic usage with different models"""
    print("=" * 50)
    print("Example 1: Basic IHA and BMCN Usage")
    print("=" * 50)
    
    # Example with ResNet (CNN)
    print("\n1. ResNet with IHA + BMCN:")
    model_cnn = timm.create_model(
        'resnet18',
        pretrained=False,
        num_classes=10,
        act_layer=iha_act_layer(kappa_init=0.1, per_channel=True),
        norm_layer=bmcn_norm_layer(eps=1e-5, momentum=0.1, affine=True)
    )
    
    x_cnn = torch.randn(4, 3, 224, 224)
    y_cnn = model_cnn(x_cnn)
    print(f"   Input: {x_cnn.shape} -> Output: {y_cnn.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_cnn.parameters()):,}")
    
    # Example with Vision Transformer
    print("\n2. Swin Transformer with IHA + BMCN:")
    model_vit = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=False,
        num_classes=100,
        act_layer=iha_act_layer(kappa_init=0.1, per_channel=False),
        norm_layer=bmcn_norm_layer(eps=1e-5, momentum=0.1, channel_last=True)
    )
    
    x_vit = torch.randn(2, 3, 224, 224)
    y_vit = model_vit(x_vit)
    print(f"   Input: {x_vit.shape} -> Output: {y_vit.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_vit.parameters()):,}")


def example_2_experiment_variants():
    """Example 2: Different experiment variants"""
    print("\n" + "=" * 50)
    print("Example 2: Experiment Variants")
    print("=" * 50)
    
    variants = {
        "B0": {"act": None, "norm": None, "desc": "Baseline (GELU + LayerNorm)"},
        "P1": {"act": iha_act_layer(kappa_init=0.1), "norm": None, "desc": "IHA only"},
        "P3": {"act": None, "norm": bmcn_norm_layer(channel_last=True), "desc": "BMCN only"},
        "P5": {"act": iha_act_layer(kappa_init=0.1, per_channel=True), 
               "norm": bmcn_norm_layer(channel_last=True), "desc": "IHA + BMCN (full proposal)"}
    }
    
    for variant, config in variants.items():
        print(f"\n{variant}: {config['desc']}")
        
        model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=50,
            act_layer=config["act"],
            norm_layer=config["norm"]
        )
        
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        params = sum(p.numel() for p in model.parameters())
        
        print(f"   Parameters: {params:,}")
        print(f"   Output shape: {y.shape}")


def example_3_small_batch_robustness():
    """Example 3: Demonstrate small batch robustness"""
    print("\n" + "=" * 50)
    print("Example 3: Small Batch Robustness")
    print("=" * 50)
    
    from src.bmcn import BMCN
    import torch.nn as nn
    
    # Compare BMCN vs BatchNorm with very small batches
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # BMCN
        bmcn = BMCN(64, channel_last=False)
        
        # Standard BatchNorm
        bn = nn.BatchNorm2d(64)
        
        x = torch.randn(batch_size, 64, 8, 8)
        
        # Test both
        bmcn.train()
        bn.train()
        
        y_bmcn = bmcn(x)
        y_bn = bn(x)
        
        print(f"   BMCN output std: {y_bmcn.std().item():.4f}")
        print(f"   BatchNorm output std: {y_bn.std().item():.4f}")
        
        # Check if outputs are reasonable (should be close to 1 for normalized data)
        bmcn_ok = 0.5 < y_bmcn.std().item() < 2.0
        bn_ok = 0.5 < y_bn.std().item() < 2.0
        
        print(f"   BMCN stable: {bmcn_ok}, BatchNorm stable: {bn_ok}")


def example_4_kappa_learning():
    """Example 4: Show how IHA kappa parameter learns"""
    print("\n" + "=" * 50)
    print("Example 4: IHA Kappa Learning")
    print("=" * 50)
    
    from src.iha import IHA
    
    # Create IHA layers with different configurations
    iha_shared = IHA(kappa_init=0.1, per_channel=False)
    iha_per_channel = IHA(kappa_init=0.1, per_channel=True)
    
    print("\n1. Shared kappa:")
    x = torch.randn(4, 32, 16, 16, requires_grad=True)
    
    # Forward pass
    y1 = iha_shared(x)
    loss1 = y1.sum()
    loss1.backward()
    
    print(f"   Initial kappa: {iha_shared.kappa.item():.6f}")
    print(f"   Kappa gradient: {iha_shared.kappa.grad.item():.6f}")
    
    print("\n2. Per-channel kappa:")
    x = torch.randn(4, 32, 16, 16, requires_grad=True)
    
    # Forward pass  
    y2 = iha_per_channel(x)
    loss2 = y2.sum()
    loss2.backward()
    
    print(f"   Kappa shape: {iha_per_channel.kappa.shape}")
    print(f"   Kappa range: [{iha_per_channel.kappa.min().item():.6f}, {iha_per_channel.kappa.max().item():.6f}]")
    print(f"   Gradient norm: {iha_per_channel.kappa.grad.norm().item():.6f}")


def example_5_memory_efficiency():
    """Example 5: Compare memory usage"""
    print("\n" + "=" * 50)
    print("Example 5: Memory Efficiency")
    print("=" * 50)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        
        models = {
            "Baseline": timm.create_model('resnet18', pretrained=False, num_classes=1000),
            "IHA+BMCN": timm.create_model(
                'resnet18', 
                pretrained=False, 
                num_classes=1000,
                act_layer=iha_act_layer(kappa_init=0.1),
                norm_layer=bmcn_norm_layer()
            )
        }
        
        for name, model in models.items():
            model = model.to(device)
            model.train()
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            
            # Forward pass
            x = torch.randn(16, 3, 224, 224, device=device)
            y = model(x)
            loss = y.sum()
            loss.backward()
            
            # Get memory usage
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            params = sum(p.numel() for p in model.parameters())
            
            print(f"\n{name}:")
            print(f"   Parameters: {params:,}")
            print(f"   Peak memory: {peak_memory:.1f} MB")
            
            del model, x, y, loss
            torch.cuda.empty_cache()
    else:
        print("CUDA not available - skipping memory comparison")


def main():
    """Run all examples"""
    print("IHA and BMCN Usage Examples")
    print("============================")
    
    try:
        example_1_basic_usage()
        example_2_experiment_variants()
        example_3_small_batch_robustness()
        example_4_kappa_learning()
        example_5_memory_efficiency()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nNext steps:")
        print("1. Run training: python train.py --config configs/P5.yaml --debug")
        print("2. Run experiments: python run_experiments.py --baselines-vs-proposals --dataset cotton80")
        print("3. Evaluate models: python evaluate.py --checkpoint best_model.pth")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
