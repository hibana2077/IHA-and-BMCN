#!/usr/bin/env python3
"""
Quick test script to verify the implementation works
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm

from src.iha import iha_act_layer
from src.bmcn import bmcn_norm_layer, BMCN
from src.dataset.ufgvc import UFGVCDataset
from utils.config import Config


def test_iha_activation():
    """Test IHA activation layer"""
    print("Testing IHA activation...")
    
    # Test with different configurations
    configs = [
        {"kappa_init": 0.1, "per_channel": False},
        {"kappa_init": 0.1, "per_channel": True},
    ]
    
    for config in configs:
        print(f"  Config: {config}")
        
        # Create IHA layer
        act_layer = iha_act_layer(**config)
        iha = act_layer(inplace=False)
        
        # Test with different input shapes
        shapes = [(2, 64, 32, 32), (2, 128, 16, 16)]
        
        for shape in shapes:
            x = torch.randn(shape)
            y = iha(x)
            
            print(f"    Input shape: {shape}, Output shape: {y.shape}")
            print(f"    Kappa: {iha.kappa.data if hasattr(iha.kappa, 'data') else 'Not initialized'}")
            
            # Test gradient flow
            loss = y.sum()
            loss.backward()
            
            if hasattr(iha.kappa, 'grad') and iha.kappa.grad is not None:
                print(f"    Kappa gradient: {iha.kappa.grad.abs().mean().item():.6f}")
    
    print("IHA test completed!\n")


def test_bmcn_normalization():
    """Test BMCN normalization layer"""
    print("Testing BMCN normalization...")
    
    # Test with different configurations
    configs = [
        {"num_features": 64, "channel_last": False},  # CNN
        {"num_features": 128, "channel_last": True},  # ViT
    ]
    
    for config in configs:
        print(f"  Config: {config}")
        
        norm_layer = BMCN(
            num_features=config["num_features"], 
            channel_last=config["channel_last"]
        )
        
        if config["channel_last"]:
            # ViT format: (N, L, C)
            x = torch.randn(4, 196, config["num_features"])
        else:
            # CNN format: (N, C, H, W)
            x = torch.randn(4, config["num_features"], 8, 8)
        
        # Test training mode
        norm_layer.train()
        y_train = norm_layer(x)
        
        # Test eval mode
        norm_layer.eval()
        y_eval = norm_layer(x)
        
        print(f"    Input shape: {x.shape}")
        print(f"    Output shape (train): {y_train.shape}")
        print(f"    Output shape (eval): {y_eval.shape}")
        print(f"    Running mean: {norm_layer.running_mean[:5]}")
        print(f"    Running var: {norm_layer.running_var[:5]}")
    
    print("BMCN test completed!\n")


def test_model_creation():
    """Test model creation with IHA and BMCN"""
    print("Testing model creation...")
    
    variants = ["B0", "P1", "P3", "P5"]
    
    for variant in variants:
        print(f"  Testing variant {variant}...")
        
        config = Config()
        config.update_for_variant(variant)
        
        # Setup layers based on variant
        act_layer = None
        norm_layer = None
        
        if variant in ['P1', 'P5']:
            per_channel = variant == 'P5'
            act_layer = iha_act_layer(kappa_init=0.1, per_channel=per_channel)
        
        if variant in ['P3', 'P5']:
            norm_layer = bmcn_norm_layer(channel_last=True)  # For Swin
        
        # Create model
        try:
            model = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=False,  # Skip pretrained for speed
                num_classes=10,
                act_layer=act_layer,
                norm_layer=norm_layer
            )
            
            # Test forward pass
            x = torch.randn(2, 3, 224, 224)
            y = model(x)
            
            print(f"    Model created successfully!")
            print(f"    Input: {x.shape}, Output: {y.shape}")
            print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        except Exception as e:
            print(f"    Error: {e}")
    
    print("Model creation test completed!\n")


def test_dataset():
    """Test dataset loading"""
    print("Testing dataset loading...")
    
    try:
        # Create a small test dataset
        dataset = UFGVCDataset(
            dataset_name="cotton80",
            root="./data",
            split="train",
            download=True
        )
        
        print(f"  Dataset: {dataset.dataset_name}")
        print(f"  Split: {dataset.split}")
        print(f"  Samples: {len(dataset)}")
        print(f"  Classes: {len(dataset.classes)}")
        
        # Test sample loading
        sample, label = dataset[0]
        print(f"  Sample shape: {sample.size if hasattr(sample, 'size') else type(sample)}")
        print(f"  Label: {label}")
        
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch_data, batch_labels = next(iter(dataloader))
        
        print(f"  Batch data shape: {batch_data.shape if hasattr(batch_data, 'shape') else type(batch_data)}")
        print(f"  Batch labels shape: {batch_labels.shape if hasattr(batch_labels, 'shape') else type(batch_labels)}")
        
    except Exception as e:
        print(f"  Dataset test failed: {e}")
        print("  This is expected if dataset files are not available")
    
    print("Dataset test completed!\n")


def test_config_system():
    """Test configuration system"""
    print("Testing configuration system...")
    
    # Test config creation
    config = Config()
    print(f"  Default config created")
    print(f"  Dataset: {config.data.dataset_name}")
    print(f"  Model: {config.model.name}")
    print(f"  Variant: {config.experiment.variant}")
    
    # Test variant updates
    variants = ["B0", "P1", "P5", "A1"]
    for variant in variants:
        config.update_for_variant(variant)
        print(f"  Variant {variant}: IHA per_channel={config.model.iha.per_channel}, kappa_init={config.model.iha.kappa_init}")
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config.save_yaml(f.name)
        
        # Load it back
        loaded_config = Config.from_yaml(f.name)
        print(f"  Config saved and loaded successfully")
        print(f"  Loaded dataset: {loaded_config.data.dataset_name}")
    
    print("Config system test completed!\n")


def main():
    """Run all tests"""
    print("Running quick implementation tests...\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Run tests
    test_config_system()
    test_iha_activation()
    test_bmcn_normalization()
    test_model_creation()
    test_dataset()
    
    print("All tests completed!")
    print("\nIf all tests passed, the implementation is ready for training.")
    print("Run with: python train.py --config configs/P5.yaml --debug")


if __name__ == "__main__":
    main()
