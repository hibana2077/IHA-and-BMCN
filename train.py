import argparse
import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
import wandb
from pathlib import Path
import time
from datetime import datetime

# Import our custom modules
from src.iha import iha_act_layer
from src.bmcn import bmcn_norm_layer
from src.dataset.ufgvc import UFGVCDataset
from utils.metrics import MetricsCalculator
from utils.config import Config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config: Config, num_classes: int):
    """Create model with IHA and BMCN layers"""
    model_name = config.model.name
    
    # Determine act_layer and norm_layer based on experiment variant
    act_layer = None
    norm_layer = None
    
    variant = config.experiment.variant
    
    if variant in ['P1', 'P2', 'P4', 'P5', 'A1', 'A2', 'A3']:
        # Use IHA activation
        per_channel = variant in ['P2', 'P5', 'A3']
        kappa_init = config.model.iha.kappa_init
        if variant == 'A1':
            kappa_init = 0.0  # Degenerate to linear
        act_layer = iha_act_layer(kappa_init=kappa_init, per_channel=per_channel)
    
    if variant in ['P3', 'P4', 'P5', 'A1', 'A2', 'A3']:
        # Use BMCN normalization
        momentum = config.model.bmcn.momentum
        affine = True
        if variant == 'A2':
            momentum = 0.0  # No running stats
        elif variant == 'A3':
            affine = False  # No γ/β
        
        # Determine if it's a ViT (channel_last) or CNN
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
        pretrained=config.model.pretrained,
        num_classes=num_classes,
        act_layer=act_layer,
        norm_layer=norm_layer
    )
    
    return model


def get_transforms(config: Config, model):
    """Get transforms following timm's pretrained model config"""
    # Get pretrained model's data config
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    
    # Create transforms for train/val/test
    train_transform = timm.data.create_transform(
        **data_cfg,
        is_training=True,
        # auto_augment=config.data.augmentation.auto_augment,
        # re_prob=config.data.augmentation.re_prob,
        # color_jitter=config.data.augmentation.color_jitter
    )
    
    val_transform = timm.data.create_transform(
        **data_cfg,
        is_training=False
    )
    
    return train_transform, val_transform


def create_dataloaders(config: Config, train_transform, val_transform):
    """Create data loaders for train/val/test splits"""
    dataset_name = config.data.dataset_name
    root = config.data.root
    
    # Create datasets
    train_dataset = UFGVCDataset(
        dataset_name=dataset_name,
        root=root,
        split='train',
        transform=train_transform,
        download=True
    )
    
    val_dataset = UFGVCDataset(
        dataset_name=dataset_name,
        root=root,
        split='val',
        transform=val_transform,
        download=False
    )
    
    test_dataset = UFGVCDataset(
        dataset_name=dataset_name,
        root=root,
        split='test',
        transform=val_transform,
        download=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, len(train_dataset.classes)


def create_optimizer_scheduler(config: Config, model, steps_per_epoch):
    """Create optimizer and learning rate scheduler"""
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Create scheduler with cosine annealing and warmup
    total_steps = config.training.epochs * steps_per_epoch
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.training.learning_rate,
        total_steps=total_steps,
        pct_start=config.training.warmup_ratio,
        anneal_strategy='cos'
    )
    
    return optimizer, scheduler


def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss = torch.clamp(loss, max=100)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Log every 100 steps
        if batch_idx % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            
            wandb.log({
                'train/step_loss': loss.item(),
                'train/step_acc': 100. * predicted.eq(targets).sum().item() / targets.size(0),
                'train/learning_rate': current_lr,
                'train/step': epoch * len(train_loader) + batch_idx
            })
    
    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device, metrics_calc=None):
    """Validate the model"""
    model.eval()
    total_loss = []
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = torch.clamp(loss, max=100)

            total_loss.append(loss.item())
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all predictions and targets
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    if metrics_calc:
        metrics = metrics_calc.calculate_all_metrics(all_outputs, all_targets)
        return sum(total_loss) / len(total_loss), metrics
    else:
        # Basic accuracy calculation
        _, predicted = all_outputs.max(1)
        acc = 100. * predicted.eq(all_targets).sum().item() / all_targets.size(0)
        return sum(total_loss) / len(total_loss), {'top1_acc': acc}


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, config, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'config': config.to_dict()
    }
    torch.save(checkpoint, filepath)


def main():
    os.environ['WANDB_MODE'] = 'offline'  # Ensure W&B is in offline mode
    parser = argparse.ArgumentParser(description='Train IHA+BMCN on UFGVC datasets')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--dataset', type=str, help='Override dataset name')
    parser.add_argument('--variant', type=str, help='Override experiment variant')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--debug', action='store_true', help='Debug mode with fewer epochs')
    
    args = parser.parse_args()
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Override config if specified
    if args.dataset:
        config.data.dataset_name = args.dataset
    if args.variant:
        config.experiment.variant = args.variant
    if args.debug:
        config.training.epochs = 5
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(config.training.seed)
    
    # Create output directory
    output_dir = Path(config.training.output_dir) / f"{config.data.dataset_name}_{config.experiment.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    
    # Initialize wandb
    wandb.init(
        project=config.experiment.project_name,
        name=f"{config.data.dataset_name}_{config.experiment.variant}",
        config=config.to_dict(),
        dir=str(output_dir)
    )
    
    # Create model (temporary for getting transforms)
    temp_model = timm.create_model(config.model.name, pretrained=True)
    train_transform, val_transform = get_transforms(config, temp_model)
    del temp_model
    
    # Create data loaders
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        config, train_transform, val_transform
    )
    
    print(f"Dataset: {config.data.dataset_name}")
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create actual model with correct number of classes
    model = create_model(config, num_classes).to(device)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_scheduler(config, model, len(train_loader))
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create metrics calculator
    metrics_calc = MetricsCalculator(num_classes=num_classes)
    
    # Training loop
    best_acc = 0.0
    best_epoch = 0
    
    print(f"Starting training for {config.training.epochs} epochs...")
    
    for epoch in range(config.training.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch, config
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, metrics_calc)
        val_acc = val_metrics['top1_acc']
        
        epoch_time = time.time() - start_time
        
        # Log epoch metrics
        log_dict = {
            'epoch': epoch,
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'val/top5_acc': val_metrics.get('top5_acc', 0),
            'time/epoch': epoch_time
        }
        
        # Add additional metrics if available
        for key, value in val_metrics.items():
            if key not in ['top1_acc', 'top5_acc']:
                log_dict[f'val/{key}'] = value
        
        wandb.log(log_dict)
        
        print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Time: {epoch_time:.1f}s")
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch
        
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, config, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_checkpoint_path = output_dir / 'best_checkpoint.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, config, best_checkpoint_path)
    
    # Final test evaluation
    print(f"\nBest validation accuracy: {best_acc:.2f}% at epoch {best_epoch}")
    
    # Load best model for test evaluation
    best_checkpoint = torch.load(output_dir / 'best_checkpoint.pth', weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_loss, test_metrics = validate(model, test_loader, criterion, device, metrics_calc)
    test_acc = test_metrics['top1_acc']
    
    print(f"Final test accuracy: {test_acc:.2f}%")
    
    # Log final test metrics
    final_log = {'test/loss': test_loss, 'test/acc': test_acc}
    for key, value in test_metrics.items():
        if key != 'top1_acc':
            final_log[f'test/{key}'] = value
    
    wandb.log(final_log)
    
    # Save final results
    results = {
        'dataset': config.data.dataset_name,
        'variant': config.experiment.variant,
        'best_val_acc': best_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_metrics': test_metrics
    }
    
    with open(output_dir / 'results.yaml', 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    wandb.finish()
    print(f"Training completed. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
