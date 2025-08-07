import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    name: str = "swin_tiny_patch4_window7_224"
    pretrained: bool = True
    
    @dataclass
    class IHAConfig:
        kappa_init: float = 0.1
        per_channel: bool = False
        eps: float = 1e-5
    
    @dataclass
    class BMCNConfig:
        eps: float = 1e-5
        momentum: float = 0.1
        affine: bool = True
    
    iha: IHAConfig = field(default_factory=IHAConfig)
    bmcn: BMCNConfig = field(default_factory=BMCNConfig)


@dataclass
class DataConfig:
    dataset_name: str = "cotton80"
    root: str = "./data"
    num_workers: int = 4
    
    @dataclass
    class AugmentationConfig:
        auto_augment: str = "rand-m9-mstd0.5-inc1"
        re_prob: float = 0.25
        color_jitter: float = 0.4
    
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    grad_clip: float = 5.0
    seed: int = 42
    output_dir: str = "./outputs"


@dataclass
class ExperimentConfig:
    variant: str = "B0"  # B0, B1, P1, P2, P3, P4, P5, A1, A2, A3
    project_name: str = "IHA-BMCN-UFGVC"
    tags: list = field(default_factory=list)
    notes: str = ""


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load config from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary"""
        # Handle nested dataclasses
        model_data = data.get('model', {})
        if 'iha' in model_data:
            model_data['iha'] = ModelConfig.IHAConfig(**model_data['iha'])
        if 'bmcn' in model_data:
            model_data['bmcn'] = ModelConfig.BMCNConfig(**model_data['bmcn'])
        
        data_config = data.get('data', {})
        if 'augmentation' in data_config:
            data_config['augmentation'] = DataConfig.AugmentationConfig(**data_config['augmentation'])
        
        return cls(
            model=ModelConfig(**model_data),
            data=DataConfig(**data_config),
            training=TrainingConfig(**data.get('training', {})),
            experiment=ExperimentConfig(**data.get('experiment', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    def save_yaml(self, path: str):
        """Save config to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def update_for_variant(self, variant: str):
        """Update config based on experiment variant"""
        self.experiment.variant = variant
        
        # Update model configuration based on variant
        if variant == "B0":
            # Default Swin-T with GELU + LayerNorm
            self.model.name = "swin_tiny_patch4_window7_224"
        elif variant == "B1":
            # Default ResNet-50 with ReLU + BatchNorm
            self.model.name = "resnet50"
        elif variant in ["P1", "P2", "P4", "P5", "A1", "A2", "A3"]:
            # Use IHA activation
            if variant == "P2" or variant == "P5" or variant == "A3":
                self.model.iha.per_channel = True
            if variant == "A1":
                self.model.iha.kappa_init = 0.0
        
        # Add variant-specific tags
        variant_descriptions = {
            "B0": "Default Swin-T (GELU + LayerNorm)",
            "B1": "Default ResNet-50 (ReLU + BatchNorm)",
            "P1": "IHA (shared κ) + BatchNorm",
            "P2": "IHA (per-channel κ) + BatchNorm",
            "P3": "GELU + BMCN",
            "P4": "IHA (shared κ) + BMCN",
            "P5": "IHA (per-channel κ) + BMCN",
            "A1": "IHA κ=0.0 + BMCN (sanity check)",
            "A2": "IHA + BMCN (momentum=0.0)",
            "A3": "IHA (per-channel) + BMCN (affine=False)"
        }
        
        self.experiment.notes = variant_descriptions.get(variant, "")
        self.experiment.tags = [variant, self.data.dataset_name]


def create_default_configs() -> Dict[str, Config]:
    """Create default configurations for all experiment variants"""
    configs = {}
    variants = ["B0", "B1", "P1", "P2", "P3", "P4", "P5", "A1", "A2", "A3"]
    
    for variant in variants:
        config = Config()
        config.update_for_variant(variant)
        configs[variant] = config
    
    return configs


def create_sweep_configs(base_config: Config, sweep_params: Dict[str, list]) -> list:
    """Create configs for hyperparameter sweep"""
    import itertools
    
    # Get all combinations of sweep parameters
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    
    configs = []
    for values in itertools.product(*param_values):
        config = Config.from_dict(base_config.to_dict())  # Deep copy
        
        # Update sweep parameters
        for name, value in zip(param_names, values):
            if name == "kappa_init":
                config.model.iha.kappa_init = value
            elif name == "momentum":
                config.model.bmcn.momentum = value
            elif name == "batch_size":
                config.training.batch_size = value
            # Add more parameters as needed
        
        configs.append(config)
    
    return configs


# Example usage and default config creation
if __name__ == "__main__":
    # Create and save default configs
    configs = create_default_configs()
    
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    for variant, config in configs.items():
        config.save_yaml(config_dir / f"{variant}.yaml")
    
    print(f"Created {len(configs)} default configurations in {config_dir}")
    
    # Create sweep configuration
    base_config = configs["P5"]  # Use P5 as base for sweep
    sweep_params = {
        "kappa_init": [0.05, 0.1, 0.2, 0.5],
        "momentum": [0.05, 0.1, 0.2],
        "batch_size": [4, 8, 16]
    }
    
    sweep_configs = create_sweep_configs(base_config, sweep_params)
    print(f"Created {len(sweep_configs)} sweep configurations")
