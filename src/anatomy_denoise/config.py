from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class DataConfig:
    root: Path = Path("data")
    train_list: Path = Path("data/splits/train.txt")
    val_list: Path = Path("data/splits/val.txt")
    image_size: Tuple[int, int] = (512, 512)


@dataclass
class Stage1Config:
    epochs: int = 100
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-2
    byol_warmup_epochs: int = 5
    byol_weight: float = 0.1
    seg_weight: float = 1.0
    num_classes: int = 7
    embed_dim: int = 96


@dataclass
class Stage2Config:
    steps: int = 250_000
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-2
    num_classes: int = 7
    timesteps: int = 1000
    kd_start_step: int = 50_000
    anatomy_start_step: int = 150_000
    anatomy_every_n_steps: int = 5
    lambda_res: float = 1.0
    lambda_kd: float = 0.1
    lambda_anatomy: float = 0.05


@dataclass
class TrainConfig:
    seed: int = 1337
    output_dir: Path = Path("outputs")
    data: DataConfig = field(default_factory=DataConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)

