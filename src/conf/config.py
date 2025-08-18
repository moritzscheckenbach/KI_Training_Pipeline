from dataclasses import dataclass, field
from typing import List

from hydra.core.config_store import ConfigStore


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    random_seed: int = 42


@dataclass
class SchedulerConfig:
    file: str = "scheduler"
    type: str = "StepLR"
    patience: int = 5
    factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class OptimizerConfig:
    type: str = "Adam"
    weight_decay: float = 0.01

    @dataclass
    class AdamConfig:
        betas: List[float] = field(default_factory=lambda: [0.9, 0.999])

    @dataclass
    class SGDConfig:
        momentum: float = 0.9

    adam: AdamConfig = field(default_factory=AdamConfig)
    sgd: SGDConfig = field(default_factory=SGDConfig)


@dataclass
class AugmentationConfig:
    file: str = "augmentation_1"


@dataclass
class TransferLearningConfig:
    enabled: bool = False
    path: str = ""
    trans_file: str = "transferlearning"

    @dataclass
    class FreezingConfig:
        enabled: bool = False
        strategy: str = "freeze_all_except_head"

        @dataclass
        class FreezeEarlyLayersConfig:
            freeze_until_layer: int = 6

        @dataclass
        class CustomFreezingConfig:
            freeze_layers: List[str] = field(default_factory=lambda: ["backbone.layer1", "backbone.layer2"])
            unfreeze_layers: List[str] = field(default_factory=lambda: ["detection_head", "backbone.layer4"])

        freeze_early_layers: FreezeEarlyLayersConfig = field(default_factory=FreezeEarlyLayersConfig)
        custom: CustomFreezingConfig = field(default_factory=CustomFreezingConfig)

    freezing: FreezingConfig = field(default_factory=FreezingConfig)

    @dataclass
    class LearningRateConfig:
        backbone_lr_multiplier: float = 0.1
        head_lr_multiplier: float = 1.0

    lr: LearningRateConfig = field(default_factory=LearningRateConfig)


@dataclass
class ModelConfig:
    type: str = "object_detection"
    file: str = "ResNet50"
    transfer_learning: TransferLearningConfig = field(default_factory=TransferLearningConfig)


@dataclass
class DatasetConfig:
    root: str = "datasets/object_detection/Type_COCO/Test_Duckiebots/"
    type: str = "Type_COCO"
    num_classes: int = 4


@dataclass
class AIPipelineConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


cs = ConfigStore.instance()
cs.store(group="schema", name="aipipeline", node=AIPipelineConfig)
