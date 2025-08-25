import importlib
import os
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import torch
import torch.nn as nn
from hydra import compose, initialize
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


class TransferLearningModel(nn.Module):
    """
    Wrapper für Transfer Learning mit verschiedenen Freeze-Strategien
    """

    def __init__(self, cfg: DictConfig, model_config: DictConfig):
        super(TransferLearningModel, self).__init__()

        self.cfg = cfg
        self.model_config = model_config
        self.base_model = self._load_pretrained_model()
        self._apply_transfer_learning_strategy()

    def _load_pretrained_model(self):
        """Lädt das vortrainierte Modell"""
        model_path = self.cfg.model.transfer_learning.path
        model_type = self.cfg.model.type
        model_file = self.model_config.model.file

        # Dynamisch das Modell-Modul laden
        model_module = importlib.import_module(f"model_architecture.{model_type}.{model_file}")
        base_model = model_module.build_model(num_classes=self.cfg.dataset.num_classes)

        # Checkpoint laden
        base_model.load_state_dict(torch.load(model_path, weights_only=True))

        print(f"✅ Loaded pretrained model from: {model_path}")
        return base_model

    def _apply_transfer_learning_strategy(self):
        """Wendet die Transfer Learning Strategie an"""
       # Wenn freezing explizit deaktiviert ist, benutze 'unfreeze_all' als Strategy
        if self.cfg.model.transfer_learning.freezing.enabled is False:
           strategy = "unfreeze_all"
        else:
           strategy = self.cfg.model.transfer_learning.freezing.strategy

        if strategy == "freeze_all_except_head":
            self._freeze_all_except_head()
        elif strategy == "freeze_early_layers":
            self._freeze_early_layers()
        elif strategy == "freeze_backbone":
            self._freeze_backbone()
        elif strategy == "unfreeze_all":
            self._unfreeze_all()
        elif strategy == "custom_freeze":
            self._custom_freeze()
        else:
            raise ValueError(f"Unknown transfer learning strategy: {strategy}")

    def _freeze_all_except_head(self):
        detection_head = self.cfg.model.transfer_learning.head_name
        """Friert alles außer dem Detection Head ein"""
        # Alle Parameter einfrieren
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Prüfen, ob das Attribut existiert
        if hasattr(self.base_model, detection_head):
            head = getattr(self.base_model, detection_head)  # entspricht self.base_model.roi_head
            for param in head.parameters():
                param.requires_grad = True

    def _freeze_early_layers(self):
        """Friert die ersten N Layer ein"""
        freeze_until_layer = self.cfg.model.transfer_learning.freezing.freeze_early_layers.freeze_until_layer

        # Dynamischen Backbone-Namen aus der Config nutzen, Fallback auf "backbone"
        backbone_name = getattr(self.cfg.model.transfer_learning, "backbone_name", None)
        if backbone_name is None:
            # Falls nicht gesetzt, prüfen ob es unter freeze_backbone konfiguriert ist (Kompatibilität)
            backbone_name = getattr(self.cfg.model.transfer_learning, "backbone_name", "backbone")

        if hasattr(self.base_model, backbone_name):
            backbone = getattr(self.base_model, backbone_name)
            children = list(backbone.children())
            for i, child in enumerate(children):
                if i < freeze_until_layer:
                    for param in child.parameters():
                        param.requires_grad = False
            print(f"🧊 Froze first {freeze_until_layer} layers of backbone '{backbone_name}'")
        else:
            print(f"⚠️ Backbone '{backbone_name}' not found on base_model. No layers frozen.")

    def _freeze_backbone(self):
        """Friert das gesamte Backbone ein"""
        backbone_name = self.cfg.model.transfer_learning.backbone_name
        if hasattr(self.base_model, backbone_name):
            backbone_layer = getattr(self.base_model, backbone_name)
            for param in backbone_layer.parameters():
                param.requires_grad = False

        print(f"🧊 Froze backbone: {backbone_name}")

    def _unfreeze_all(self):
        """Alle Parameter trainierbar machen"""
        for param in self.base_model.parameters():
            param.requires_grad = True

        print("🔥 Unfroze all parameters")

    def _custom_freeze(self):
        """Custom Freeze basierend auf Layer-Namen"""
        # TODO: Custom Freezing anpassen!!!
        freeze_layers = self.cfg.model.transfer_learning.freezing.custom.freeze_layers
        unfreeze_layers = self.cfg.model.transfer_learning.freezing.custom.unfreeze_layers

        # Erst alles einfrieren
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Spezifische Layer unfreezen
        for layer_name in unfreeze_layers:
            if hasattr(self.base_model, layer_name):
                layer = getattr(self.base_model, layer_name)
                for param in layer.parameters():
                    param.requires_grad = True

        print(f"🧊 Custom freeze: unfroze {unfreeze_layers}")

    def forward(self, x, targets=None):
        """Forward pass durch das Base Model"""
        return self.base_model(x, targets)

    def get_trainable_parameters(self):
        """Zeigt trainierbare Parameter"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())

        print(f"📊 Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
        return trainable, total


def build_model_tr(cfg=None):
    """Factory function für Transfer Learning Model"""
    if cfg is None:
        raise ValueError("Config is required for transfer learning")

    model_config_path = Path(cfg.model.transfer_learning.path)
    model_config_folder = model_config_path.parent.parent
    config_file = model_config_folder / "configs/config.yaml"

    # Lade die YAML-Datei direkt
    model_config = OmegaConf.load(config_file)

    model = TransferLearningModel(cfg=cfg, model_config=model_config)
    model.get_trainable_parameters()
    return model


def get_input_size(cfg):

    model_config_path = Path(cfg.model.transfer_learning.path)
    model_config_folder = model_config_path.parent.parent
    config_file = model_config_folder / "configs/config.yaml"

    # Lade die YAML-Datei direkt
    model_config = OmegaConf.load(config_file)

    model_type = model_config.model.type
    model_name = model_config.model.file
    """Return expected input size"""
    model_architecture = importlib.import_module(f"model_architecture.{model_type}.{model_name}")
    inputsize_x, inputsize_y = model_architecture.get_input_size()

    return inputsize_x, inputsize_y
