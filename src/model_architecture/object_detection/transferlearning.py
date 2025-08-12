import torch
import torch.nn as nn
import importlib
from typing import Dict, List, Optional

class TransferLearningModel(nn.Module):
    """
    Wrapper für Transfer Learning mit verschiedenen Freeze-Strategien
    """
    
    def __init__(self, config: Dict):
        super(TransferLearningModel, self).__init__()
        
        self.config = config
        self.base_model = self._load_pretrained_model()
        self._apply_transfer_learning_strategy()
        
    def _load_pretrained_model(self):
        """Lädt das vortrainierte Modell"""
        model_path = self.config["transfer_learning"]["pretrained_model_path"]
        model_type = self.config["transfer_learning"]["base_model_type"]
        model_file = self.config["transfer_learning"]["base_model_file"]
        
        # Dynamisch das Modell-Modul laden
        model_module = importlib.import_module(f"model_architecture.{model_type}.{model_file}")
        base_model = model_module.build_model()
        
        # Checkpoint laden
        checkpoint = torch.load(model_path, map_location='cpu')
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Loaded pretrained model from: {model_path}")
        return base_model
    
    def _apply_transfer_learning_strategy(self):
        """Wendet die Transfer Learning Strategie an"""
        strategy = self.config["transfer_learning"]["strategy"]
        
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
        """Friert alles außer dem Detection Head ein"""
        # Alle Parameter einfrieren
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Nur Detection Head unfreezen
        if hasattr(self.base_model, 'detection_head'):
            for param in self.base_model.detection_head.parameters():
                param.requires_grad = True
        
        print("🧊 Froze all layers except detection head")
    
    def _freeze_early_layers(self):
        """Friert die ersten N Layer ein"""
        freeze_until_layer = self.config["transfer_learning"].get("freeze_until_layer", 6)
        
        if hasattr(self.base_model, 'backbone'):
            children = list(self.base_model.backbone.children())
            for i, child in enumerate(children):
                if i < freeze_until_layer:
                    for param in child.parameters():
                        param.requires_grad = False
        
        print(f"🧊 Froze first {freeze_until_layer} layers")
    
    def _freeze_backbone(self):
        """Friert das gesamte Backbone ein"""
        if hasattr(self.base_model, 'backbone'):
            for param in self.base_model.backbone.parameters():
                param.requires_grad = False
        
        print("🧊 Froze backbone")
    
    def _unfreeze_all(self):
        """Alle Parameter trainierbar machen"""
        for param in self.base_model.parameters():
            param.requires_grad = True
            
        print("🔥 Unfroze all parameters")
    
    def _custom_freeze(self):
        """Custom Freeze basierend auf Layer-Namen"""
        freeze_layers = self.config["transfer_learning"].get("freeze_layers", [])
        unfreeze_layers = self.config["transfer_learning"].get("unfreeze_layers", [])
        
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
    
    def get_input_size(self):
        """Return expected input size"""
        return self.base_model.get_input_size()
    
    def get_trainable_parameters(self):
        """Zeigt trainierbare Parameter"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        print(f"📊 Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
        return trainable, total

def build_model(num_classes=20, config=None):
    """Factory function für Transfer Learning Model"""
    if config is None:
        raise ValueError("Config is required for transfer learning")
    
    model = TransferLearningModel(config)
    model.get_trainable_parameters()  # Info ausgeben
    return model