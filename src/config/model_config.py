from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class LayerConfig:
    """Configuration for a single layer."""
    layer_type: str
    params: Dict[str, Any]

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    epochs: int = 100
    batch_size: int = 32
    patience: int = 30
    verbose: bool = True
    validation_split: float = 0.0

@dataclass
class ModelConfig:
    """Complete model configuration."""
    layers: List[LayerConfig]
    loss_function: str
    loss_params: Dict[str, Any]
    training_config: TrainingConfig

    @classmethod
    def create_default_spiral_config(cls):
        """Create a default configuration for spiral dataset."""
        return cls(
            layers=[
                LayerConfig("Dense", {"n_inputs": 2, "n_neurons": 128, "learning_rate": 0.002, "optimizer": "adam"}),
                LayerConfig("BatchNormalization", {}),
                LayerConfig("ReLU", {}),
                LayerConfig("Dropout", {"rate": 0.1}),
                
                LayerConfig("Dense", {"n_inputs": 128, "n_neurons": 64, "learning_rate": 0.002, "optimizer": "adam"}),
                LayerConfig("BatchNormalization", {}),
                LayerConfig("ReLU", {}),
                LayerConfig("Dropout", {"rate": 0.1}),
                
                LayerConfig("Dense", {"n_inputs": 64, "n_neurons": 32, "learning_rate": 0.002, "optimizer": "adam"}),
                LayerConfig("BatchNormalization", {}),
                LayerConfig("ReLU", {}),
                LayerConfig("Dropout", {"rate": 0.1}),
                
                LayerConfig("Dense", {"n_inputs": 32, "n_neurons": 3, "learning_rate": 0.002, "optimizer": "adam"}),
                LayerConfig("Softmax", {}),
            ],
            loss_function="CategoricalCrossentropy",
            loss_params={"regularization_l2": 0.0001},
            training_config=TrainingConfig(epochs=500, batch_size=32, patience=30)
        )