from .spatiotemporal_tcn import SpatioTemporalTCN
from .threed_cnn import threeDCNN
from .spatial_temporal_transformer import SpatialTemporalTransformer
from .resnet18 import ResNet18BlinkModel
from .resnet import ResNetBlink
from .efficientnet import EfficientNetBlinkModel
from .resnetLSTM import ResNetLSTM
from .cnnbigru import CNNBiGRUBlinkModel

def get_model(name: str):
    name = name.lower()
    if name == "tcn":
        return SpatioTemporalTCN()
    elif name == "3dcnn":
        return threeDCNN()
    elif name == "transformer":
        return SpatialTemporalTransformer()
    elif name == "resnet":
        return ResNetBlink()
    elif name == "resnet18":
        return ResNet18BlinkModel()
    elif name == "efficientnet":
        return EfficientNetBlinkModel()
    elif name == "resnetlstm":
        return ResNetLSTM(in_channels=1)
    elif name == "cnnbigru":
        return CNNBiGRUBlinkModel()
    else:
        raise ValueError(f"[ERROR] Unknown model name: {name}")


