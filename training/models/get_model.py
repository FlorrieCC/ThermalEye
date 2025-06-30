from .spatiotemporal_tcn import SpatioTemporalTCN
from .threed_cnn import threeDCNN
from .spatial_temporal_transformer import SpatialTemporalTransformer
from .threed_cnn import threeDCNN

def get_model(name: str):
    name = name.lower()
    if name == "tcn":
        return SpatioTemporalTCN()
    elif name == "3dcnn":
        return threeDCNN()
    elif name == "transformer":
        return SpatialTemporalTransformer()
    elif name == "threed_cnn":
        return threeDCNN()
    else:
        raise ValueError(f"[ERROR] Unknown model name: {name}")