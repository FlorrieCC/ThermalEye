from .spatiotemporal_tcn import SpatioTemporalTCN
from .threed_cnn import threeDCNN
from .spatial_temporal_transformer import SpatialTemporalTransformer

def get_model(name: str):
    name = name.lower()
    if name == "tcn":
        return SpatioTemporalTCN()
    elif name == "3dcnn":
        return threeDCNN()
    elif name == "transformer":
        return SpatialTemporalTransformer()
    else:
        raise ValueError(f"[ERROR] Unknown model name: {name}")
