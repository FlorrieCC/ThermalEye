from .spatiotemporal_tcn import SpatioTemporalTCN
from .threed_cnn import threeDCNN

def get_model(name: str):
    name = name.lower()
    if name == "tcn":
        return SpatioTemporalTCN()
    elif name == "3dcnn":
        return threeDCNN()
    else:
        raise ValueError(f"[ERROR] Unknown model name: {name}")
