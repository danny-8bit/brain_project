import torch.nn as nn
from monai.networks.nets import SegResNet, SwinUNETR

try:
    from monai.networks.nets import SegResNetDS
    HAS_SEGRESNETDS = True
except ImportError:
    HAS_SEGRESNETDS = False


def build_model(name, in_channels=4, out_channels=3, roi_size=(128, 128, 128), **kwargs):
    name = name.lower()

    if name == "segresnet":
        # 优先使用 SegResNetDS (带 deep supervision)
        if HAS_SEGRESNETDS:
            return SegResNetDS(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                init_filters=kwargs.get("init_filters", 32),
                blocks_down=(1, 2, 2, 4, 4),
                dsdepth=4,
            )
        else:
            return SegResNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                init_filters=kwargs.get("init_filters", 32),
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
                dropout_prob=kwargs.get("dropout", 0.2),
            )

    elif name == "swinunetr":
        try:
            return SwinUNETR(
                img_size=roi_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=kwargs.get("feature_size", 48),
                use_checkpoint=kwargs.get("use_checkpoint", True),
                spatial_dims=3,
            )
        except TypeError:
            return SwinUNETR(
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=kwargs.get("feature_size", 48),
                use_checkpoint=kwargs.get("use_checkpoint", True),
                spatial_dims=3,
            )

    elif name == "mednext":
        # 需要安装 mednext：pip install mednextv1
        # 或克隆 https://github.com/MIC-DKFZ/MedNeXt
        try:
            from nnunet_mednext import create_mednext_v1
        except ImportError:
            raise ImportError(
                "MedNeXt not installed. Install from "
                "https://github.com/MIC-DKFZ/MedNeXt"
            )
        return create_mednext_v1(
            num_input_channels=in_channels,
            num_classes=out_channels,
            model_id="B",       # S / B / M / L
            kernel_size=3,
            deep_supervision=True,
        )

    else:
        raise ValueError(f"Unknown model: {name}")