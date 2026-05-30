import torch.nn as nn
from monai.networks.nets import (
    SegResNet,
    SwinUNETR,
    UNet,
    VNet,
    AttentionUnet,
    UNETR,
)

from .baselines import UNetPlusPlus3D, DeepMedic3D, TransBTS3D

try:
    from monai.networks.nets import SegResNetDS
    HAS_SEGRESNETDS = True
except ImportError:
    HAS_SEGRESNETDS = False


def build_model(name, in_channels=4, out_channels=3, roi_size=(128, 128, 128), **kwargs):
    name = name.lower()
    init_filters = kwargs.get("init_filters", 32)
    dropout = kwargs.get("dropout", 0.0)

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
                dropout_prob=dropout,
            )

    elif name in ("unet", "unet3d"):
        channels = [init_filters, init_filters * 2, init_filters * 4, init_filters * 8, init_filters * 16]
        strides = [2, 2, 2, 2]
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=0,
            dropout=dropout,
        )

    elif name == "resunet":
        channels = [init_filters, init_filters * 2, init_filters * 4, init_filters * 8, init_filters * 16]
        strides = [2, 2, 2, 2]
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=kwargs.get("num_res_units", 2),
            dropout=dropout,
        )

    elif name == "vnet":
        dropout_prob_down = kwargs.get("dropout_prob_down", dropout)
        dropout_prob_up = kwargs.get("dropout_prob_up", (dropout, dropout))
        if isinstance(dropout_prob_up, (int, float)):
            dropout_prob_up = (float(dropout_prob_up), float(dropout_prob_up))
        return VNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob_down=dropout_prob_down,
            dropout_prob_up=dropout_prob_up,
        )

    elif name in ("attention_unet", "attunet"):
        channels = [init_filters, init_filters * 2, init_filters * 4, init_filters * 8, init_filters * 16]
        strides = [2, 2, 2, 2]
        try:
            return AttentionUnet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                strides=strides,
                dropout=dropout,
            )
        except TypeError:
            return AttentionUnet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                strides=strides,
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

    elif name == "unetr":
        return UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=roi_size,
            feature_size=kwargs.get("feature_size", 16),
            hidden_size=kwargs.get("hidden_size", 768),
            mlp_dim=kwargs.get("mlp_dim", 3072),
            num_heads=kwargs.get("num_heads", 12),
            pos_embed=kwargs.get("pos_embed", "perceptron"),
            dropout_rate=dropout,
            spatial_dims=3,
        )

    elif name in ("unetpp", "unetplusplus"):
        return UNetPlusPlus3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=init_filters,
        )

    elif name == "deepmedic":
        return DeepMedic3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=kwargs.get("deepmedic_filters", 24),
        )

    elif name == "transbts":
        return TransBTS3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=init_filters,
            num_heads=kwargs.get("num_heads", 4),
            transformer_layers=kwargs.get("transformer_layers", 2),
            mlp_ratio=kwargs.get("mlp_ratio", 2),
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