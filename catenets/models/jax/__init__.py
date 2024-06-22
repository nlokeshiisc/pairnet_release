"""
JAX-based implementations for the CATE estimators.
"""
from typing import Any

from catenets.models.jax.disentangled_nets import SNet3
from catenets.models.jax.flextenet import FlexTENet
from catenets.models.jax.offsetnet import OffsetNet
from catenets.models.jax.pseudo_outcome_nets import (
    DRNet,
    PseudoOutcomeNet,
    PWNet,
    RANet,
)
from catenets.models.jax.representation_nets import (
    DragonNet,
    SNet1,
    SNet2,
    TARNet,
    CFRNet,
    PairNet,
)
from catenets.models.jax.vc_net import VCNet, VCNetPairNet
from catenets.models.jax.continuous_baselines import TARNetC, DRNetC, DRNetPairNet
from catenets.models.jax.rnet import RNet
from catenets.models.jax.snet import SNet
from catenets.models.jax.tnet import TNet
from catenets.models.jax.xnet import XNet

SNET1_NAME = "SNet1"
T_NAME = "TNet"
SNET2_NAME = "SNet2"
PSEUDOOUT_NAME = "PseudoOutcomeNet"
SNET3_NAME = "SNet3"
SNET_NAME = "SNet"
XNET_NAME = "XNet"
RNET_NAME = "RNet"
DRNET_NAME = "DRNet"
PWNET_NAME = "PWNet"
RANET_NAME = "RANet"
TARNET_NAME = "TARNet"
CFRNET_NAME = "CFRNet"
PAIRNET_NAME = "PairNet"
FLEXTE_NAME = "FlexTENet"
OFFSET_NAME = "OffsetNet"
DRAGON_NAME = "DragonNet"
VCNET_NAME = "VCNet"
VCNETPAIRNET_NAME = "VCNETPairNet"
TARNETC_NAME = "TARNetC"  # continuous treatment version of TARNet
DRNETC_NAME = "DRNetC"  # continuous treatment version of DRNet
DRNETPAIRNET_NAME = "DRNetPairNet"

ALL_MODELS = [
    T_NAME,
    SNET1_NAME,
    SNET2_NAME,
    SNET3_NAME,
    SNET_NAME,
    PSEUDOOUT_NAME,
    RNET_NAME,
    XNET_NAME,
    DRNET_NAME,
    PWNET_NAME,
    RANET_NAME,
    TARNET_NAME,
    CFRNET_NAME,
    PAIRNET_NAME,
    FLEXTE_NAME,
    OFFSET_NAME,
    VCNET_NAME,
    VCNETPAIRNET_NAME,
    TARNETC_NAME,
    DRNETC_NAME,
    DRNETPAIRNET_NAME,
]
MODEL_DICT = {
    T_NAME: TNet,
    SNET1_NAME: SNet1,
    SNET2_NAME: SNet2,
    SNET3_NAME: SNet3,
    SNET_NAME: SNet,
    PSEUDOOUT_NAME: PseudoOutcomeNet,
    RNET_NAME: RNet,
    XNET_NAME: XNet,
    DRNET_NAME: DRNet,
    PWNET_NAME: PWNet,
    RANET_NAME: RANet,
    TARNET_NAME: TARNet,
    CFRNET_NAME: CFRNet,
    PAIRNET_NAME: PairNet,
    DRAGON_NAME: DragonNet,
    OFFSET_NAME: OffsetNet,
    FLEXTE_NAME: FlexTENet,
    VCNET_NAME: VCNet,
    VCNETPAIRNET_NAME: VCNetPairNet,
    TARNETC_NAME: TARNetC,
    DRNETC_NAME: DRNetC,
    DRNETPAIRNET_NAME: DRNetPairNet,
}

__all__ = [
    T_NAME,
    SNET1_NAME,
    SNET2_NAME,
    SNET3_NAME,
    SNET_NAME,
    PSEUDOOUT_NAME,
    RNET_NAME,
    XNET_NAME,
    DRNET_NAME,
    PWNET_NAME,
    RANET_NAME,
    TARNET_NAME,
    DRAGON_NAME,
    FLEXTE_NAME,
    OFFSET_NAME,
    VCNET_NAME,
    VCNETPAIRNET_NAME,
    TARNETC_NAME,
    DRNETC_NAME,
    DRNETPAIRNET_NAME,
    CFRNET_NAME,
    PAIRNET_NAME,
]


def get_catenet(name: str) -> Any:
    if name not in ALL_MODELS:
        raise ValueError(
            f"Model name should be in catenets.models.jax.ALL_MODELS You passed {name}"
        )
    return MODEL_DICT[name]
