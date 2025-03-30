# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SAMURAI: SAM for zero-shot visual tracking with Motion-Aware Memory
"""

__version__ = "0.1.0"

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

if not GlobalHydra.instance().is_initialized():
    initialize_config_module("samurai", version_base="1.2")

# Import necessary components to make them available at package level
# from samurai.sam2.modeling.sam2_base import SAM2Base
# from samurai.sam2.sam2_video_predictor import SAM2VideoPredictor