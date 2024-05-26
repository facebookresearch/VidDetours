# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

DEFAULT_VIDEO1_TOKEN = "<video1>"
DEFAULT_VIDEO2_TOKEN = "<video2>"
# DEFAULT_IM_START_TOKEN is the same if model.config.mm_use_im_start_end is set to True
VIDEO1_TOKEN_INDEX = -201
VIDEO2_TOKEN_INDEX = -202