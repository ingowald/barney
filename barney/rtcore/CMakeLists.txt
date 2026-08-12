# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# "backend" variable gets set by parent dir, and is one of either
# 'cpu', 'cuda', 'optix', 'hip' or 'hiprt'

include(${backend}.cmake)

