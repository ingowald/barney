/*
 * Copyright (c) 2019++ NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "barney/material/host/Material.h"

namespace barney {
  
  /*! the 'pbr' values stolen from VisRTX */
  struct VisRTXMaterial : public barney::Material {
    VisRTXMaterial(ModelSlot *owner) : Material(owner) {}
    virtual ~VisRTXMaterial() = default;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override { return "VisRTXMaterial"; }

    void createDD(DD &dd, int deviceID) const override;
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override {};
    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    vec3f baseColor = vec3f(1.f);
    float opacity   { 1.f };
    float metallic  { 1.f };
    float roughness { 1.f };
    float ior       { 1.45f };
  };
  
}
