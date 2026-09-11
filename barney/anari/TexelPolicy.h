// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "anari/frontend/type_utility.h"
#include "native/include/barney.h"
#include <mutex>
#include <set>

namespace BARNEY_NS {
  namespace anari {

    /*! how an ANARI element type reaches the device. exactly one
      route applies to any given type; the route decides whether the
      host has to touch the data at all */
    enum class Route {
      /*! handed over in its original type, zero-copy. the device
        widens on read (typedRead / hardware fetch) */
      PRISTINE,
      /*! 3-component data widened to 4 in its own precision, ANARI
        fill (0,0,0,1). textures only - the hardware has no
        3-component texel formats */
      PAD,
      /*! sRGB source splatted to RGBA8 so the decode flag applies to
        the right lanes. textures only */
      SPLAT_SRGB,
      /*! narrowed to float32 of the same width on upload. lossy for
        sources with more than 24 significant bits */
      DEMOTE_F32,
      /*! nothing can carry this type */
      UNSUPPORTED,
    };

    /*! the element kinds barney distinguishes. this is the axis the
      ANARI enum does not expose directly: FLOAT16 and UFIXED16 share
      a base type, so typenameOf() cannot tell them apart */
    enum class Kind {
      FLOAT,    //!< float16/32/64
      UNORM,    //!< ufixed8/16/32/64  - normalized, [0,1]
      SNORM,    //!< fixed8/16/32/64   - normalized, [-1,1]
      INT,      //!< (u)int8/16/32/64  - element reads, no filtering
      OTHER,
    };

    struct Described {
      Kind kind{Kind::OTHER};
      int  bits{0};        //!< bits per component of the source
      int  components{0};  //!< 1..4
      bool srgb{false};
      bool valid{false};
    };

    /*! decompose an ANARI element type into the (kind, bits,
      components) axes the policy is expressed over. the ANARI enum is
      laid out in regular groups of four (scalar, VEC2, VEC3, VEC4),
      but we spell the groups out rather than doing arithmetic on enum
      values so that a renumbering upstream breaks the build instead
      of silently mis-routing texels */
    inline Described describe(ANARIDataType type)
    {
      auto grp = [&](ANARIDataType base, Kind kind, int bits) -> bool {
        if (type < base || type > base + 3)
          return false;
        return true;
      };
      Described d;
      struct Group { ANARIDataType base; Kind kind; int bits; };
      static const Group groups[] = {
        { ANARI_INT8,     Kind::INT,   8  },
        { ANARI_UINT8,    Kind::INT,   8  },
        { ANARI_INT16,    Kind::INT,   16 },
        { ANARI_UINT16,   Kind::INT,   16 },
        { ANARI_INT32,    Kind::INT,   32 },
        { ANARI_UINT32,   Kind::INT,   32 },
        { ANARI_INT64,    Kind::INT,   64 },
        { ANARI_UINT64,   Kind::INT,   64 },
        { ANARI_FIXED8,   Kind::SNORM, 8  },
        { ANARI_UFIXED8,  Kind::UNORM, 8  },
        { ANARI_FIXED16,  Kind::SNORM, 16 },
        { ANARI_UFIXED16, Kind::UNORM, 16 },
        { ANARI_FIXED32,  Kind::SNORM, 32 },
        { ANARI_UFIXED32, Kind::UNORM, 32 },
        { ANARI_FIXED64,  Kind::SNORM, 64 },
        { ANARI_UFIXED64, Kind::UNORM, 64 },
        { ANARI_FLOAT16,  Kind::FLOAT, 16 },
        { ANARI_FLOAT32,  Kind::FLOAT, 32 },
        { ANARI_FLOAT64,  Kind::FLOAT, 64 },
      };
      for (const auto &g : groups) {
        if (grp(g.base, g.kind, g.bits)) {
          d.kind       = g.kind;
          d.bits       = g.bits;
          d.components = int(type - g.base) + 1;
          d.valid      = true;
          return d;
        }
      }
      // the sRGB types are their own little group, and they are not
      // laid out by component count: R, RA, RGB, RGBA
      switch (type) {
      case ANARI_UFIXED8_R_SRGB:    d = {Kind::UNORM,8,1,true,true}; return d;
      case ANARI_UFIXED8_RA_SRGB:   d = {Kind::UNORM,8,2,true,true}; return d;
      case ANARI_UFIXED8_RGB_SRGB:  d = {Kind::UNORM,8,3,true,true}; return d;
      case ANARI_UFIXED8_RGBA_SRGB: d = {Kind::UNORM,8,4,true,true}; return d;
      default: break;
      }
      return d;
    }

    /*! whether rtcore can build a *texture* of this type. buffers and
      textures do not have the same capabilities: an attribute buffer
      is raw bytes plus a typedRead, while a texture needs a channel
      descriptor that a backend actually implements. types missing here
      fall back to DEMOTE_F32 instead of reaching a backend that would
      abort on them */
    inline bool hasTextureFormat(BNDataType type)
    {
      switch ((int)type) {
      case BN_FLOAT32:
      case BN_FLOAT32_VEC4:
      case BN_UFIXED8:
      case BN_UFIXED8_VEC2:
      case BN_UFIXED8_RGBA:
      case BN_UFIXED8_RGBA_SRGB:
      case BN_UFIXED16:
      case BN_UFIXED16_VEC2:
      case BN_FLOAT16:
      case BN_FLOAT16_VEC2:
      case BN_FLOAT16_VEC4:
        return true;
      default:
        // notably BN_FLOAT32_VEC2 (rtc::FLOAT2) and BN_FLOAT16_VEC3
        // (rtc::HALF3, cpu-only): in the enum, not in the backends
        return false;
      }
    }

    struct Routed {
      BNDataType target{BN_DATA_UNDEFINED};
      Route      route{Route::UNSUPPORTED};
      /*! sRGB is decoded when the texel is sampled, so the colour
        space travels with the texture rather than with the data */
      BNTextureColorSpace colorSpace{BN_COLOR_SPACE_LINEAR};
      /*! set when the route is forced by a missing backend format
        rather than chosen by policy - the caller reports this once so
        a gap in rtcore cannot hide as a deliberate decision */
      bool       forcedByCapability{false};
    };

    /*! the barney texel/element type for a natively-carried
      (kind,bits,components) combination, or BN_DATA_UNDEFINED if we
      have no such type. this is the single place that knows what the
      backends actually implement */
    inline BNDataType nativeTypeFor(Kind kind, int bits, int components,
                                    bool srgb)
    {
      if (components < 1 || components > 4)
        return BN_DATA_UNDEFINED;
      if (srgb) {
        // the decode flag applies to the leading (colour) lanes only,
        // which is exactly right for R and RGBA. RA and RGB need a
        // splat - see routeTexel
        if (kind != Kind::UNORM || bits != 8)
          return BN_DATA_UNDEFINED;
        if (components == 1) return BN_UFIXED8;
        if (components == 4) return BN_UFIXED8_RGBA_SRGB;
        return BN_DATA_UNDEFINED;
      }

      switch (kind) {
      case Kind::FLOAT:
        if (bits == 16) {
          static const BNDataType t[5] = { BN_DATA_UNDEFINED,
            BN_FLOAT16, BN_FLOAT16_VEC2, BN_FLOAT16_VEC3, BN_FLOAT16_VEC4 };
          return t[components];
        }
        if (bits == 32) {
          static const BNDataType t[5] = { BN_DATA_UNDEFINED,
            BN_FLOAT32, BN_FLOAT32_VEC2, BN_FLOAT32_VEC3, BN_FLOAT32_VEC4 };
          return t[components];
        }
        return BN_DATA_UNDEFINED; // float64: demoted
      case Kind::UNORM:
        if (bits == 8) {
          static const BNDataType t[5] = { BN_DATA_UNDEFINED,
            BN_UFIXED8, BN_UFIXED8_VEC2, BN_DATA_UNDEFINED, BN_UFIXED8_RGBA };
          return t[components];
        }
        if (bits == 16) {
          static const BNDataType t[5] = { BN_DATA_UNDEFINED,
            BN_UFIXED16, BN_UFIXED16_VEC2, BN_DATA_UNDEFINED,
            BN_DATA_UNDEFINED /* no USHORT4 in rtcore yet */ };
          return t[components];
        }
        return BN_DATA_UNDEFINED; // ufixed32/64: demoted
      case Kind::SNORM:
        // 8/16-bit snorm could be carried natively - CUDA decodes them
        // exactly as ANARI's toFloat4 does (divisor 127/32767, clamped
        // at -1) - but rtcore has no signed texel types yet, so for now
        // every snorm width is demoted and flagged forcedByCapability
        return BN_DATA_UNDEFINED;
      case Kind::INT:
      case Kind::OTHER:
      default:
        return BN_DATA_UNDEFINED;
      }
    }

    /*! route an ANARI element type for the TEXTURE path (image
      samplers). textures are the constrained case: the hardware
      fetches 1/2/4 components only, so 3-component sources are padded
      in their own precision */
    inline Routed routeTexel(ANARIDataType type)
    {
      const Described d = describe(type);
      Routed r;
      if (!d.valid)
        return r;

      // sRGB is a decode flag, not a conversion. the hardware decodes
      // the leading colour lanes and leaves alpha linear, which is
      // already what ANARI wants for R (r,0,0,1) and RGBA
      // (srgb(rgb),a). RA must become (srgb(r),0,0,linear(a)) and RGB
      // has no 3-component texel format, so both are splatted to RGBA8
      if (d.srgb) {
        r.colorSpace = BN_COLOR_SPACE_SRGB;
        BNDataType native = nativeTypeFor(d.kind, d.bits, d.components, true);
        if (native != BN_DATA_UNDEFINED) {
          r.target = native;
          r.route  = Route::PRISTINE;
          return r;
        }
        r.target = BN_UFIXED8_RGBA_SRGB;
        r.route  = Route::SPLAT_SRGB;
        return r;
      }

      // 3-component: no such texel format exists on any hardware, so
      // widen in the source's own precision when we have a 4-wide
      // native type for it
      if (d.components == 3) {
        BNDataType padded = nativeTypeFor(d.kind, d.bits, 4, false);
        if (padded != BN_DATA_UNDEFINED && hasTextureFormat(padded)) {
          r.target = padded;
          r.route  = Route::PAD;
          return r;
        }
      } else {
        BNDataType native = nativeTypeFor(d.kind, d.bits, d.components, false);
        if (native != BN_DATA_UNDEFINED && hasTextureFormat(native)) {
          r.target = native;
          r.route  = Route::PRISTINE;
          return r;
        }
      }

      // everything else is narrowed to float32 on upload. only
      // BN_FLOAT32 and BN_FLOAT32_VEC4 exist as texture formats
      // (rtc::FLOAT2 is declared but unimplemented, and no hardware
      // has a 3-component texel), so anything wider than a scalar
      // lands on vec4
      r.target = (d.components == 1) ? BN_FLOAT32 : BN_FLOAT32_VEC4;
      r.route  = Route::DEMOTE_F32;
      // a demote that only happens because rtcore lacks the format
      // (rather than because the source is float64 or 32/64-bit fixed,
      // which we would narrow anyway) is a capability gap, not a
      // policy decision - the caller reports those so a hole in rtcore
      // cannot pass as a deliberate route
      const bool narrowedByPolicy =
        (d.kind == Kind::FLOAT && d.bits == 64) ||
        (d.kind == Kind::INT) ||
        ((d.kind == Kind::SNORM || d.kind == Kind::UNORM) && d.bits >= 32);
      r.forcedByCapability = !narrowedByPolicy;
      return r;
    }

    /*! route an ANARI element type for the ARRAY path (vertex
      attributes, primitive sampler arrays). same policy as textures
      minus the component rule: arrays have no fixed-width fetch, so
      3-component data stays 3-component */
    inline Routed routeArray(ANARIDataType type)
    {
      const Described d = describe(type);
      Routed r;
      if (!d.valid)
        return r;

      // off the texture path there is no sampler to carry a colour
      // space, and typedRead has no sRGB cases, so sRGB arrays are
      // decoded on the host by the demote (ANARITypeProperties'
      // toFloat4 applies anari_from_srgb for us).
      // RA is the exception to same-width: its toFloat4 is
      // (srgb(r),0,0,linear(a)), not (r,a,0,1). a vec2 demote plus
      // typedRead's vec2 fill would put alpha in G.
      if (d.srgb) {
        static const BNDataType f32[5] = { BN_DATA_UNDEFINED,
          BN_FLOAT32, BN_FLOAT32_VEC2, BN_FLOAT32_VEC3, BN_FLOAT32_VEC4 };
        r.target = (d.components == 2) ? BN_FLOAT32_VEC4 : f32[d.components];
        r.route  = Route::DEMOTE_F32;
        return r;
      }

      BNDataType native = nativeTypeFor(d.kind, d.bits, d.components, false);
      if (native != BN_DATA_UNDEFINED) {
        r.target = native;
        r.route  = Route::PRISTINE;
        return r;
      }

      static const BNDataType f32[5] = { BN_DATA_UNDEFINED,
        BN_FLOAT32, BN_FLOAT32_VEC2, BN_FLOAT32_VEC3, BN_FLOAT32_VEC4 };
      r.target = f32[d.components];
      r.route  = Route::DEMOTE_F32;
      // same rule as routeTexel: only a demote we would not have done
      // on precision grounds counts as a capability gap
      const bool narrowedByPolicy =
        (d.kind == Kind::FLOAT && d.bits == 64) ||
        (d.kind == Kind::INT) ||
        ((d.kind == Kind::SNORM || d.kind == Kind::UNORM) && d.bits >= 32);
      r.forcedByCapability = !narrowedByPolicy;
      return r;
    }

    /*! true the first time a given type is seen, so a caller can
      report a capability gap once instead of once per upload. the
      registry is shared by every device in the process, hence the
      lock: ANARI lets applications drive several devices from
      different threads */
    inline bool firstTimeSeen(ANARIDataType type)
    {
      static std::mutex mutex;
      static std::set<ANARIDataType> seen;
      std::lock_guard<std::mutex> lock(mutex);
      return seen.insert(type).second;
    }

  }
}
