#pragma once

#include "barney/barneyConfig.h"
#include <cstring>
#include <cassert>
#include <mutex>
#include <vector>
#include <map>
#include <memory>
#include <sstream>
#include "barney/barney.h"


#include "owl/common/math/AffineSpace.h"
#include "owl/common/math/random.h"

namespace rtc {
  
  using namespace owl::common;
    
  using range1f = interval<float>;
  
  namespace device {
    typedef struct _OpaqueAccel   *AccelHandle;
    typedef struct _OpaqueTextureObject *TextureObject;
  };
  
  typedef enum {
    UCHAR,
    USHORT,

    INT,
    INT2,
    INT3,
    INT4,
      
    FLOAT,
    FLOAT2,
    FLOAT3,
    FLOAT4,
      
    UCHAR4,
    NUM_DATA_TYPES
  } DataType;

  typedef enum {
    WRAP,CLAMP,BORDER,MIRROR,
  } AddressMode;
      
  typedef enum {
    FILTER_MODE_POINT,FILTER_MODE_LINEAR,
  } FilterMode;
    
  typedef enum {
    COLOR_SPACE_LINEAR, COLOR_SPACE_SRGB,
  } ColorSpace;
  
  struct TextureDesc {
    FilterMode filterMode = FILTER_MODE_LINEAR;
    AddressMode addressMode[3] = { CLAMP, CLAMP, CLAMP };
    const vec4f borderColor = {0.f,0.f,0.f,0.f};
    bool normalizedCoords = true;
    ColorSpace colorSpace = COLOR_SPACE_LINEAR;
  };

}



