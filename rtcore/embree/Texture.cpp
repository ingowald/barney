#include "rtcore/embree/Texture.h"

namespace barney {
  namespace embree {

    Texture::Texture(TextureData *const data,
                     const rtc::TextureDesc &desc)
      : rtc::Texture(data,desc)
    {
      BARNEY_NYI();
    }

    TextureData::TextureData(Device *device,
                             vec3i dims,
                             rtc::DataType format,
                             const void *texels)
      : rtc::TextureData(device,dims,format),
        device(device)
    { BARNEY_NYI();}

        
    rtc::Texture *TextureData::createTexture(const rtc::TextureDesc &desc) 
    { return new Texture(this,desc); }

  }
}
