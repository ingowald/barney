// #include "rtc/common/RTCore.h"

// namespace barney {
//   namespace rtc {
// #if !BARNEY_BACKEND_CUDA
//     Backend *createBackend_cuda()
//     {
//       throw std::runtime_error("#barney: cuda support not compiled in!");
//     }
// #endif
// #if !BARNEY_BACKEND_OPTIX
//     Backend *createBackend_optix()
//     {
//       throw std::runtime_error("#barney: optix support not compiled in!");
//     }
// #endif
// #if !BARNEY_BACKEND_EMBREE
//     Backend *createBackend_embree()
//     {
//       throw std::runtime_error("#barney: embree/cpu support not compiled in!");
//     }
// #endif
//   }

//   int RTCore::getDeviceCount()
//   {
//     return get()->numPhysicalDevices;
//   }
  
//   rtc::Backend *RTCore::create()
//   {
//     std::string fromEnv = getenv("BARNEY_BACKEND");
//     if (fromEnv == "") {
//       // nothing to do, leave defautl selection below
//     } else if (fromEnv == "optix") {
//       return rtc::createBackend_optix();
//     } else if (fromEnv == "embree" || fromEnv == "cpu") {
//       return rtc::createBackend_embree();
//     } else if (fromEnv == "cuda") {
//       return rtc::createBackend_cuda();
//     } else {
//       throw std::runtime_error
//         ("#barney: user requested unknown barney backend '"+fromEnv+"'");
//     }

//     rtc::Backend *be = nullptr;
// #if BARNEY_HAVE_CUDA
//     try {
//       be = rtc::createBackend_optix();
//     } catch (...) {}
//     if (!be) {
//       try {
//         be = rtc::createBackend_optix();
//       } catch (...) {}
//     }
//     if (be && be->numPhysicalDevices == 0) {
//       std::cout << "#barney: no GPUs found; trying CPU callback" << std::endl;
//       delete be;
//       be = nullptr;
//     }
// #endif
//     be = rtc::createBackend_embree();
//     return be;
//   }
  
//   rtc::Backend *RTCore::get()
//   {
//     if (!singleton) singleton = create();
//     return singleton;
//   }
  
//   rtc::Backend *RTCore::singleton = nullptr;
// }
