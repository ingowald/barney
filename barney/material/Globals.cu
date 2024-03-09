// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "barney/material/Globals.h"
#include "barney/material/bsdfs/MicrofacetAlbedo.h"

namespace barney {
  namespace render {

    float integrateAverage(const float *f, int size, int numSamples=1024)
    {
      // Trapezoidal rule
      const int n = numSamples+1;
      float sum = 0.f;
      for (int i=0;i<n;i++) {
        const float cosThetaO = (float)i / (n-1);
        const float x = cosThetaO * (size-1);
        sum += interp1DLinear(x, f, size) * cosThetaO * ((i == 0 || i == n-1) ? 0.5f : 1.f);
      }

      float totalSum = sum; //reduce_add(sum);
      return min(2.f * (totalSum / (n-1)), 1.f);
    }
    
    // static void MicrofacetDielectricAlbedoTable_precompute()
    // OWLBuffer MicrofacetDielectricAlbedoTable_precompute(const DevGroup *devGroup)
    // {
    //   const uniform int size = MICROFACET_DIELECTRIC_ALBEDO_TABLE_SIZE;
    //   const uniform float minEta = rcp(MICROFACET_DIELECTRIC_ALBEDO_TABLE_MAX_IOR);
    //   const uniform float maxEta = rcp(MICROFACET_DIELECTRIC_ALBEDO_TABLE_MIN_IOR);

    //   std::vector<float> MicrofacetDielectricAlbedoTable_dir(size*size*size);
    //   std::vector<float> MicrofacetDielectricAlbedoTable_avg(size*size);
    //   std::vector<float> MicrofacetDielectricAlbedoReflectionTable_dir(size*size*size);
    //   std::vector<float> MicrofacetDielectricAlbedoReflectionTable_avg(size*size);

    //   const int numSamples = 1024;
    //   for (int k = 0; k < size; k++) {
    //     const float roughness = (float)k / (size-1.f);
    //     for (int j = 0; j < size; j++) {
    //       const float eta = lerp((float)j / (size-1.f), minEta, maxEta);
    //       for (int i=0;i<size;i++) {
    //         const float cosThetaO = (float)i / (size-1.f);
    //         // dirPtr[i] = func(cosThetaO, eta, roughness, numSamples);
    //         MicrofacetDielectricAlbedoTable_dir[i+size*(j+size*(k))] =
    //           MicrofacetDielectricAlbedo_integrate();
    //         MicrofacetDielectricReflectionAlbedoTable_dir[i+size*(j+size*(k))] =
    //           MicrofacetDielectricReflectionAlbedo_integrate();
    //       }
          
    //       // compute the average albedo
    //       *avgPtr = MicrofacetAlbedoTable_integrateAvg(dirPtr, size);
    //       MicrofacetDielectricAlbedoTable_avg[j+size*(k)] =
    //         integrateAverage(&MicrofacetDielectricAlbedoTable_dir[0+size*(j+size*(k))],
    //                          size);
    //       MicrofacetDielectricAlbedoReflectionTable_avg[j+size*(k)] =
    //         integrateAverage(&MicrofacetDielectricReflectionAlbedoTable_dir[0+size*(j+size*(k))],
    //                          size);
    //     }
    //   }
      
      
    //   // MicrofacetDielectricAlbedoTable_dir = uniform new float[size*size*size];
      // MicrofacetDielectricAlbedoTable_avg = uniform new float[size*size];
      // MicrofacetDielectricAlbedoTable_precompute(&MicrofacetDielectricAlbedo_integrate,
      //                                            size, minEta, maxEta,
      //                                            MicrofacetDielectricAlbedoTable_dir,
      //                                            MicrofacetDielectricAlbedoTable_avg);

      // MicrofacetDielectricReflectionAlbedoTable_dir = uniform new float[size*size*size];
      // MicrofacetDielectricReflectionAlbedoTable_avg = uniform new float[size*size];
      // MicrofacetDielectricAlbedoTable_precompute(&MicrofacetDielectricReflectionAlbedo_integrate,
      //                                            size, minEta, maxEta,
      //                                            MicrofacetDielectricReflectionAlbedoTable_dir,
      //                                            MicrofacetDielectricReflectionAlbedoTable_avg);
      
    //   OWLBuffer MicrofacetDielectricAlbedoTable_dir_buffer
    //     = owlDeviceBufferCreate(devGroup->owl,OWL_FLOAT,size*size*size,
    //                             MicrofacetDielectricAlbedoTable_dir.data());
    //   }
    //   return MicrofacetDielectricAlbedoTable_dir_buffer;
    // }

    // static void MicrofacetAlbedoTable_precompute()
    // {
    //   const uniform int size = MICROFACET_ALBEDO_TABLE_SIZE;

    //   uniform float* uniform dirPtr = MicrofacetAlbedoTable_dir = uniform new float[size*size];
    //   uniform float* uniform avgPtr = MicrofacetAlbedoTable_avg = uniform new float[size];

    //   for (uniform int j = 0; j < size; j++)
    //     {
    //       const float roughness = (float)j / (size-1);
    //       // compute the direction albedo for each cosThetaO
    //       foreach (i = 0 ... size)
    //         {
    //           const float cosThetaO = (float)i / (size-1);
    //           dirPtr[i] = MicrofacetAlbedo_integrate(cosThetaO, roughness);
    //         }

    //       // compute the average albedo
    //       *avgPtr = MicrofacetAlbedoTable_integrateAvg(dirPtr, size);

    //       dirPtr += size;
    //       avgPtr++;
    //     }
    // }

    
    Globals::Globals(const DevGroup *devGroup)
    {
      const int size = MICROFACET_DIELECTRIC_ALBEDO_TABLE_SIZE;
      const float minEta = rcp(MICROFACET_DIELECTRIC_ALBEDO_TABLE_MAX_IOR);
      const float maxEta = rcp(MICROFACET_DIELECTRIC_ALBEDO_TABLE_MIN_IOR);

      std::vector<float> MicrofacetDielectricAlbedoTable_dir(size*size*size);
      std::vector<float> MicrofacetDielectricAlbedoTable_avg(size*size);
      std::vector<float> MicrofacetDielectricReflectionAlbedoTable_dir(size*size*size);
      std::vector<float> MicrofacetDielectricReflectionAlbedoTable_avg(size*size);

      // const int numSamples = 1024;
      for (int k = 0; k < size; k++) {
        const float roughness = (float)k / (size-1.f);
        for (int j = 0; j < size; j++) {
          const float eta = lerp((float)j / (size-1.f), minEta, maxEta);
          for (int i=0;i<size;i++) {
            const float cosThetaO = (float)i / (size-1.f);
            // dirPtr[i] = func(cosThetaO, eta, roughness, numSamples);
            MicrofacetDielectricAlbedoTable_dir[i+size*(j+size*(k))] =
              MicrofacetDielectricAlbedo_integrate(cosThetaO, eta, roughness);
            MicrofacetDielectricReflectionAlbedoTable_dir[i+size*(j+size*(k))] =
              MicrofacetDielectricReflectionAlbedo_integrate(cosThetaO, eta, roughness);
          }
          
          MicrofacetDielectricAlbedoTable_avg[j+size*(k)] =
            integrateAverage(&MicrofacetDielectricAlbedoTable_dir[0+size*(j+size*(k))],
                             size);
          MicrofacetDielectricReflectionAlbedoTable_avg[j+size*(k)] =
            integrateAverage(&MicrofacetDielectricReflectionAlbedoTable_dir[0+size*(j+size*(k))],
                             size);
        }
      }

      
      MicrofacetDielectricAlbedoTable_dir_buffer
        = owlDeviceBufferCreate(devGroup->owl,OWL_FLOAT,
                                MicrofacetDielectricAlbedoTable_dir.size(),
                                MicrofacetDielectricAlbedoTable_dir.data());
      MicrofacetDielectricAlbedoTable_avg_buffer
        = owlDeviceBufferCreate(devGroup->owl,OWL_FLOAT,
                                MicrofacetDielectricAlbedoTable_avg.size(),
                                MicrofacetDielectricAlbedoTable_avg.data());
      
      MicrofacetDielectricReflectionAlbedoTable_dir_buffer
        = owlDeviceBufferCreate(devGroup->owl,OWL_FLOAT,
                                MicrofacetDielectricReflectionAlbedoTable_dir.size(),
                                MicrofacetDielectricReflectionAlbedoTable_dir.data());
      MicrofacetDielectricReflectionAlbedoTable_avg_buffer
        = owlDeviceBufferCreate(devGroup->owl,OWL_FLOAT,
                                MicrofacetDielectricReflectionAlbedoTable_avg.size(),
                                MicrofacetDielectricReflectionAlbedoTable_avg.data());
      
      //                           MicrofacetDielectricAlbedoTable(devGroup);
      // MicrofacetDielectricReflectionAlbedoTable_dir_buffer
      //   = MicrofacetDielectricAlbedoTable_precompute(devGroup);
    }

    Globals::DD Globals::getDD(const Device::SP &device) const
    {
      DD dd;
      dd.MicrofacetDielectricAlbedoTable_dir
        = (float *)owlBufferGetPointer(MicrofacetDielectricAlbedoTable_dir_buffer,
                                       device->owlID);
      dd.MicrofacetDielectricReflectionAlbedoTable_dir
        = (float *)owlBufferGetPointer(MicrofacetDielectricReflectionAlbedoTable_dir_buffer,
                                       device->owlID);
      dd.MicrofacetDielectricAlbedoTable_avg
        = (float *)owlBufferGetPointer(MicrofacetDielectricAlbedoTable_avg_buffer,
                                       device->owlID);
      dd.MicrofacetDielectricReflectionAlbedoTable_avg
        = (float *)owlBufferGetPointer(MicrofacetDielectricReflectionAlbedoTable_avg_buffer,
                                       device->owlID);
      return dd;
    }
    
  }
}
