// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

#include "barney/DeviceContext.h"
#include "barney/fb/FrameBuffer.h"
#include "barney/fb/TiledFB.h"
#include "barney/render/World.h"
#include "barney/GlobalModel.h"

namespace barney {
  namespace render {
  
    // enum { MAX_PATH_DEPTH = 10 };

    inline __device__
    float safe_eps(float f, vec3f v)
    {
      return max(f,1e-5f*reduce_max(abs(v)));
    }

    
    typedef enum {
      RENDER_MODE_UNDEFINED,
      RENDER_MODE_LOCAL,
      RENDER_MODE_AO,
      RENDER_MODE_PT
    } RenderMode;
  
    inline __device__
    vec3f randomDirection(Random &rng)
    {
      vec3f v;
      while (true) {
        v.x = 1.f-2.f*rng();
        v.y = 1.f-2.f*rng();
        v.z = 1.f-2.f*rng();
        if (dot(v,v) <= 1.f)
          return normalize(v);
      }
    }

    struct LightSample {
      /* direction _to_ light */
      vec3f dir;
      /*! radiance coming _from_ dir */
      vec3f L;
      /*! distance to this light sample */
      float dist;
      /*! pdf used for sampling */
      float pdf;
    };
  
    inline __device__ bool sampleAreaLights(LightSample &ls,
                                            const render::World::DD &world,
                                            const vec3f P,
                                            const vec3f N,
                                            Random &random,
                                            bool dbg)
    {
      if (world.numQuadLights == 0) return false;
      static const int RESERVOIR_SIZE = 8;
      int   lID[RESERVOIR_SIZE];
      float u[RESERVOIR_SIZE];
      float v[RESERVOIR_SIZE];
      float weights[RESERVOIR_SIZE];
      float sumWeights = 0.f;
      QuadLight light;
      for (int i=0;i<RESERVOIR_SIZE;i++) {
        lID[i] = min(int(random()*world.numQuadLights),
                     world.numQuadLights-1);
        weights[i] = 0.f;
        light = world.quadLights[lID[i]];
        u[i] = random();
        v[i] = random();
        float lightArea = light.area;
        if (lightArea < 0.f)
          printf("INVALID NEGATIVE LIGHT AREA on light %i/%i : %f\n",
                 lID[i],world.numQuadLights,lightArea);
        vec3f LN = light.normal;
        vec3f LP = light.corner + u[i]*light.edge0 + v[i]*light.edge1;
        vec3f lightDir = LP - P;
        float lightDist = length(lightDir);
        if (lightDist < 1e-3f) continue;
      
        lightDir *= 1.f/lightDist;

        float weight = dot(lightDir,N);
        if (weight <= 1e-3f) continue;
        weight *= -dot(lightDir,LN);
        if (weight <= 1e-3f) continue;
        if (lightArea == 0.f || reduce_max(light.emission) == 0)
          printf("invalid light! %f : %f %f %f\n",
                 lightArea,
                 light.emission.x,
                 light.emission.y,
                 light.emission.z);
        weight *= (1.f/(lightDist*lightDist)) * lightArea * reduce_max(light.emission);
        if (isnan(sumWeights) || weight < 0.f)
          printf("area lights: weight[%i:%i] is nan or negative: dist  %f area %f emission %f %f %f\n",
                 i,lID[i],lightDist,lightArea,
                 light.emission.x,
                 light.emission.y,
                 light.emission.z);
        sumWeights += weight;
        weights[i] = weight;
      }
      if (isnan(sumWeights))
        printf("area lights: sumWeights is nan!\n");
      if (sumWeights == 0.f) return false;
      float r = random()*sumWeights;
      int i=0;
      while (i<RESERVOIR_SIZE && r >= weights[i]) { r-= weights[i]; ++i; }
      if (i == RESERVOIR_SIZE) return false;
    
      light = world.quadLights[lID[i]];
      vec3f LP = light.corner + u[i]*light.edge0 + v[i]*light.edge1;
      vec3f LD = LP-P;
      ls.dir  = normalize(LD);
      ls.dist = length(LD);
      ls.L    = light.emission * (light.area * -dot(light.normal,ls.dir) / (ls.dist*ls.dist));
      ls.pdf
        = weights[i]/sumWeights
        * (float(RESERVOIR_SIZE)/float(world.numQuadLights));
      if (ls.pdf <= 0.f)
        printf("invalid area light PDF %f from i %i weight %f sum %f\n",
               ls.pdf,i,weights[i],sumWeights);
      return true;
    }

    inline __device__ bool sampleDirLights(LightSample &ls,
                                           const World::DD &world,
                                           const vec3f P,
                                           const vec3f N,
                                           Random &random,
                                           bool dbg)
    {
      if (dbg) printf("num dirlights %i\n",world.numDirLights);

      if (world.numDirLights == 0) return false;
      static const int RESERVOIR_SIZE = 8;
      int   lID[RESERVOIR_SIZE];
      float weights[RESERVOIR_SIZE];
      float sumWeights = 0.f;
      DirLight light;
    
      for (int i=0;i<RESERVOIR_SIZE;i++) {
        lID[i] = min(int(random()*world.numDirLights),
                     world.numDirLights-1);
        weights[i] = 0.f;
        light = world.dirLights[lID[i]];
        vec3f lightDir = -light.direction;
        float weight = dot(lightDir,N);
        if (dbg) printf("light #%i, dir %f %f %f weight %f\n",lID[i],lightDir.x,lightDir.y,lightDir.z,weight);
        if (weight <= 1e-3f) continue;
        weight *= reduce_max(light.radiance);
        if (weight <= 1e-3f) continue;
        if (dbg) printf("radiance %f %f %f weight %f\n",light.radiance.x,light.radiance.y,light.radiance.z,weight);
        weights[i] = weight;
        sumWeights += weight;
      }
      if (sumWeights == 0.f) return false;
      float r = random()*sumWeights;
      int i=0;
      while (i<RESERVOIR_SIZE && r >= weights[i]) { r-= weights[i]; ++i; }
      if (i == RESERVOIR_SIZE) return false;
    
      light = world.dirLights[lID[i]];
      ls.dir  = -light.direction;
      ls.dist = 1e10f;//INFINITY;
      ls.L    = light.radiance;
      ls.pdf
        = weights[i]/sumWeights
        * (float(RESERVOIR_SIZE)/float(world.numDirLights));
      return weights[i] != 0.f;
    }

    inline __device__ bool sampleLights(LightSample &ls,
                                        const World::DD &world,
                                        const vec3f P,
                                        const vec3f Ng,
                                        Random &random,
                                        bool dbg)
    {
      LightSample als;
      float alsWeight = (sampleAreaLights(als,world,P,Ng,random,dbg)
                         ? (reduce_max(als.L)/als.pdf)
                         : 0.f);
      LightSample dls;
      float dlsWeight = (sampleDirLights(dls,world,P,Ng,random,dbg)
                         ? (reduce_max(dls.L)/dls.pdf)
                         : 0.f);
      float sumWeights = alsWeight+dlsWeight;
      if (sumWeights == 0.f) return false;
      if (random()*sumWeights < alsWeight) {
        ls = als;
        ls.pdf *= alsWeight/sumWeights;
      } else {
        ls = dls;
        ls.pdf *= dlsWeight/sumWeights;
      }
      if (dbg)
        printf(" light weights %f %f\n",
               alsWeight,dlsWeight);
      if (isnan(ls.pdf) || (ls.pdf <= 0.f)) return false;
      return true;
    }




    inline __device__
    float schlick(float cosine,
                  float ref_idx)
    {
      float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
      r0 = r0 * r0;
      return r0 + (1.0f - r0)*powf((1.0f - cosine), 5.0f);
    }
  
  

    inline __device__
    bool refract(const vec3f& v,
                 const vec3f& n,
                 float ni_over_nt,
                 vec3f &refracted)
    {
      vec3f uv = normalize(v);
      float dt = dot(uv, n);
      float discriminant = 1.0f - ni_over_nt * ni_over_nt*(1 - dt * dt);
      if (discriminant > 0.f) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
        return true;
      }
      else
        return false;
    }
  
    // inline __device__
    // vec3f reflect(const vec3f &v,
    //               const vec3f &n)
    // {
    //   return v - 2.0f*dot(v, n)*n;
    // }
  


    // inline __device__
    // bool scatter_glass(vec3f &scattered_direction,
    //                    Random &random,
    //                    // const vec3f &org,
    //                    const vec3f &dir,
    //                    // const vec3f &P,
    //                    vec3f N,
    //                    const float ior
    //                    // ,
    //                    // PerRayData &prd
    //                    )
    // {
    //   // const vec3f org   = optixGetWorldRayOrigin();
    //   // const vec3f dir   = normalize((vec3f)optixGetWorldRayDirection());

    //   // N = normalize(N);
    //   vec3f outward_normal;
    //   vec3f reflected = reflect(dir,N);
    //   float ni_over_nt;
    //   // prd.out.attenuation = vec3f(1.f, 1.f, 1.f); 
    //   vec3f refracted;
    //   float reflect_prob;
    //   float cosine;
  
    //   if (dot(dir,N) > 0.f) {
    //     outward_normal = -N;
    //     ni_over_nt = ior;
    //     cosine = dot(dir, N);// / vec3f(dir).length();
    //     cosine = sqrtf(1.f - ior*ior*(1.f-cosine*cosine));
    //   }
    //   else {
    //     outward_normal = N;
    //     ni_over_nt = 1.0 / ior;
    //     cosine = -dot(dir, N);// / vec3f(dir).length();
    //   }
    //   if (refract(dir, outward_normal, ni_over_nt, refracted)) 
    //     reflect_prob = schlick(cosine, ior);
    //   else 
    //     reflect_prob = 1.f;

    //   // prd.out.scattered_origin = P;
    //   if (random() < reflect_prob) 
    //     // prd.out.
    //     scattered_direction = reflected;
    //   else 
    //     // prd.out.
    //     scattered_direction = refracted;
  
    //   return true;
    // }


  
    inline __device__
    vec3f radianceFromEnv(const World::DD &world,
                          Ray &ray)
    { auto &env = world.envMapLight;
      if (env.texture) {
        vec3f d = xfmVector(env.transform,normalize(ray.dir));
        float theta = pbrtSphericalTheta(d);
        float phi   = pbrtSphericalPhi(d);
        const float invPi  = 1.f/M_PI;
        const float inv2Pi = 1.f/(2.f*M_PI);
        vec2f uv(phi * inv2Pi, theta * invPi);

        float4 color = tex2D<float4>(env.texture,uv.x,uv.y);
        float envLightPower = 1.f;
        return envLightPower*vec3f(color.x,color.y,color.z);
      } else {
        return world.radiance;
      }
    }

    /*! return dedicated background, if specifeid; otherwise return envmap color */
    inline __device__
    vec3f backgroundOrEnv(const World::DD &world,
                          Ray &ray)
    {
      if (world.envMapLight.texture)
        return radianceFromEnv(world,ray);
      return
        ray.missColor;
    }

    /*! ugh - that should all go into material::AnariPhysical .... */
    template<int MAX_PATH_DEPTH>
    inline __device__
    void bounce(const World::DD &world,
                vec3f &fragment,
                Ray &path,
                Ray &shadowRay,
                int pathDepth)
    {
      const float EPS = 1e-4f;

      const bool  hadNoIntersection  = !path.hadHit();
      const vec3f incomingThroughput = path.throughput;

      if (0 && path.dbg)
        printf("(%i) ------------------------------------------------------------------\n -> incoming %f %f %f dir %f %f %f t %f ismiss %i\n",
               pathDepth,
               path.org.x,
               path.org.y,
               path.org.z,
               (float)path.dir.x,
               (float)path.dir.y,
               (float)path.dir.z,
               path.tMax,int(hadNoIntersection));

      if (path.isShadowRay) {
        // ==================================================================
        // shadow ray = all we have to do is add carried radiance if it
        // reached the light, and discards
        // ==================================================================
                 
        if (hadNoIntersection) {
          // fragment = clamp((vec3f)path.throughput,vec3f(0.f),vec3f(1.f));
          fragment = (vec3f)path.throughput;
          if (0 && path.dbg) printf("shadow miss, frag %f %f %f\n",
                               fragment.x,
                               fragment.y,
                               fragment.z);
        }

        // this path is done.
        shadowRay.tMax = -1.f;
        path.tMax = -1.f;
        return;
      }

      vec3f Ng = path.getN();

      const bool  isVolumeHit        = (Ng == vec3f(0.f));
      if (!isVolumeHit)
        Ng = normalize(Ng);
      const bool  hitWasOnFront      = dot((vec3f)path.dir,Ng) < 0.f;
      if (!hitWasOnFront)
        Ng = - Ng;

      if (hadNoIntersection) {
        // ==================================================================
        // regular ray that did NOT hit ANYTHING 
        // ==================================================================
        if (pathDepth == 0) {
          // ----------------------------------------------------------------
          // PRIMARY ray that didn't hit anything -> background
          // ----------------------------------------------------------------
          if (path.dbg)
            printf("miss primary %f %f %f\n",
                   path.missColor.x,
                   path.missColor.y,
                   path.missColor.z);
          fragment = path.missColor;
          // fragment = path.throughput * backgroundOrEnv(world,path);
          
          // const vec3f fromEnv
          //   = // 1.5f*
          //   backgroundOrEnv(world,path);

          // const vec3f tp = path.throughput;
          // const vec3f addtl = tp
          //   * fromEnv;
          // fragment = addtl;
        } else {
          // ----------------------------------------------------------------
          // SECONDARY ray that didn't hit anything -> env-light
          // ----------------------------------------------------------------
          // this path had at least one bounce, but now bounced into
          // nothingness - compute env-light contribution, and weigh it
          // with the path's carried throughput.
          const vec3f fromEnv = radianceFromEnv(world,path);
          if (0 && path.dbg)
            printf("fromenv %f %f %f\n",
                   fromEnv.x,
                   fromEnv.y,
                   fromEnv.z);
          fragment = path.throughput * fromEnv;
        }
        // no outgoing rays; this path is done.
        path.tMax = -1.f;
        return;
      }
    

      // ==================================================================
      // this ray DID hit something: compute its local frame buffer
      // contribution at this hit point (if any), and generate secondary
      // ray and shadow ray (if applicable), with proper weights.
      // ==================================================================    
      Random &random = (Random &)path.rngSeed;
      const PackedBSDF bsdf = path.getBSDF();
      // bool doTransmission = false;
        // =  ((float)path.mini.transmission > 0.f)
        // && (random() < (float)path.mini.transmission);
      render::DG dg;
      dg.P  = path.P;
      dg.Ng = Ng;
      dg.Ns = Ng;
      dg.wo = -normalize((vec3f)path.dir);
      dg.insideMedium = path.isInMedium;
      // for volumes:
      // if (dg.Ng == vec3f(0.f))
      //   dg.Ng = dg.Ns = -path.dir;
      
      if (0 && path.dbg)
        printf("dg.N %f %f %f\n",
               dg.Ns.x,
               dg.Ns.y,
               dg.Ns.z);

      vec3f frontFacingSurfaceOffset
        = safe_eps(EPS,dg.P)*Ng;
      // vec3f dg_P
      //   = path.P+frontFacingSurfaceOffset;
// if (path.dbg)
      //   printf("(%i) hit trans %f ior %f, dotrans %i\n",
      //          pathDepth,
      //          (float)path.transmission,
      //          (float)path.ior,
      //          int(doTransmission));
// #if 1
      // if (path.dbg) printf("mattype %i\n",path.materialType);




      // ==================================================================
      // FIRST, let us look at generating any shadow rays, if
      // applicable; this way we can later modify the incoming ray in
      // place when we generate the outgoing ray.
      // ==================================================================
      LightSample ls;
      // todo check if BSDF is perfectly specular
      if (sampleLights(ls,world,dg.P,dg.Ng,random,0 && path.dbg)
          // && 
          // (path.materialType != GLASS)
          ) {
        if (0 && path.dbg) printf("eval light %f %f %f\n",
                                         ls.dir.x,
                                         ls.dir.y,
                                         ls.dir.z);
        EvalRes f_r = bsdf.eval(dg,ls.dir,0 && path.dbg);
        if (0 && path.dbg) printf("eval light res %f %f %f: %f\n",
                                         f_r.value.x,
                                         f_r.value.y,
                                         f_r.value.z,
                                         f_r.pdf);
        
        if (!f_r.valid()) {
          shadowRay.tMax = -1.f;
        } else {
          vec3f tp_sr
            = (incomingThroughput)
            //            * (1.f/ls.pdf)
            * f_r.value
            * ls.L
            * (isVolumeHit?1.f:fabsf(dot(dg.Ng,ls.dir)))
            /// f_r.pdf
            ;
          shadowRay.makeShadowRay(/* thrghhpt */tp_sr,
                                  /* surface: */dg.P + 10*frontFacingSurfaceOffset,
                                  /* to light */ls.dir,
                                  /* length   */ls.dist * (1.f-2.f*EPS));
          // if (path.dbg) printf("new shadow ray len %f %f\n",ls.dist,shadowRay.tMax);
          shadowRay.rngSeed = path.rngSeed + 1; random();
          shadowRay.dbg = path.dbg;
          shadowRay.pixelID = path.pixelID;
        }
      }
      
      // ==================================================================
      // now, let's decide what to do with the ray itself
      // ==================================================================
      path.tMax = -1.f;
      if (pathDepth >= MAX_PATH_DEPTH)
        return;
      // if (isVolumeHit) {
      //   // iw - make this disappear; this can/shoudl be handled by a
      //   // proper 'volume matrial' (ie, Phase function)
      //   path.dir = sampleCosineWeightedHemisphere(-vec3f(path.dir),random);
      //   path.throughput = .8f * path.throughput * bsdf.getAlbedo();//baseColor;
      //   return;
      // }
      
      ScatterResult scatterResult;
      // if (path.dbg)
        bsdf.scatter(scatterResult,dg,random,path.dbg);
      if (!scatterResult.valid() || scatterResult.pdf == 0.f)
        return;
      path.org        = dg.P + scatterResult.offsetDirection * frontFacingSurfaceOffset;
      path.dir        = normalize(scatterResult.dir);
      path.throughput
        = path.throughput * scatterResult.f_r
        // * fabsf(dot(dg.Ng,path.dir))
        / (scatterResult.pdf + 1e-10f);
      path.clearHit();
      
      if (0 && path.dbg)
        printf("scatter dir %f %f %f tp %f %f %f\n",
               (float)path.dir.x,
               (float)path.dir.y,
               (float)path.dir.z,
               (float)path.throughput.x,
               (float)path.throughput.y,
               (float)path.throughput.z);
      // } else {
      //   path.dir = sampleCosineWeightedHemisphere(dg.Ns,random);
      //   EvalRes f_r = path.eval(world.globals,dg,path.dir,path.dbg);
        
      //   if (f_r.pdf == 0.f || isinf(f_r.pdf) || isnan(f_r.pdf)) {
      //     path.tMax = -1.f;
      //   } else {
      //     path.throughput = path.throughput * f_r.value
      //       / (f_r.pdf + 1e-10f)
      //       ;
      //   }
      // }
      
// #if 0
//       if ((path.materialType == GLASS)
//           ||
//           (path.materialType == BLENDER)
//           ) {
//         // dg.wo = normalize(neg(path.dir));
//         SampleRes sampleRes;
//         sampleRes.pdf = 0.f;
//         if (path.materialType == GLASS)
//           sampleRes
//             = path.glass.sample(dg,random,path.dbg);
//         else if (path.materialType == BLENDER)
//           sampleRes
//             = path.blender.sample(dg,random,path.dbg);

//         if (sampleRes.pdf < 1e-6f) {
//           path.tMax = -1.f;
//         } else {
//           path.dir = normalize(sampleRes.wi);
//           if (sampleRes.type & BSDF_SPECULAR_TRANSMISSION) {
//             // doTransmission = true;
//             path.isInMedium = !path.isInMedium;
//             path.org = path.P - frontFacingSurfaceOffset;
//           } else {
//             path.org = path.P + frontFacingSurfaceOffset;
//           }
//           path.throughput = path.throughput * sampleRes.weight
//             /* pdf is inf for glass ....  /sampleRes.pdf */
//             /(isinf(sampleRes.pdf) ? 1.f : sampleRes.pdf)
//             ;

//           bool wasLeavingMedium
//             =  (sampleRes.type & BSDF_SPECULAR_TRANSMISSION)
//             && !path.isInMedium;
//           if (wasLeavingMedium) {
//             vec3f attenuation
//               = path.glass.mediumInside.attenuation;
//             attenuation = exp(attenuation*path.tMax);
//             path.throughput = path.throughput * attenuation;
//           }
//           path.tMax = INFINITY;
//         }
//       } else {
//         // ------------------------------------------------------------------
//         // not perfectly specular - do diffuse bounce for now...
//         // ------------------------------------------------------------------

//         // save local path weight for the shadow ray:
//         path.org = path.P + safe_eps(EPS,path.P)*Ng;
//         if (isVolumeHit) {
//           path.dir = sampleCosineWeightedHemisphere(-vec3f(path.dir),random);
//           path.throughput = .8f * path.throughput * path.getAlbedo();//baseColor;
//         } else {
//           path.dir = sampleCosineWeightedHemisphere(dg.Ns,random);
//           EvalRes f_r = path.eval(world.globals,dg,path.dir,path.dbg);

//           if (f_r.pdf == 0.f || isinf(f_r.pdf) || isnan(f_r.pdf)) {
//             path.tMax = -1.f;
//           } else {
//             path.throughput = path.throughput * f_r.value
//               / (f_r.pdf + 1e-10f)
//               ;
//           }
//         }
//       }
//       // ------------------------------------------------------------------
//       // so far we HAVE generated an outgoing path, but it have haev
//       // very low throughput/weak impact on the image - use russian
//       // roulette to either ake it 'stronger', or kill it.
//       // ------------------------------------------------------------------
//       if (((pathDepth < MAX_PATH_DEPTH)
//           ||
//           (pathDepth == 0 && MAX_PATH_DEPTH == 0)
//           ) && path.tMax > 0.f) {
//         path.dir = normalize(path.dir);
//         path.tMax   = INFINITY;
//         path.clearHit();

//         float maxWeight = reduce_max((vec3f)path.throughput);
//         if (maxWeight < .3f) {
//           if (random() < maxWeight) {
//             path.throughput = path.throughput * (1.f/maxWeight);
//           } else {
//             path.tMax = -1.f;
//           }
//         }
//         if (MAX_PATH_DEPTH == 0)
//           path.tMax = 1e-20f;
//       } else
//         path.tMax = -1.f;
// #endif
    }
  

    template<int MAX_PATH_DEPTH>
    __global__
    void g_shadeRays_pt(World::DD world,
                        AccumTile *accumTiles,
                        int accumID,
                        Ray *readQueue,
                        int numRays,
                        Ray *writeQueue,
                        int *d_nextWritePos,
                        int generation)
    {
      int tid = threadIdx.x + blockIdx.x*blockDim.x;
      if (tid >= numRays) return;

      Ray path = readQueue[tid];
      // what we'll add into the frame buffer
      vec3f fragment = 0.f;
      float z = path.tMax;
      // create a (potential) shadow ray, and init to 'invalid'
      Ray shadowRay;
      shadowRay.tMax = -1.f;
      
        // printf("sammpling dir for N %f %f %f\n",dg.N.x,dg.N.y,dg.N.z);

      // bounce that ray on the scene, possibly generating a) a fragment
      // to add to frame buffer; b) a outgoing ray (in-place
      // modification of 'path'); and/or c) a shadow ray
      bounce<MAX_PATH_DEPTH>(world,fragment,
             path,shadowRay,
             generation);
    
      // write shadow and bounce ray(s), if any were generated
      // if (path.dbg)
      //   printf("path.tmax %f shadowray.tmax %f frag %f %f %f\n",
      //          path.tMax,shadowRay.tMax,
      //          fragment.x,fragment.y,fragment.z);
      if (shadowRay.tMax > 0.f) {
        writeQueue[atomicAdd(d_nextWritePos,1)] = shadowRay;
      }
      if (path.tMax > 0.f) {
        writeQueue[atomicAdd(d_nextWritePos,1)] = path;
      }

      // and write the shade fragment, if generated
      int tileID  = path.pixelID / pixelsPerTile;
      int tileOfs = path.pixelID % pixelsPerTile;
      float4 &valueToAccumInto
        = accumTiles[tileID].accum[tileOfs];

      // ==================================================================
      // add to accum buffer. be careful of two things:
      //
      // a) since each pixel could have two DIFFERENT rays in the
      // queue (shadow ray and bounce ray) we cannot simply 'add', but
      // have to use an atomic add, because these could be in the same
      // warp.
      //
      // b) since we don't have an explicit frame buffer clear we
      // still have to make sure each pixel is written - not added -
      // exactly once in the first generation of the first frame.
      // ==================================================================
      if (accumID == 0 && generation == 0) {
        // if (path.dbg) printf("init frag %f %f %f\n",fragment.x,fragment.y,fragment.z);
        valueToAccumInto = make_float4(fragment.x,fragment.y,fragment.z,0.f);
      } else {
        // if (path.dbg) printf("adding frag %f %f %f\n",fragment.x,fragment.y,fragment.z);

        if (fragment.x > 0.f)
          atomicAdd(&valueToAccumInto.x,fragment.x);
        if (fragment.y > 0.f)
          atomicAdd(&valueToAccumInto.y,fragment.y);
        if (fragment.z > 0.f)
          atomicAdd(&valueToAccumInto.z,fragment.z);
      }

      // and for apps that need a depth buffer, write z
      if (generation == 0) {
        float &tile_z = accumTiles[tileID].depth[tileOfs];
        if (accumID == 0) 
          tile_z = z;
        else
          tile_z = min(tile_z,z);
      }
    }
  }  

  using namespace render;
  
  void DeviceContext::shadeRays_launch(GlobalModel *model,
                                       TiledFB *fb,
                                       int generation)
  {
    SetActiveGPU forDuration(device);
    int numRays = rays.numActive;
    int bs = 128;
    int nb = divRoundUp(numRays,bs);

    static RenderMode renderMode = RENDER_MODE_UNDEFINED;
    if (renderMode == RENDER_MODE_UNDEFINED) {
      const char *_fromEnv = getenv("BARNEY_RENDER");
      if (!_fromEnv)
        _fromEnv = "pt";
      const std::string mode = _fromEnv;
      if (mode == "AO" || mode == "ao")
        renderMode = RENDER_MODE_AO;
      else if (mode == "PT" || mode == "pt")
        renderMode = RENDER_MODE_PT;
      else if (mode == "local" || mode == "LOCAL")
        renderMode = RENDER_MODE_LOCAL;
      else
        throw std::runtime_error("unknown barney render mode '"+mode+"'");
    }

    DevGroup *dg = device->devGroup;
    World *world = &model->getSlot(dg->lmsIdx)->world;

    if (nb) {
      switch(renderMode) {
#if 0
      case RENDER_MODE_LOCAL:
        g_shadeRays_pt<0>
          <<<nb,bs,0,device->launchStream>>>
          (world->getDD(device),
           fb->accumTiles,fb->owner->accumID,
           rays.traceAndShadeReadQueue,numRays,
           rays.receiveAndShadeWriteQueue,rays._d_nextWritePos,generation);
        break;
      case RENDER_MODE_AO:
        g_shadeRays_pt<1>
          <<<nb,bs,0,device->launchStream>>>
          (world->getDD(device),
           fb->accumTiles,fb->owner->accumID,
           rays.traceAndShadeReadQueue,numRays,
           rays.receiveAndShadeWriteQueue,rays._d_nextWritePos,generation);
        break;
      case RENDER_MODE_PT:
#else
      default:
#endif

        g_shadeRays_pt<8><<<nb,bs,0,device->launchStream>>>
          (world->getDD(device),
           fb->accumTiles,fb->owner->accumID,
           rays.traceAndShadeReadQueue,numRays,
           rays.receiveAndShadeWriteQueue,rays._d_nextWritePos,generation);
        break;

      }
    }
  }

}
