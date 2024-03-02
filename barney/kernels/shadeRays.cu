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
#define DEFAULT_RADIANCE_FROM_ENV .8f
  
    enum { MAX_PATH_DEPTH = 10 };
  
    typedef enum {
      RENDER_MODE_UNDEFINED,
      // RENDER_MODE_LOCAL,
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

#if 0
    __global__
    void g_shadeRays_local(AccumTile *accumTiles,
                           int accumID,
                           Ray *readQueue,
                           int numRays,
                           Ray *writeQueue,
                           int *d_nextWritePos,
                           int generation)
    {
      if (generation != 0) return;
    
      int tid = threadIdx.x + blockIdx.x*blockDim.x;
      if (tid >= numRays) return;

      Ray ray = readQueue[tid];
    
      vec3f albedo = (vec3f)ray.hit.baseColor;
      vec3f fragment = 0.f;
      float z = INFINITY;
      if (!ray.hadHit) {
        fragment = (vec3f)ray.hit.baseColor;
      } else {
        z = ray.tMax;
        vec3f dir = ray.dir;
        vec3f Ng = ray.hit.N;
        const bool isVolumeHit = (Ng == vec3f(0.f));
        if (!isVolumeHit) Ng = normalize(Ng);
        float NdotD = dot(Ng,normalize(dir));
        if (NdotD > 0.f) Ng = - Ng;
      
        // let's do some ambient eyelight-style shading, anyway:
        float scale
          = isVolumeHit
          ? .5f
          : (.2f + .4f*fabsf(NdotD));
        fragment
          = albedo
          * scale
          * ray.throughput;
      }
      int tileID  = ray.pixelID / pixelsPerTile;
      int tileOfs = ray.pixelID % pixelsPerTile;
    
      float4 &valueToAccumInto
        = accumTiles[tileID].accum[tileOfs];
      float  &tile_z
        = accumTiles[tileID].depth[tileOfs];
      vec4f valueToAccum = make_float4(fragment.x,fragment.y,fragment.z,0.f);
      if (accumID > 0)
        valueToAccum = valueToAccum + (vec4f)valueToAccumInto;
    
      if (generation == 0) {
        if (accumID == 0)
          tile_z = z;
        else
          tile_z = min(tile_z,z);
      }

      valueToAccumInto = valueToAccum;
    }
#endif




    __global__
    void g_shadeRays_ao(AccumTile *accumTiles,
                        int accumID,
                        Ray *readQueue,
                        int numRays,
                        Ray *writeQueue,
                        int *d_nextWritePos,
                        int generation)
    {
      int tid = threadIdx.x + blockIdx.x*blockDim.x;
      if (tid >= numRays) return;

      Ray ray = readQueue[tid];
    
      vec3f albedo = ray.hit.getAlbedo();//(vec3f)ray.hit.baseColor;
      vec3f fragment = 0.f;
      float z = INFINITY;
      // if (0 && ray.dbg) {
      //   printf("============================================ ray: hadHit %i, t %f P %f %f %f base %f %f %f N %f %f %f\n",
      //          ray.hadHit,
      //          ray.tMax,
      //          (float)ray.hit.P.x,
      //          (float)ray.hit.P.y,
      //          (float)ray.hit.P.z,
      //          (float)ray.hit.baseColor.x,
      //          (float)ray.hit.baseColor.y,
      //          (float)ray.hit.baseColor.z,
      //          (float)ray.hit.N.x,
      //          (float)ray.hit.N.y,
      //          (float)ray.hit.N.z);
      // }
      if (!ray.hadHit) {
        if (generation == 0) {
          // for primary rays we have pre-initialized basecolor to a
          // background color in generateRays(); let's just use this, so
          // generaterays can pre--set whatever color it wasnts for
          // non-hitting rays
          fragment
            = ray.hit.missColor;
          // .crossHair
          //   ? vec3f(1.f,0.f,0.f)
          //   : (vec3f)ray.hit.miss.color;

        } else {
          vec3f ambientIllum = vec3f(1.f);
          fragment = ray.throughput * ambientIllum;
        }
      } else {
        z = ray.tMax;
        vec3f dir = ray.dir;
        vec3f Ng = ray.hit.getN();
        const bool isVolumeHit = (Ng == vec3f(0.f));
        if (!isVolumeHit) Ng = normalize(Ng);
        float NdotD = dot(Ng,normalize(dir));
        if (NdotD > 0.f) Ng = - Ng;
      
        // let's do some ambient eyelight-style shading, anyway:
      
        const float eyeLightWeight
          = isVolumeHit
          ? .5f
          : (.2f + .4f*fabsf(NdotD));
        const float ao_ambient_component = .1f;

        const float scale = ao_ambient_component * eyeLightWeight;
        // scale *= 0.001f;
        vec3f tp = ray.throughput;
        fragment
          = albedo
          * scale
          * ray.throughput;
        // if (ray.dbg) {
        //   printf("gen %i fragment %f %f %f\n",generation,fragment.x,fragment.y,fragment.z);
        //   printf("gen %i Ng %f %f %f\n",generation,Ng.x,Ng.y,Ng.z);
        //   printf("gen %i albedo %f %f %f\n",generation,albedo.x,albedo.y,albedo.z);
        //   printf("gen %i tp %f %f %f\n",generation,tp.x,tp.y,tp.z);
        // }
      
        // and then add a single diffuse bounce (ae, ambient occlusion)
        Random &rng = (Random &)ray.rngSeed;
        if (ray.hadHit && generation == 0) {
          Ray bounce;
          bounce.org = ray.hit.P + 1e-5f*Ng;
          // if (ray.dbg)
          // printf("bounce org %f %f %f\n",
          //        bounce.org.x,
          //        bounce.org.y,
          //        bounce.org.z);
          bounce.dir = normalize(Ng + randomDirection(rng));
          bounce.tMax = INFINITY;
          bounce.dbg = ray.dbg;
          bounce.hadHit = false;
          bounce.pixelID = ray.pixelID;
          rng();
          bounce.rngSeed = ray.rngSeed;
          rng();
          bounce.throughput = 
            // .6f *
            .8f *
            ray.throughput * albedo;
          writeQueue[atomicAdd(d_nextWritePos,1)] = bounce;
        }
      }
      int tileID  = ray.pixelID / pixelsPerTile;
      int tileOfs = ray.pixelID % pixelsPerTile;
    
      float4 &valueToAccumInto
        = accumTiles[tileID].accum[tileOfs];
      float  &tile_z
        = accumTiles[tileID].depth[tileOfs];
      vec4f valueToAccum = make_float4(fragment.x,fragment.y,fragment.z,0.f);

      // if (ray.dbg)
      //   printf("gen %i accumulating %f %f %f %f\n",
      //          generation,
      //          valueToAccum.x,
      //          valueToAccum.y,
      //          valueToAccum.z,
      //          valueToAccum.w);
    
      if (accumID > 0)
        valueToAccum = valueToAccum + (vec4f)valueToAccumInto;
    
      if (generation == 0) {
        if (accumID == 0)
          tile_z = z;
        else
          tile_z = min(tile_z,z);
      }
        
      valueToAccumInto = valueToAccum;
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
      // if (dbg) printf("num dirlights %i\n",world.numDirLights);
    
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
        // if (dbg) printf("light #%i, dir %f %f %f weight %f\n",lID[i],lightDir.x,lightDir.y,lightDir.z,weight);
        if (weight <= 1e-3f) continue;
        weight *= reduce_max(light.radiance);
        if (weight <= 1e-3f) continue;
        // if (dbg) printf("radiance %f %f %f weight %f\n",light.radiance.x,light.radiance.y,light.radiance.z,weight);
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
  
    inline __device__
    vec3f reflect(const vec3f &v,
                  const vec3f &n)
    {
      return v - 2.0f*dot(v, n)*n;
    }
  


    inline __device__
    bool scatter_glass(vec3f &scattered_direction,
                       Random &random,
                       // const vec3f &org,
                       const vec3f &dir,
                       // const vec3f &P,
                       vec3f N,
                       const float ior
                       // ,
                       // PerRayData &prd
                       )
    {
      // const vec3f org   = optixGetWorldRayOrigin();
      // const vec3f dir   = normalize((vec3f)optixGetWorldRayDirection());

      // N = normalize(N);
      vec3f outward_normal;
      vec3f reflected = reflect(dir,N);
      float ni_over_nt;
      // prd.out.attenuation = vec3f(1.f, 1.f, 1.f); 
      vec3f refracted;
      float reflect_prob;
      float cosine;
  
      if (dot(dir,N) > 0.f) {
        outward_normal = -N;
        ni_over_nt = ior;
        cosine = dot(dir, N);// / vec3f(dir).length();
        cosine = sqrtf(1.f - ior*ior*(1.f-cosine*cosine));
      }
      else {
        outward_normal = N;
        ni_over_nt = 1.0 / ior;
        cosine = -dot(dir, N);// / vec3f(dir).length();
      }
      if (refract(dir, outward_normal, ni_over_nt, refracted)) 
        reflect_prob = schlick(cosine, ior);
      else 
        reflect_prob = 1.f;

      // prd.out.scattered_origin = P;
      if (random() < reflect_prob) 
        // prd.out.
        scattered_direction = reflected;
      else 
        // prd.out.
        scattered_direction = refracted;
  
      return true;
    }


    inline __device__
    vec3f sampleCosineWeightedHemisphere(vec3f Ns, Random &random)
    {
      while (1) {
        vec3f p = 2.f*vec3f(random(),random(),random()) - vec3f(1.f);
        if (dot(p,p) > 1.f) continue;
        return normalize(normalize(p)
                         +Ns//vec3f(0.f,0.f,1.f)
                         );
      }
    }
  
  inline __device__ float pbrt_clampf(float f, float lo, float hi)
  { return max(lo,min(hi,f)); }
  
  inline __device__ float pbrtSphericalTheta(const vec3f &v)
  {
    return acosf(pbrt_clampf(v.z, -1.f, 1.f));
  }
  
  inline __device__ float pbrtSphericalPhi(const vec3f &v)
  {
    float p = atan2f(v.y, v.x);
    return (p < 0.f) ? (p + float(2.f * M_PI)) : p;
  }

  
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
      } else
        return DEFAULT_RADIANCE_FROM_ENV;
    }

    /*! return dedicated background, if specifeid; otherwise return envmap color */
    inline __device__
    vec3f backgroundOrEnv(const World::DD &world,
                          Ray &ray)
    {
      if (world.envMapLight.texture)
        return radianceFromEnv(world,ray);
      return
        ray.hit.missColor;
      // .crossHair
      //   ? vec3f(1.f)
      //   : ray.hit.missolor;
    }

    inline __device__
    float safe_eps(float f, vec3f v)
    {
      return max(f,1e-5f*reduce_max(abs(v)));
    }

    /*! ugh - that should all go into material::AnariPhysical .... */
    inline __device__
    void bounce(const World::DD &world,
                vec3f &fragment,
                Ray &path,
                Ray &shadowRay,
                int pathDepth)
    {
      const float EPS = 1e-4f;

      const bool  hadNoIntersection  = !path.hadHit;
      const vec3f incomingThroughput = path.throughput;
      shadowRay.tMax = -1;

      // if (path.dbg)
      //   printf(" -> incoming %f %f %f dir %f %f %f %f\n",
      //          path.org.x,
      //          path.org.y,
      //          path.org.z,
      //          (float)path.dir.x,
      //          (float)path.dir.y,
      //          (float)path.dir.z,
      //          path.tMax);
    
      if (path.isShadowRay) {
        // ==================================================================
        // shadow ray = all we have to do is add carried radiance if it
        // reached the light, and discards
        // ==================================================================
        if (hadNoIntersection) 
          fragment = path.throughput;
        // this path is done.
        path.tMax = 0.f;
        return;
      }

      vec3f Ng = path.hit.getN();
      const bool  isVolumeHit        = (Ng == vec3f(0.f));
      if (!isVolumeHit)
        Ng = normalize(Ng);
      const vec3f notFaceForwardedNg = Ng;
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
          fragment = path.throughput * backgroundOrEnv(world,path);

          const vec3f fromEnv = 1.5f*backgroundOrEnv(world,path);
          const vec3f tp = path.throughput;
          const vec3f addtl = tp
            * fromEnv;
          fragment = addtl;
          
        } else {
          // ----------------------------------------------------------------
          // SECONDARY ray that didn't hit anything -> env-light
          // ----------------------------------------------------------------
          // this path had at least one bounce, but now bounced into
          // nothingness - compute env-light contribution, and weigh it
          // with the path's carried throughput.
          fragment = path.throughput * radianceFromEnv(world,path);
        }
        // no outgoing rays; this path is done.
        path.tMax = -1.f;
        return;
      }
      if (pathDepth == 0.f)
        path.throughput = vec3f(1.f);
    

      // ==================================================================
      // this ray DID hit something: compute its local frame buffer
      // contribution at this hit point (if any), and generate secondary
      // ray and shadow ray (if applicable), with proper weights.
      // ==================================================================    
      Random &random = (Random &)path.rngSeed;
    
      const bool doTransmission
        =  ((float)path.hit.mini.transmission > 0.f)
        && (random() < (float)path.hit.mini.transmission);
      render::DG dg;
      dg.N = Ng;
      dg.w_o = -(vec3f)path.dir;
      // if (path.dbg)
      //   printf("(%i) hit trans %f ior %f, dotrans %i\n",
      //          pathDepth,
      //          (float)path.hit.transmission,
      //          (float)path.hit.ior,
      //          int(doTransmission));

      if (/* for non-glass this SHOULD be done by isec program! */doTransmission) {
        // ------------------------------------------------------------------
        // transmission, refleciton, or refraction
        // ------------------------------------------------------------------
        vec3f dir = from_half(path.dir);
        scatter_glass(dir,
                      random,
                      path.dir,notFaceForwardedNg,
                      path.hit.mini.ior);
        path.dir = dir;
        if (dot(dir,notFaceForwardedNg) > 0.f) {
          path.org = path.hit.P + safe_eps(EPS,path.hit.P)*Ng;
        } else {
          path.org = path.hit.P - safe_eps(EPS,path.hit.P)*Ng;
          path.isInMedium = 1;
        }
        /* ************* TODO - MISSING SOME METALLIC/REFLECTANCE HERE *********** */
      } else {
        // ------------------------------------------------------------------
        // not perfectly specular - do diffuse bounce for now...
        // ------------------------------------------------------------------

        // save local path weight for the shadow ray:
        path.org = path.hit.P + safe_eps(EPS,path.hit.P)*Ng;
        if (isVolumeHit) {
          path.dir = sampleCosineWeightedHemisphere(-vec3f(path.dir),random);
          path.throughput = .8f * path.throughput * path.hit.getAlbedo();//hit.baseColor;
        } else { 
          path.dir = sampleCosineWeightedHemisphere(dg.N,random);
          path.throughput = path.throughput * path.hit.eval(dg,path.dir);//baseColor;
        }
      }

      // ------------------------------------------------------------------
      // so far we HAVE generated an outgoing path, but it have haev
      // very low throughput/weak impact on the image - use russian
      // roulette to either ake it 'stronger', or kill it.
      // ------------------------------------------------------------------
      if (pathDepth < MAX_PATH_DEPTH) {
        path.dir = normalize(path.dir);
        path.tMax   = INFINITY;
        path.hadHit = false;

        float maxWeight = reduce_max((vec3f)path.throughput);
        if (maxWeight < .3f) {
          if (random() < maxWeight) {
            path.throughput = path.throughput * (1.f/maxWeight);
          } else {
            path.tMax = -1.f;
          }
        }
      } else
        path.tMax = -1.f;
    
      // if (path.dbg)
      //   printf(" -> outgoing %f %f %f dir %f %f %f %f\n",
      //          path.org.x,
      //          path.org.y,
      //          path.org.z,
      //          (float)path.dir.x,
      //          (float)path.dir.y,
      //          (float)path.dir.z,
      //          path.tMax);
    
    
      // ==================================================================
      // now, check for shadow ray
      // ==================================================================
      LightSample ls;
      if (!doTransmission && sampleLights(ls,world,path.hit.P,Ng,random,path.dbg)) {
        shadowRay.makeShadowRay(/* thrghhpt */(incomingThroughput*ls.L)*(1.f/ls.pdf)
                                * path.hit.eval(dg,ls.dir),
                                /* surface: */path.hit.P + EPS*Ng,
                                /* to light */ls.dir,
                                /* length   */ls.dist * (1.f-2.f*EPS));
        shadowRay.rngSeed = path.rngSeed + 1;
        shadowRay.dbg = path.dbg;
        shadowRay.pixelID = path.pixelID;
      }
    }
  

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
      // if (path.dbg) printf("  # shading FROM %lx TO %lx\n",
      //                      readQueue,writeQueue);
      // what we'll add into the frame buffer
      vec3f fragment = 0.f;
      float z = path.tMax;
      // create a (potential) shadow ray, and init to 'invalid'
      Ray shadowRay;

        // printf("sammpling dir for N %f %f %f\n",dg.N.x,dg.N.y,dg.N.z);

      // bounce that ray on the scene, possibly generating a) a fragment
      // to add to frame buffer; b) a outgoing ray (in-place
      // modification of 'path'); and/or c) a shadow ray
      bounce(world,fragment,
             path,shadowRay,
             generation);
    
      // write shadow and bounce ray(s), if any were generated
      if (path.tMax > 0.f)
        writeQueue[atomicAdd(d_nextWritePos,1)] = path;
      if (shadowRay.tMax > 0.f) 
        writeQueue[atomicAdd(d_nextWritePos,1)] = shadowRay;

      // and write the shade fragment, if generated
      int tileID  = path.pixelID / pixelsPerTile;
      int tileOfs = path.pixelID % pixelsPerTile;
      float4 &valueToAccumInto
        = accumTiles[tileID].accum[tileOfs];
      vec4f valueToAccum = make_float4(fragment.x,fragment.y,fragment.z,0.f);
      if (accumID > 0)
        valueToAccum = valueToAccum + (vec4f)valueToAccumInto;
      valueToAccumInto = valueToAccum;

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
      // else if (mode == "local")
      //   renderMode = RENDER_MODE_LOCAL;
      else
        throw std::runtime_error("unknown barney render mode '"+mode+"'");
    }

    DevGroup *dg = device->devGroup;
    World *world = &model->getSlot(dg->lmsIdx)->world;

    if (nb) {
      switch(renderMode) {
      // case RENDER_MODE_LOCAL:
      //   g_shadeRays_local<<<nb,bs,0,device->launchStream>>>
      //     (fb->accumTiles,fb->owner->accumID,
      //      rays.traceAndShadeReadQueue,numRays,
      //      rays.receiveAndShadeWriteQueue,rays._d_nextWritePos,generation);
      //   break;
      case RENDER_MODE_AO:
        g_shadeRays_ao<<<nb,bs,0,device->launchStream>>>
          (fb->accumTiles,fb->owner->accumID,
           rays.traceAndShadeReadQueue,numRays,
           rays.receiveAndShadeWriteQueue,rays._d_nextWritePos,generation);
        break;
      case RENDER_MODE_PT:
        g_shadeRays_pt<<<nb,bs,0,device->launchStream>>>
          (world->getDD(device),
           fb->accumTiles,fb->owner->accumID,
           rays.traceAndShadeReadQueue,numRays,
           rays.receiveAndShadeWriteQueue,rays._d_nextWritePos,generation);
        break;
      }
    }
  }

}
