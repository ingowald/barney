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

#define ENV_LIGHT_SAMPLING 1

#define USE_MIS 1
    
    // inline __device__ float abs(float f) { return fabsf(f); }
    
  inline __device__ float square(float f) { return f*f; }
  
    
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

    inline __device__
    bool sampleAreaLights(Light::Sample &ls,
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
      QuadLight::DD light;
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
      ls.direction
        = normalize(LD);
      ls.distance
        = length(LD);
      ls.radiance
        = light.emission
        * (light.area * -dot(light.normal,ls.direction)
           / square(ls.distance));
      ls.pdf
        = weights[i]/sumWeights
        * (float(RESERVOIR_SIZE)/float(world.numQuadLights));
      if (ls.pdf <= 0.f)
        printf("invalid area light PDF %f from i %i weight %f sum %f\n",
               ls.pdf,i,weights[i],sumWeights);
      return true;
    }

    inline __device__
    bool sampleDirLights(Light::Sample &ls,
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
      DirLight::DD light;
    
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
      ls.direction
        = -light.direction;
      ls.distance
        = INFINITY;
      ls.radiance
        = light.radiance;
      ls.pdf
        = weights[i]/sumWeights
        * (float(RESERVOIR_SIZE)/float(world.numDirLights));
      return weights[i] != 0.f;
    }

    inline __device__
    bool sampleEnvLight(Light::Sample &ls,
                        const World::DD &world,
                        const vec3f P,
                        const vec3f N,
                        Random &random,
                        bool dbg)
    {
      /* in barney, the environment is either a explicit hdri map (in
         EnvMapLight); or a uniform brightness of 'world.radiance' */
      if (world.envMapLight.texture)
        ls = world.envMapLight.sample(random,dbg);
      else {
        ls.direction = randomDirection(random);
        ls.radiance  = world.radiance;
        ls.pdf       = ONE_OVER_FOUR_PI;
        ls.distance  = INFINITY;
      }
      return true;
    }

    inline __device__
    bool sampleLights(Light::Sample &ls,
                      const World::DD &world,
                      const vec3f P,
                      const vec3f Ng,
                      Random &random,
#if USE_MIS
                      bool &lightNeedsMIS,
#endif
                      bool dbg)
    {
#if USE_MIS
      lightNeedsMIS = false;
#endif
      Light::Sample als;
      float alsWeight
        = (sampleAreaLights(als,world,P,Ng,random,dbg)
           ? (reduce_max(als.radiance)/als.pdf)
           : 0.f);
      Light::Sample dls;
      float dlsWeight
        = (sampleDirLights(dls,world,P,Ng,random,dbg)
           ? (reduce_max(dls.radiance)/dls.pdf)
           : 0.f);
      
#if ENV_LIGHT_SAMPLING
      Light::Sample els;
      float elsWeight
        = (sampleEnvLight(els,world,P,Ng,random,dbg)
           ? (reduce_max(els.radiance)/els.pdf)
           : 0.f);
        // = world.envMapLight.sample(random,dbg);
#else
      float elsWeight = 0.f;
#endif
      
      float sumWeights
        = alsWeight+dlsWeight+elsWeight;
      if (sumWeights == 0.f) return false;

      elsWeight *= 1.f/sumWeights;
      alsWeight *= 1.f/sumWeights;
      dlsWeight *= 1.f/sumWeights;
      
      float r = random();
      if (r <= alsWeight) {
        ls = als;
        ls.pdf *= alsWeight;
#if ENV_LIGHT_SAMPLING
      } else if (r <= alsWeight+elsWeight) {
        ls = els;
        ls.pdf *= elsWeight;
# if USE_MIS
      lightNeedsMIS = true;
# endif
#endif
      } else {
        ls = dls;
        ls.pdf *= dlsWeight;
      }
      // if (dbg)
      //   printf(" light weights %f %f\n",
      //          alsWeight,dlsWeight);
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
    {
      auto &env = world.envMapLight;
      if (env.texture) {
        vec3f d = xfmVector(env.toLocal,normalize(ray.dir));
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

    /*! if there _is_ a dedicated env-map light specified, this looks
        up the background color from that map; otherwise, it returns
        the 'ray.misscolor' that the primary ray generation has set as
        default color for this ray */
    inline __device__
    vec3f primaryRayMissColor(const World::DD &world,
                          Ray &ray)
    {
      if (world.envMapLight.texture)
        return radianceFromEnv(world,ray);
      return
        // primary rays do store a default misscolor in the ray itself
        // - we simply return this if there's no env-map.
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
        printf("(%i) ------------------------------------------------------------------\n -> incoming %f %f %f dir %f %f %f t %f ismiss %i, bsdf %i\n",
               pathDepth,
               path.org.x,
               path.org.y,
               path.org.z,
               (float)path.dir.x,
               (float)path.dir.y,
               (float)path.dir.z,
               path.tMax,int(hadNoIntersection),(int)path.bsdfType);

      if (path.isShadowRay) {
        // ==================================================================
        // shadow ray = all we have to do is add carried radiance if it
        // reached the light, and discards
        // ==================================================================
                 
        if (hadNoIntersection) {
          // fragment = clamp((vec3f)path.throughput,vec3f(0.f),vec3f(1.f));
          fragment =
# if USE_MIS
            path.misWeight *
#endif
            (vec3f)path.throughput;
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
      vec3f Ngff = Ng;
      if (!hitWasOnFront)
        Ngff = - Ng;

      if (hadNoIntersection) {
        // ==================================================================
        // regular ray that did NOT hit ANYTHING 
        // ==================================================================
        if (pathDepth == 0) {
          // ----------------------------------------------------------------
          // PRIMARY ray that didn't hit anything -> background
          // ----------------------------------------------------------------
          // if (path.dbg)
          //   printf("miss primary %f %f %f\n",
          //          path.missColor.x,
          //          path.missColor.y,
          //          path.missColor.z);
          // fragment = path.missColor;
          fragment = primaryRayMissColor(world,path);
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
#if ENV_LIGHT_SAMPLING
# if USE_MIS
          const vec3f fromEnv = radianceFromEnv(world,path);
          fragment = path.throughput * fromEnv * path.misWeight;
# else
          fragment = vec3f(0.f);
# endif
#else
          const vec3f fromEnv = radianceFromEnv(world,path);
          if (0 && path.dbg)
            printf("fromenv %f %f %f\n",
                   fromEnv.x,
                   fromEnv.y,
                   fromEnv.z);
          fragment = path.throughput * fromEnv;
#endif
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
      
      // if (1 && path.dbg)
      //   printf("dg.N %f %f %f\n",
      //          dg.Ns.x,
      //          dg.Ns.y,
      //          dg.Ns.z);

      vec3f frontFacingSurfaceOffset
        = safe_eps(EPS,dg.P)*Ngff;
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
      Light::Sample ls;
      // todo check if BSDF is perfectly specular
#if USE_MIS
      bool lightNeedsMIS = false;
#endif
      if (sampleLights(ls,world,dg.P,Ngff,random,
#if USE_MIS
                       lightNeedsMIS,
#endif
                       0 && path.dbg)
          // && 
          // (path.materialType != GLASS)
          ) {
        if (0 && path.dbg) printf("eval light %f %f %f\n",
                                  ls.direction.x,
                                  ls.direction.y,
                                  ls.direction.z);
        EvalRes f_r
          = bsdf.eval(dg,ls.direction,0 && path.dbg)
          // * fabsf(dot(dg.Ng,ls.direction))
          ;
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
            * (1.f/ls.pdf)
            * f_r.value
            * ls.radiance
            * (isVolumeHit?1.f:fabsf(dot(dg.Ng,ls.direction)))
            /// f_r.pdf
            ;
          shadowRay.makeShadowRay
            (/* thrghhpt */tp_sr,
             /* surface: */dg.P + frontFacingSurfaceOffset,
             /* to light */ls.direction,
             /* length   */ls.distance * (1.f-2.f*EPS));
          // if (path.dbg) printf("new shadow ray len %f %f\n",ls.dist,shadowRay.tMax);
          shadowRay.rngSeed = path.rngSeed + 1; random();
          shadowRay.dbg = path.dbg;
          shadowRay.pixelID = path.pixelID;
#if USE_MIS
          shadowRay.misWeight = 1.f;
#endif
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
      // if (path.dbg) printf("scatter result.valid ? %i\n",
      //                      int(scatterResult.valid()));
      if (!scatterResult.valid() || scatterResult.pdf == 0.f)
        return;

      if (0 && path.dbg)
        printf("offsetting into sign %f, direction %f %f %f\n",
               scatterResult.offsetDirection,
               frontFacingSurfaceOffset.x,
               frontFacingSurfaceOffset.y,
               frontFacingSurfaceOffset.z);
      path.org
        = dg.P + scatterResult.offsetDirection * frontFacingSurfaceOffset;
      if (0 && path.dbg)
        printf("path scattered from %f %f %f to %f %f %f, dot %f\n",
               (float)path.dir.x, 
               (float)path.dir.y, 
               (float)path.dir.z, 
               (float)scatterResult.dir.x, 
               (float)scatterResult.dir.y, 
               (float)scatterResult.dir.z,
               dot(path.dir,scatterResult.dir));
      path.dir        = normalize(scatterResult.dir);

      
      vec3f scatterFactor
        = scatterResult.f_r
        // * fabsf(dot(dg.Ng,path.dir))
        / (isinf(scatterResult.pdf)? 1.f : (scatterResult.pdf + 1e-10f));
      path.throughput
        = path.throughput * scatterFactor;
      path.clearHit();
      if (scatterResult.changedMedium)
        path.isInMedium = !path.isInMedium;
      
      if (0 && path.dbg)
        printf("scatter dir %f %f %f tp %f %f %f\n",
               (float)path.dir.x,
               (float)path.dir.y,
               (float)path.dir.z,
               (float)path.throughput.x,
               (float)path.throughput.y,
               (float)path.throughput.z);

#if USE_MIS
      if (lightNeedsMIS && !isinf(scatterResult.pdf)) {
        float pdf_scatterRay_scatterDir = scatterResult.pdf;
        float pdf_lightRay_lightDir     = ls.pdf;
        float pdf_lightRay_scatterDir   = world.envMapLight.pdf(path.dir);
        float pdf_scatterRay_lightDir   = bsdf.pdf(dg,ls.direction);
        
        path.misWeight
          = pdf_scatterRay_scatterDir
          / (pdf_scatterRay_scatterDir + pdf_lightRay_scatterDir);
        shadowRay.misWeight
          = pdf_lightRay_lightDir
          / (pdf_lightRay_lightDir + pdf_scatterRay_lightDir);
        
        if ((float)shadowRay.misWeight < 1e-5f)
          shadowRay.tMax  = -1.f;
      } else
        path.misWeight = 1.f;
#endif
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

#if 1
      // clamping ...
      float clampMax = 100.f*(1+accumID);
      fragment = min(fragment,vec3f(clampMax));
#endif
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
    World *world = model->getSlot(dg->lmsIdx)->world.get();

    if (nb) {
      World::DD devWorld = world->getDD(device);
           
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
        (devWorld,
         fb->accumTiles,fb->owner->accumID,
           rays.traceAndShadeReadQueue,numRays,
           rays.receiveAndShadeWriteQueue,rays._d_nextWritePos,generation);
        break;

      }
    }
  }

}
