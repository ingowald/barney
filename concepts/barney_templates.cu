#include <owl/owl_device.h>
#include <optix_device.h>
#include <owl/owl.h>
#include <owl/common/math/box.h>
#include <vector>

using namespace owl::common;

// ==================================================================

OWLContext getOWL()
{
  extern OWLContext ctx;
  return ctx;
}
  

namespace cuBQL {
  struct bvh_t { int bla; };
};

struct TransferFunction {
  struct DD {
    inline __device__
    vec4f map(float f) const { return values[int(f)]; }
    
    vec4f *values;
    int    numValues;
  };
  static void addVars(std::vector<OWLVarDecl> &vars, int ofs)
  {
    vars.push_back(OWLVarDecl{"xf.numValues",OWL_INT,ofs+OWL_OFFSETOF(DD,numValues)});
  }
  void setVariables(OWLGeom geom)
  { owlGeomSet1i(geom,"xf.numValues",values.size()); }
  std::vector<vec4f> values;
};

struct ScalarField {
  struct DD {};
  virtual void setVariables(OWLGeom geom) = 0;
};

/*! HOST part of somethign that forms a specific volume implementation */
struct VolumeAccel {
  VolumeAccel(ScalarField *field, TransferFunction *xf)
    : xf(xf), field(field)
  {}

  void setVariables(OWLGeom geom)
  {
    field->setVariables(geom);
    xf->setVariables(geom);
  }
  
  TransferFunction *xf;
  ScalarField *field;

  // virtual std::string getTypeName() const = 0;
  // virtual OWLGeomType createGT() const = 0;
};

/*! DEVICE part of somethign that forms a specific volume implementation */
template<typename SampleableField>
struct VolumeOver {
  struct DD {
    typename SampleableField::DD field;
    TransferFunction::DD xf;

    inline __device__ vec4f sampleAndMap(vec3f P) const
    {
      float f = field.sample(P);
      return xf.map(f);
    }
  };
  static void addVars(std::vector<OWLVarDecl> &vars, int ofs)
  {
    SampleableField::addVars(vars,ofs+OWL_OFFSETOF(DD,field));
    TransferFunction::addVars(vars,ofs+OWL_OFFSETOF(DD,xf));
  }
  
};


// ==================================================================
// let's define some actual fields

struct UMeshField : public ScalarField {
  struct DD : public ScalarField::DD {
    inline __device__ float primScalar(int tetID, vec3f P) const
    { return verts[tets[tetID].x].x; }
    
    vec4i *tets;
    vec4f *verts;
  };
  static void addVars(std::vector<OWLVarDecl> &vars, int ofs)
  {
    vars.push_back(OWLVarDecl{"mesh.tets",OWL_BUFPTR,ofs+OWL_OFFSETOF(DD,tets)});
  }

  struct SamplerTravState {
    inline __device__
    void testPrim(const DD &dd, int primID, vec3f P)
    { retVal = dd.primScalar(primID,P); }

    inline __device__ float returnValue() { return retVal; }
    
    float retVal = NAN;
  };
  
  static std::string getProgName()
  { return "UMeshField"; }

  void setVariables(OWLGeom geom) override
  {
    owlGeomSetBuffer(geom,"mesh.tets",tetsBuffer);
  }
                            
  std::vector<vec4i> tets;
  std::vector<vec4f> verts;
  OWLBuffer tetsBuffer;
  OWLBuffer vertsBuffer;
};

struct ChomboField : public ScalarField {
  struct Block { float v[8][8][8]; };

  struct DD : public ScalarField::DD {
    Block *blocks;
  };
  static void addVars(std::vector<OWLVarDecl> &vars, int ofs)
  {
    vars.push_back(OWLVarDecl{"chombo.blocks",OWL_BUFPTR,ofs+OWL_OFFSETOF(DD,blocks)});
  }
  static std::string getProgName()
  { return "ChomboField"; }


  struct SamplerTravState {
    inline __device__
    void testPrim(const DD &dd, int primID, vec3f P)
    {
      auto &block = dd.blocks[primID];
      float weight = block.v[(int)P.x][(int)P.y][(int)P.z];
      sumWeights += weight;
      sumWeightedValues += P.x*weight;
    }

    inline __device__ float returnValue()
    { return sumWeights == 0.f ? float(NAN) : (sumWeightedValues/sumWeights); }

    float sumWeights = 0.f;
    float sumWeightedValues = 0.f;
  };
  
  
  std::vector<Block> blocks;
};

// ==================================================================
// let's make those fields sampleable

namespace cuBQL {
  template<typename PrimCodeLambda>
  inline __device__ void traverse(cuBQL::bvh_t bvh, const PrimCodeLambda &primCode)
  {
    primCode(bvh.bla);
  }
}

/*! HOST SIDE ONLY */
struct CUBQLSamplerHost {
  CUBQLSamplerHost(ScalarField *);
  
  void build();
  void setVariables(OWLGeom geom);
  
  ScalarField *field;
  cuBQL::bvh_t bvh;
};
/*! DEVICE SIDE */
template<typename Field>
struct CUBQLSampler {
  using HostSide = CUBQLSamplerHost;
  
  struct DD : public Field::DD {
    cuBQL::bvh_t bvh;

    inline __device__
    float sample(vec3f P) const
    {
      typename Field::SamplerTravState prd;
      // float retVal = NAN;
      cuBQL::traverse(bvh,[&](int primID){
        prd.testPrim(*this,primID,P);
      });
      return prd.returnValue();
    }
  };

  static void addVars(std::vector<OWLVarDecl> &vars, int ofs)
  {
    Field::addVars(vars,ofs);
    printf("todo");
  }
  
  static std::string getProgName()
  { return "CUBQLSampler_"+Field::getProgName(); }
  
  /* does NOT have a createGT() because it will never be its own owl
     geom - it'll only ever contribute (templated) routines TO ANOTHER
     geom created by the VolumeGeom */
};

// ==================================================================


struct MCGrid {
  struct DD {
    vec3i dims;
    float *majorants;
    float2 *cells;
  };
  vec3i dims;
  std::vector<float> majorants;

  void buildMCs(ScalarField *field)
  { printf("alloc grid, then call field->rasterInto()\n"); }
  void buildMajorants(TransferFunction *xf)
  { printf("compute majorants from existing macro cells and xf\n"); }
  
  static void addVars(std::vector<OWLVarDecl> &vars, int ofs)
  {
    vars.push_back(OWLVarDecl{"mcGrid.dims",OWL_INT3,ofs+OWL_OFFSETOF(DD,dims)});
  }
  void setVariables(OWLGeom geom)
  { owlGeomSet3i(geom,"mcGrid.dims",3,3,3); }
};


template<typename CellCodeLambda>
inline __device__ void dda3(MCGrid::DD dd, const CellCodeLambda &lambda)
{
  for (int ix=0;ix<dd.dims.x;ix++)
    lambda(vec3i(ix,ix,ix));
}


// struct DDATraverserHost
// {
//   DDATraverserHost(ScalarField *field)
//     : field(field)
//   {}
  
//   void build(OWLGeomType (*createGT)(const char *), const char *ptxCode)
//   {
//     mcGrid.build(field);
//     geom = owlGeomCreate(getOWL(),createGT(ptxCode));
//     owlGeomSetPrimCount(geom,1);
//     setVariables(geom);
//   }

//   void setVariables(OWLGeom geom)
//   {
//     mcGrid.setVariables(geom);
//     field->setVariables(geom);
//   }
  
//   MCGrid mcGrid;
//   ScalarField *field;
//   OWLGeom geom = 0;
// };

struct MCGridAccel {
  MCGridAccel(VolumeAccel *volume,
              OWLGeom (*createGeom)(MCGridAccel *, const char *),
              const char *ptxCode)
    : volume(volume),
      ptxCode(ptxCode)
  {}

  void prepareInitialBuild()
  {
    // gt = createGT(ptxCode);

    mcGrid.buildMCs(volume->field);
    geom = createGeom(this,ptxCode);
    // geom = owlGeomCreate(getOWL(),createGT(ptxCode));
    setVariables(geom);
    // owlGeomSetPrimCount(geom,1);
  }
  
  void build()
  {
    mcGrid.buildMajorants(volume->xf);
    setVariables(geom);
  }

  void setVariables(OWLGeom geom)
  {
    mcGrid.setVariables(geom);
    volume->setVariables(geom);
  }
  
  MCGrid mcGrid;
  VolumeAccel *volume;
  // OWLGeomType gt = 0;
  OWLGeom geom = 0;
  OWLGeom (*createGeom)(MCGridAccel *, const char *);
  const char *ptxCode;
};
  
// /* HIGH-PRI: MERGE WITH DDATraverserHost !!!! */
// struct RTXMCTraverserHost
// {
//   RTXMCTraverserHost(ScalarField *field)
//     : field(field)
//   {}
  
//   void build(OWLGeomType (*createGT)(const char *), const char *ptxCode)
//   {
//     mcGrid.build(field);
//     geom = owlGeomCreate(getOWL(),createGT(ptxCode));
//     owlGeomSetPrimCount(geom,1);
//     setVariables(geom);
//   }

//   void setVariables(OWLGeom geom)
//   {
//     mcGrid.setVariables(geom);
//     field->setVariables(geom);
//   }
  
//   MCGrid mcGrid;
//   ScalarField *field;
//   OWLGeom geom = 0;
// };


/*! class that creates a optix user geometry with a single prim
    covering the entire DDA grid, and whose intersection code performs
    a cuda dda traversal through a macro cell grid, performs woodcock
    trackign in each (non-empty) macro cells, and and calls the
    respective SampleVolume::sampleAndMap() function for each woodcock
    step */
template<typename SampleableVolume>
struct DDATraverserGeom {
  typedef typename SampleableVolume::HostSide SamplerHost;
  using TravHost = MCGridAccel;
  
  struct DD : public VolumeOver<SampleableVolume>::DD {
    MCGrid::DD mcGrid;

    inline __device__ void intersect(int /*ignoreForThisClass*/) const
    {
      dda3(this->mcGrid,[&](vec3i cell) {
        for (float t=0.f;t<1.f;t+= .1f) {
          vec4f sample = this->sampleAndMap(vec3f(cell.x+t));
          if (sample.w > 0.f) return;
        }
      });
    }

  };

  static OWLGeom createGeom(MCGridAccel *mcGridAccel, const char *ptxCode)
  {
    xxx();
  }
  static void addVars(std::vector<OWLVarDecl> &vars, int ofs)
  {
    VolumeOver<SampleableVolume>::addVars(vars,ofs);
    MCGrid::addVars(vars,ofs+OWL_OFFSETOF(DD,mcGrid));
  }
  static OWLGeomType createGT(const char *ptxCode);
  
  static std::string getProgName()
  { return "DDATraverserGeom_"+SampleableVolume::getProgName(); }
  
  void build();
};


/*! class that creates a optix user geometry with one prim per
    (non-empty) macro cell, and whose intersection code looks up the
    repsecigve cell, then performs woodcock trackign in said cell;
    calling the respective SampleVolume::sampleAndMap() function for
    each woodcock step */
template<typename SampleableVolume>
struct RTXMCTraverserGeom {
  typedef typename SampleableVolume::HostSide SamplerHost;
  using TravHost = MCGridAccel;
  
  struct DD : public VolumeOver<SampleableVolume>::DD {
    MCGrid::DD mcGrid;
    
    inline __device__ void intersect(int primID) const
    {
      vec3i cell(primID,1,2);
      for (float t=0.f;t<1.f;t+= .1f) {
        vec4f sample = this->sampleAndMap(vec3f(cell.x+t));
        if (sample.w > 0.f) return;
      }
    }
  };

  static OWLGeom createGeom(MCGridAccel *mcGridAccel, const char *ptxCode)
  {
    yyy();
  }
  static void addVars(std::vector<OWLVarDecl> &vars, int ofs)
  {
    VolumeOver<SampleableVolume>::addVars(vars,ofs);
    MCGrid::addVars(vars,ofs+OWL_OFFSETOF(DD,mcGrid));
  }
  static OWLGeomType createGT(const char *ptxCode);
  
  static std::string getProgName()
  { return "RTXMCTraverserGeom_"+SampleableVolume::getProgName(); }
  
  void build();
};








template<typename SampleableVolumeWithMCGrid>
OWLGeomType DDATraverserGeom<SampleableVolumeWithMCGrid>::createGT(const char *ptxCode)
{
  OWLContext owl = getOWL();
  std::vector<OWLVarDecl> vars;
  addVars(vars,0);

  OWLModule module = owlModuleCreate(getOWL(),ptxCode);
  OWLGeomType gt = owlGeomTypeCreate
    (owl,OWL_GEOM_USER,
     sizeof(typename SampleableVolumeWithMCGrid::DD),vars.data(),vars.size());
  std::string progName = getProgName();
  owlGeomTypeSetIntersectProg(gt,0,module,progName.c_str());
  return gt;
}





template<typename DDType>
struct VolumeAccelGeomFor : public VolumeAccel
{
  VolumeAccelGeomFor(ScalarField *field,
                     TransferFunction *xf,
                     const char *ptxCode)
    : VolumeAccel(field,xf),
      traverser(this,DDType::createGeom,ptxCode),
      ptxCode(ptxCode),
      sampler(field)
  { build(); }

  void build()
  {
    sampler.build();
    traverser.build();
  }

  const char *ptxCode;
  typename DDType::TravHost    traverser;
  typename DDType::SamplerHost sampler;
};


// ==================================================================
// THIS HERE IS FOR **GEOMS*** THAT WILL HAVE AN ISEC PROGRAM ON THE
// ==================================================================

// ..................................................................
extern char UMeshField_mcDDA_sampleCUBQL_ptx[];

VolumeAccel *create_UMeshField_mcDDA_sampleCUBQL(UMeshField *field,
                                                   TransferFunction *xf)
{ return new VolumeAccelGeomFor<DDATraverserGeom<CUBQLSampler<UMeshField>>>
    (field,xf,UMeshField_mcDDA_sampleCUBQL_ptx); }

OPTIX_INTERSECT_PROGRAM(DDATraverserGeom_CUBQLSampler_UMeshField)()
{
  typedef typename DDATraverserGeom<CUBQLSampler<UMeshField>>::DD DD;
  auto &self = owl::getProgramData<DD>();
  self.intersect(optixGetPrimitiveIndex());
}


// ..................................................................
extern char UMeshField_mcRTX_sampleCUBQL_ptx[];

VolumeAccel *create_UMeshField_mcRTX_sampleCUBQL(UMeshField *field,
                                                 TransferFunction *xf)
{ return new VolumeAccelGeomFor<RTXMCTraverserGeom<CUBQLSampler<UMeshField>>>
    (field,xf,UMeshField_mcRTX_sampleCUBQL_ptx); }

OPTIX_INTERSECT_PROGRAM(RTXTraverserGeom_CUBQLSampler_UMeshField)()
{
  typedef typename RTXMCTraverserGeom<CUBQLSampler<UMeshField>>::DD DD;
  auto &self = owl::getProgramData<DD>();
  self.intersect(optixGetPrimitiveIndex());
}


// ..................................................................
extern char ChomboField_mcDDA_sampleCUBQL_ptx[];

VolumeAccel *create_ChomboField_mcDDA_sampleCUBQL(ChomboField *field,
                                                    TransferFunction *xf)
{ return new VolumeAccelGeomFor<DDATraverserGeom<CUBQLSampler<ChomboField>>>
    (field,xf,ChomboField_mcDDA_sampleCUBQL_ptx); }

OPTIX_INTERSECT_PROGRAM(DDATraverserGeom_CUBQLSampler_ChomboField)()
{
  typedef typename DDATraverserGeom<CUBQLSampler<ChomboField>>::DD DD;
  auto &self = owl::getProgramData<DD>();
  self.intersect(optixGetPrimitiveIndex());
}

// // ..................................................................
// extern char UMeshFieldGeom_mcDDA_sampleRTX_ptx[];
// VolumeAccel *create_UMeshField_mcDDA_sampleRTX(UMeshField *field,
//                                                  TransferFunction *xf)
// { return new VolumeAccelFor<DDATraverser<RTXSampler<UMeshField>>>
//     (field,xf,UMeshField_mcDDA_sampleRTX_ptx); }

// OPTIX_INTERSECT_PROGRAM(DDATraverser_RTXSampler_UMeshField)()
// {
//   typedef typename DDATraverser<CUBQLSampler<UMeshField>>::DD DD;
//   auto &self = owl::getProgramData<DD>();
//   DDATraverser<CUBQLSampler<UMeshField>>::traverse(self);
// }

// // ..................................................................
// extern char ChomboFieldGeom_mcDDA_sampleRTX_ptx[];
// VolumeAccel *create_ChomboFieldGeom_mcDDA_sampleRTX(ChomboField *field,
//                                                  TransferFunction *xf)
// {
//   return new VolumeAccelFor<DDATraverser<RTXSampler<ChomboField>>>
//     (field,xf,ChomboField_mcDDA_sampleRTX_ptx);
// }

// OPTIX_INTERSECT_PROGRAM(DDATraverser_RTXSampler_ChomboField)()
// {
//   typedef typename DDATraverser<CUBQLSampler<ChomboField>>::DD DD;
//   auto &self = owl::getProgramData<DD>();
//   DDATraverser<CUBQLSampler<ChomboField>>::traverse(self);
// }


