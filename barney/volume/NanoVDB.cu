// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/Context.h"
#include "barney/volume/NanoVDB.h"
#include "barney/common/Texture.h"
#include "rtcore/ComputeInterface.h"
#include "barney/volume/MCGrid.cuh"

namespace BARNEY_NS {

  RTC_IMPORT_USER_GEOM(/*file*/NanoVDB,/*name*/NanoVDBMC_float,
                       /*geomtype device data */
                       MCVolumeAccel<NanoVDBDataSampler<float>>::DD,false,false);
  RTC_IMPORT_USER_GEOM(/*file*/NanoVDB,/*name*/NanoVDBMC_Iso_float,
                       /*geomtype device data */
                       MCIsoSurfaceAccel<NanoVDBDataSampler<float>>::DD,false,false);


  // NanoVDBData::PLD *NanoVDBData::getPLD(Device *device)
  // { return &perLogical[device->contextRank()]; }

  
  /*! compute kernel that computes macro-cell information for a 3D
      structured data grid */
  __rtc_global
  void NanoVDBData_float_computeMCs(const rtc::ComputeInterface &ci,
                                    /* kernel ARGS */
                                    MCGrid::DD mcGrid,
                                    NanoVDBData::DD dd)
  {
    vec3i mcDims = mcGrid.dims;
    int tid = ci.launchIndex().x;
    if (tid >= mcDims.x*mcDims.y*mcDims.z) return;
    vec3i mcID(tid  %  mcDims.x,
               (tid /  mcDims.x) % mcDims.y,
               tid  / (mcDims.x  * mcDims.y));

    if (mcID.x >= mcGrid.dims.x) return;
    if (mcID.y >= mcGrid.dims.y) return;
    if (mcID.z >= mcGrid.dims.z) return;
    
    const box3f mcBounds(mcGrid.gridOrigin+vec3f(mcID)*mcGrid.gridSpacing,
                         mcGrid.gridOrigin+vec3f(mcID+1)*mcGrid.gridSpacing);

    nanovdb::CoordBBox bbox(
        {(int)mcBounds.lower.x,(int)mcBounds.lower.y,(int)mcBounds.lower.z},
        {(int)mcBounds.upper.x+1,(int)mcBounds.upper.y+1,(int)mcBounds.upper.z+1});

    // nanovdb::NanoGrid<float> *gridPtr = (nanovdb::NanoGrid<float> *)gridData;
    using GridType = typename nanovdb::Grid<nanovdb::NanoTree<float>>;
    GridType *gridPtr = (GridType *)dd.gridData;
    auto acc = gridPtr->getAccessor();

    range1f scalarRange;
    for (nanovdb::CoordBBox::Iterator iter = bbox.begin(); iter; ++iter) {
      float value = acc.getValue(*iter);
      scalarRange.extend(value);
    }
    int cellIdx = mcID.x + mcGrid.dims.x * (mcID.y + mcGrid.dims.y * (mcID.z));
    // if (cellIdx % 1230 == 0)
    //   printf("cell %i %i %i range %f %f\n",
    //          mcID.x,mcID.y,mcID.z,
    //          scalarRange.lower,
    //          scalarRange.upper);
    mcGrid.scalarRanges[cellIdx] = scalarRange;
    // const box4f cellBounds(vec4f(mcBounds.lower,scalarRange.lower),
    //                        vec4f(mcBounds.upper,scalarRange.upper));
    // rasterBox(mcGrid,getBox(dd.worldBounds),cellBounds);
  }

  
  NanoVDBData::NanoVDBData(Context *context,
                                 const DevGroup::SP &devices)
    : ScalarField(context,devices)
  {}

  
  template<typename T>
  void NanoVDBDataSampler<T>::build()
  {
    // nothign to do!?
  }

  MCGrid::SP NanoVDBData::buildMCs() 
  {
    if (gridType != nanovdb::GridType::Float)
      throw std::runtime_error("grid type is not float");
    
    MCGrid::SP mcGrid = std::make_shared<MCGrid>(devices);
    vec3i mcDims = vec3i(128);//divRoundUp(numCells,vec3i(cellsPerMC));
    mcGrid->resize(mcDims);
    mcGrid->gridOrigin = worldBounds.lower;
    mcGrid->gridSpacing = worldBounds.size() * rcp(vec3f(mcDims));
    
    mcGrid->clearCells();
    PING;
    for (auto device : *devices) {
      size_t lc64 = (size_t)mcDims.x*(size_t)mcDims.y*(size_t)mcDims.z;
      int lc = int(lc64);
      assert(lc == lc64);
      int bs = 128;
      int nb = divRoundUp(lc,bs);
      __rtc_launch(device->rtc,
                   NanoVDBData_float_computeMCs,
                   nb,bs,
                   mcGrid->getDD(device),
                   getDD(device));
    }
    PING;
    for (auto device : *devices)
      device->sync();
    PING;
    return mcGrid;
  }

  template<typename T>
  typename NanoVDBDataSampler<T>::DD NanoVDBDataSampler<T>::getDD(Device *device)
  {
    assert(sf);
    DD dd;
    // dd.texObj = sf->texture->getDD(device);
    // assert(dd.texObj);
    // dd.cellGridOrigin  = sf->gridOrigin;
    // dd.cellGridSpacing = sf->gridSpacing;
    // dd.numCells        = sf->numCells;
  // sf.data.nvdbRegular.voxelSize = m_voxelSize;
  // sf.data.nvdbRegular.origin = m_bounds.lower;
  // sf.data.nvdbRegular.gridData = m_deviceBuffer.ptr();
  // sf.data.nvdbRegular.gridType = m_gridMetadata->gridType();
    // dd.worldBounds = sf->worldBounds;

    dd.voxelSize   = sf->voxelSize;
    assert(sf->data);
    dd.nvdbGrid    = (NVDBGridT*)sf->data->getDD(device);
    assert(dd.nvdbGrid);
    // dd.gridType = m_gridMetadata->gridType();

    // PING; PRINT(dd.worldBounds);

    return dd;
  }

  NanoVDBData::DD NanoVDBData::getDD(Device *device)
  {
    assert(this->data);
    
    DD dd;
    dd.worldBounds = worldBounds;
    dd.voxelSize   = voxelSize;
    dd.gridType    = gridType;
    dd.gridSize    = gridSize;
    dd.gridData    = data->getDD(device);
    
    PING; PRINT(worldBounds);
    assert(dd.gridData);
    return dd;
  }
  
  VolumeAccel::SP NanoVDBData::createAccel(Volume *volume) 
  {
    auto sampler = std::make_shared<NanoVDBDataSampler<float>>(this);
    return std::make_shared<MCVolumeAccel<NanoVDBDataSampler<float>>>
      (volume,
       createGeomType_NanoVDBMC_float,
       sampler);
  }
  
  IsoSurfaceAccel::SP NanoVDBData::createIsoAccel(IsoSurface *isoSurface) 
  {
    auto sampler = std::make_shared<NanoVDBDataSampler<float>>(this);
    return std::make_shared<MCIsoSurfaceAccel<NanoVDBDataSampler<float>>>
      (isoSurface,
       createGeomType_NanoVDBMC_Iso_float,
       sampler);
  }
  
  // ==================================================================
  bool NanoVDBData::set1i(const std::string &member,
                          const int &value)
  {
    // if (member == "gridType") {
    //   assert(sizeof(gridType) == sizeof(int));
    //   (int&)gridType = value;
    //   return true;
    // }
    return false;
  }
  
  // ==================================================================
  bool NanoVDBData::set3f(const std::string &member,
                          const vec3f &value) 
  {
    // if (member == "worldBounds.lower") {
    //   worldBounds.lower = value;
    //   return true;
    // }
    // if (member == "worldBounds.upper") {
    //   worldBounds.upper = value;
    //   return true;
    // }
    // if (member == "voxelSize") {
    //   voxelSize = value;
    //   return true;
    // }
    return false;
  }

  // ==================================================================
  bool NanoVDBData::setObject(const std::string &member,
                                 const Object::SP &value) 
  {
    if (member == "data") {
      PING;
      data = value->as<PODData>();
      assert(data);
      // BNTextureAddressMode addressModes[3] = {
      //   BN_TEXTURE_CLAMP,BN_TEXTURE_CLAMP,BN_TEXTURE_CLAMP
      // };
      // texture = std::make_shared<Texture>((Context*)context,scalars,
      //                                     BN_TEXTURE_LINEAR,
      //                                     addressModes,
      //                                     BN_COLOR_SPACE_LINEAR);
      // textureNN = std::make_shared<Texture>((Context*)context,scalars,
      //                                       BN_TEXTURE_NEAREST,
      //                                       addressModes,
      //                                       BN_COLOR_SPACE_LINEAR);
      return true;
    }
    return false;
  }
  // ==================================================================
  bool NanoVDBData::setData(const std::string &member,
                            const Data::SP &value) 
  {
    if (member == "data") {
      PING;
      data = value->as<PODData>();
      assert(data);
      // BNTextureAddressMode addressModes[3] = {
      //   BN_TEXTURE_CLAMP,BN_TEXTURE_CLAMP,BN_TEXTURE_CLAMP
      // };
      // texture = std::make_shared<Texture>((Context*)context,scalars,
      //                                     BN_TEXTURE_LINEAR,
      //                                     addressModes,
      //                                     BN_COLOR_SPACE_LINEAR);
      // textureNN = std::make_shared<Texture>((Context*)context,scalars,
      //                                       BN_TEXTURE_NEAREST,
      //                                       addressModes,
      //                                       BN_COLOR_SPACE_LINEAR);
      return true;
    }
    return false;
  }

  // ==================================================================
  void NanoVDBData::commit() 
  {
    PING;

    // Data might not be aligned, make sure we get something that works for
    // nanovdb.
    size_t numBytes = data->size();
    PRINT(numBytes);
    
    auto dev0 = (*devices)[0];
    auto nvHostBuffer = nanovdb::HostBuffer::create(numBytes);
    data->download(dev0,nvHostBuffer.data());
    
    // SetActiveGPU forDuration(dev0);
    // PODData::PLD *dataPLD = data->getPLD(dev0);
    
    // rtc::Buffer  *dataBuffer = dataPLD->rtcBuffer;
    // // std::memcpy(hostbuffer.data(), m_data->data(AddressSpace::HOST), m_data->size());
    // std::memcpy(hostbuffer.data(),
    //             dataPLD->data->getDD(), data->size());

    auto gridHandle = nanovdb::GridHandle<>(std::move(nvHostBuffer));
    const nanovdb::GridMetaData *gridMetadata = gridHandle.gridMetaData();
    auto boundsMin = gridMetadata->worldBBox().min();
    auto boundsMax = gridMetadata->worldBBox().max();
    
    worldBounds
      = box3f(vec3f(boundsMin[0], boundsMin[1], boundsMin[2]),
              vec3f(boundsMax[0], boundsMax[1], boundsMax[2]));
    nanovdb::Vec3d nvVoxelSize = gridMetadata->voxelSize();
    voxelSize = vec3f((const vec3d&)nvVoxelSize);
    PING; PRINT(worldBounds);
    
    auto nvGridSize = gridMetadata->indexBBox().dim();
    gridSize.x = nvGridSize[0];
    gridSize.y = nvGridSize[1];
    gridSize.z = nvGridSize[2];
    PRINT(gridSize);
    PRINT(worldBounds);
    PRINT(voxelSize);
    
    gridType = gridMetadata->gridType();
    
    // for (auto device : *devices) {
    //   PLD *pld = getPLD(device); 
    //   auto rtc = device->rtc;
    //   SetActiveGPU forDuration(device);
    //   pld->gridData = rtc::alloc(gridHandle.size());
    //   // rtc::copy(pld->gridData,static_cast<const std::byte *>(gridHandle.data()),
    //   //           gridHandle.size());
    // }
  }
  
}

