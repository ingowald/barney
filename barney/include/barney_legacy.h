/*! helper function to fill in a BNCamera structure from a more
    user-friendly from/at/up/fovy specification */
BARNEY_API
void bnPinholeCamera(BNCamera  camera,
                     bn_float3 from,
                     bn_float3 at,
                     bn_float3 up,
                     float     fovy,
                     float     aspect);








// // ------------------------------------------------------------------
// // soon to be deprecated, but still the only way to create those
// // ------------------------------------------------------------------
// BARNEY_API
// BNScalarField bnUMeshCreate(BNContext context,
//                             int whichSlot,
//                             // vertices, 4 floats each (3 floats position,
//                             // 4th float scalar value)
//                             const float4 *vertices, int numVertices,
//                             /*! array of all the vertex indices of all
//                                 elements, one after another;
//                                 ie. elements with different vertex
//                                 counts can come in any order, so a
//                                 mesh with one tet and one hex would
//                                 have an index array of size 12, with
//                                 four for the tet and eight for the
//                                 hex */
//                             const int *_indices, int numIndices,
//                             /*! one int per logical element, stating
//                                 where in the indices array it's N
//                                 differnt vertices will be located */
//                             const int *_elementOffsets,
//                             int numElements,
//                             // // tets, 4 ints in vtk-style each
//                             // const int *tets,       int numTets,
//                             // // pyramids, 5 ints in vtk-style each
//                             // const int *pyrs,       int numPyrs,
//                             // // wedges/tents, 6 ints in vtk-style each
//                             // const int *wedges,     int numWedges,
//                             // // general (non-guaranteed cube/voxel) hexes, 8
//                             // // ints in vtk-style each
//                             // const int *hexes,      int numHexes,
//                             // //
//                             // int numGrids,
//                             // // offsets into gridIndices array
//                             // const int *_gridOffsets,
//                             // // grid dims (3 floats each)
//                             // const int *_gridDims,
//                             // // grid domains, 6 floats each (3 floats min corner,
//                             // // 3 floats max corner)
//                             // const float *gridDomains,
//                             // // grid scalars
//                             // const float *gridScalars,
//                             // int numGridScalars,
//                             const bn_float3 *domainOrNull=0);


// BARNEY_API
// BNScalarField bnBlockStructuredAMRCreate(BNContext context,
//                                          int whichSlot,
//                                          /*TODO:const float *cellWidths,*/
//                                          // block bounds, 6 ints each (3 for min,
//                                          // 3 for max corner)
//                                          const int *blockBounds, int numBlocks,
//                                          // refinement level, per block,
//                                          // finest is level 0,
//                                          const int *blockLevels,
//                                          // offsets into blockData array
//                                          const int *blockOffsets,
//                                          // block scalars
//                                          const float *blockScalars, int numBlockScalars);


// struct BNMaterialHelper {
//   bn_float3   baseColor     { .7f,.7f,.7f };
//   float       transmission  { 0.f };
//   float       ior           { 1.45f };
//   float       metallic      { 1.f };
//   float       roughness     { 0.f };
//   BNTexture2D alphaTexture  { 0 };
//   BNTexture2D colorTexture  { 0 };
// };

// /*! c++ helper function */
// inline void bnSetAndRelease(BNObject target, const char *paramName,
//                             BNObject value)
// { bnSetObject(target,paramName,value); bnRelease(value); }

// /*! c++ helper function */
// inline void bnSetAndRelease(BNObject target, const char *paramName,
//                             BNData value)
// { bnSetData(target,paramName,value); bnRelease(value); }
  
// /*! helper function for assinging leftover BNMaterial definition from old API */
// inline void bnAssignMaterial(BNGeom geom,const BNMaterialHelper *material)
// {
//   bnSet3fc(geom,"material.baseColor",material->baseColor);
//   bnSet1f(geom,"material.transmission",material->transmission);
//   bnSet1f(geom,"material.ior",material->ior);
//   if (material->colorTexture)
//     bnSetObject(geom,"material.colorTexture",material->colorTexture);
//   if (material->alphaTexture)
//     bnSetObject(geom,"material.alphaTexture",material->alphaTexture);
//   bnCommit(geom);
// }




// // ------------------------------------------------------------------
// // DEPRECATED
// // ------------------------------------------------------------------
// BARNEY_API
// BNGeom bnTriangleMeshCreate(BNContext context,
//                             int whichSlot,
//                             const BNMaterialHelper *material,
//                             const int3 *indices,
//                             int numIndices,
//                             const float3 *vertices,
//                             int numVertices,
//                             const float3 *normals,
//                             const float2 *texcoords);

// // ------------------------------------------------------------------
// // DEPRECATED
// // ------------------------------------------------------------------
// BARNEY_API
// BNScalarField bnStructuredDataCreate(BNContext context,
//                                      int whichSlot,
//                                      int3 dims,
//                                      BNDataType /*ScalarType*/ type,
//                                      const void *scalars,
//                                      float3 gridOrigin,
//                                      float3 gridSpacing);

inline void bnSetAndRelease(BNObject o, const char *n, BNObject v)
{
  bnSetObject(o,n,v);
  bnRelease(v);
}
inline void bnSetAndRelease(BNObject o, const char *n, BNData v)
{
  bnSetData(o,n,v);
  bnRelease(v);
}
