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

#pragma once

#include "barney/Object.h"

namespace barney {

  const std::string to_string(BNDataType type);
  
  struct Data : public Object {
    typedef std::shared_ptr<Data> SP;

    Data(DataGroup *owner,
         BNDataType type,
         size_t numItems);
    virtual ~Data() = default;
    
    static Data::SP create(DataGroup *owner,
                           BNDataType type,
                           size_t numItems,
                           const void *items);
    
    BNDataType type  = BN_DATA_UNDEFINED;
    size_t     count = 0;
  };

  struct PODData : public Data {
    PODData(DataGroup *owner,
                           BNDataType type,
                           size_t numItems,
            const void *items);
    virtual ~PODData();
    
    OWLBuffer  owl   = 0;
  };

  /*! data array over reference-counted barney object handles (e.g.,
      BNTexture's, BNlight's, etc. this has to make sure that objects
      put into this data array will remain properly refcoutned. */
  struct ObjectRefsData : public Data {
    ObjectRefsData(DataGroup *owner,
                   BNDataType type,
                   size_t numItems,
                   const void *items);
    std::vector<Object::SP> items;
  };
  
};
