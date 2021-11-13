//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _RtensorA_add_plus_cop
#define _RtensorA_add_plus_cop

#include "GenericCop.hpp"

// deprecated!!! 

namespace cnine{

#ifdef _WITH_CUDA
  template<typename CMAP>
  void RtensorA_add_plus_cu(const CMAP& map, RtensorArrayA& r, const RtensorArrayA& x, const RtensorArrayA& y, 
    const cudaStream_t& stream);
#endif 


  class RtensorA_add_plus_cop: public BinaryCop<RtensorA,RtensorArrayA>{
  public:

    RtensorA_add_plus_cop(){}
    
    void operator()(RtensorA& r, const RtensorA& x, const RtensorA& y) const{
      r.add(x);
      r.add(y);
    }

    template<typename IMAP>
    void operator()(const IMAP& map, RtensorArrayA& r, const RtensorArrayA& x, const RtensorArrayA& y) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
      RtensorA_add_plus_cu(map,r,x,y,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#else
      CNINE_NOCUDA_ERROR;
#endif
    }

    static string shortname(){
      return "add_plus";
    }

  };

}

#endif
