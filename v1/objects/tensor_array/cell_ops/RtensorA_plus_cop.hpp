//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _RtensorA_plus_cop
#define _RtensorA_plus_cop

#include "GenericCop.hpp"
#include "Cmaps2.hpp"


namespace cnine{

#ifdef _WITH_CUDA

  template<typename CMAP>
  void RtensorA_plus_cu(const CMAP& map, RtensorArrayA& r, const RtensorArrayA& x, const RtensorArrayA& y, 
    const cudaStream_t& stream, const int add_flag);

  template<typename CMAP>
  void RtensorA_plus_accumulator_cu(const CMAP& map, RtensorArrayA& r, const RtensorArrayA& x, const RtensorArrayA& y, const cudaStream_t& stream);

#endif 


  class RtensorA_plus_cop{ //: public BinaryCop<RtensorA,RtensorArrayA>{
  public:

    RtensorA_plus_cop(){}

    /*
    void operator()(RtensorA& r, const RtensorA& x, const RtensorA& y) const{
      r.set(x);
      r.add(y);
    }
    */

    void apply(RtensorA& r, const RtensorA& x, const RtensorA& y, const int add_flag=0) const{
      if(add_flag==0){
	r.set(x);
	r.add(y);
      }else{
	r.add(x);
	r.add(y);
      }
    }
 
    template<typename CMAP, typename = typename std::enable_if<std::is_base_of<Direct_cmap,CMAP>::value, CMAP>::type>
    void apply(const CMAP& map, RtensorArrayA& r, const RtensorArrayA& x, const RtensorArrayA& y, 
      const int add_flag=0) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
      RtensorA_plus_cu(map,r,x,y,stream,add_flag);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#endif
    }

    template<typename CMAP, typename = typename std::enable_if<std::is_base_of<Masked2_cmap,CMAP>::value, CMAP>::type>
    void accumulate(const CMAP& map, RtensorArrayA& r, const RtensorArrayA& x, const RtensorArrayA& y, const int add_flag=0) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
      RtensorA_plus_accumulator_cu(map,r,x,y,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#endif
    }

  };

}

#endif


