//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _RtensorA_copy_cop
#define _RtensorA_copy_cop

#include "GenericCop.hpp"
#include "Cmaps2.hpp"


namespace cnine{

#ifdef _WITH_CUDA

  template<typename CMAP>
  void RtensorA_copy_cu(const CMAP& map, RtensorArrayA& r, const RtensorArrayA& x, const cudaStream_t& stream);

  template<typename CMAP>
  void RtensorA_add_copy_cu(const CMAP& map, RtensorArrayA& r, const RtensorArrayA& x, const cudaStream_t& stream);

  template<typename CMAP>
  void RtensorA_copy_accumulator_cu(const CMAP& map, RtensorArrayA& r, const RtensorArrayA& x, const cudaStream_t& stream);

#endif 


  class RtensorA_copy_cop{
  public:

    RtensorA_copy_cop(){}


    void apply(RtensorA& r, const RtensorA& x) const{
      r=x;
    }

    void add(RtensorA& r, const RtensorA& x) const{
      r.add(x);
    }


    template<typename CMAP>
    void apply(const CMAP& cmap, RtensorArrayA& r, const RtensorArrayA& x) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
	RtensorA_copy_cu(cmap,r,x,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#endif
    }

    template<typename CMAP>
    void add(const CMAP& cmap, RtensorArrayA& r, const RtensorArrayA& x) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
	RtensorA_add_copy_cu(cmap,r,x,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#endif
    }

    template<typename CMAP>
    void accumulate(const CMAP& cmap, RtensorArrayA& r, const RtensorArrayA& x) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
	RtensorA_copy_accumulator_cu(cmap,r,x,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#endif
    }


  };


}

#endif


  //template<typename CMAP, typename = typename std::enable_if<std::is_base_of<DirectCmap, CMAP>::value, CMAP>::type>
  //void CtensorA_copy_cu(const CMAP& map, CtensorArrayA& r, const CtensorArrayA& x, const cudaStream_t& stream);

  //template<typename CMAP, typename = typename std::enable_if<std::is_base_of<AccumulatorCmap, CMAP>::value, CMAP>::type>
  //void CtensorA_copy_cu(const CMAP& map, CtensorArrayA& r, const CtensorArrayA& x, const cudaStream_t& stream);

    /*
    template<typename CMAP, typename = typename std::enable_if<std::is_base_of<AccumulatorCmap, CMAP>::value, CMAP>::type>
    void operator()(const CMAP& map, CtensorArrayA& r, const CtensorArrayA& x) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
      CtensorA_copy_accumulator_cu(map,r,x,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#else
      CNINE_NOCUDA_ERROR;
#endif
    }
    */
    /*
    template<typename CMAP>
    void operator()(const CMAP& cmap, CtensorArrayA& r, const CtensorArrayA& x) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
	CtensorA_copy_cu(cmap,r,x,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#endif
    }
    */

    /*
    void operator()(CtensorA& r, const CtensorA& x) const{
      r=x;
    }
    */

