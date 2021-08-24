// This file is part of cnine, a lightweight C++ tensor library. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _Cnine_Cmaps2
#define _Cnine_Cmaps2

#include "Cnine_base.hpp"

namespace cnine{


  class Cmap_base{
  public:

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(0);
    }

    __device__ int n_accum(const int b) const{
      return 0;
    }

    __device__ int target(const int b) const{
      return 0;
    }

    __device__ int lst_ptr(const int b) const{
      return 0;
    }

    __device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(0,0,0);
    }

    __device__ thrust::tuple<int,int> source(const int lst, const int b, const int i) const{
      return thrust::make_tuple(0,0);
    }

#endif 

  };


  class Direct_cmap: public Cmap_base{
  public:

  };

  class Masked2_cmap: public Cmap_base{
  public:

  };



  /*
  class DirectCmap{
  public:
  };

  class UnaryDirectCmap: public DirectCmap{
  public:
  };

  class BinaryDirectCmap: public DirectCmap{
  public:
  };



  class AccumulatorCmap{
  public:
  };
  */
  




}

#endif 
