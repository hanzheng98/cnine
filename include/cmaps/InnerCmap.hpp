//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _InnerCmap
#define _InnerCmap

#include "Cmaps2.hpp"


namespace cnine{

  class InnerCmap: public Masked2_cmap{
  public:

    int I;
    //int J=0;

    template<typename OP, typename ARR>
    InnerCmap(const OP& op, ARR& r, const ARR& x, const ARR& y, const int add_flag=0){
      I=x.aasize;
      assert(y.aasize==I);
      if(r.dev==0){
	decltype(r.get_cell(0)) t=r.cell(0);
       	for(int i=0; i<I; i++)
	  op.apply(t,x.cell(i),y.cell(i),add_flag);
	  //if(add_flag==0) op.apply(t,x.cell(i),y.cell(i));
	  //else op.add(t,x.cell(i),y.cell(i));
      }
      if(r.dev==1){
	//op.apply(*this,r,x,y,add_flag);
	op.accumulate(*this,r,x,y,add_flag);
      }
    }

    /*
    dim3 blockdims() const{
      return dim3(I);
    }

    __device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(0,k,k);
    }
    */

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(1);
    }

    __device__ int n_accum(const int b) const{
      return I;
    }

    __device__ int target(const int b) const{
      return 0;
    }

    __device__ int lst_ptr(const int b) const{
      return 0;
    }

    __device__ thrust::tuple<int,int> source(const int lst, const int b, const int i) const{
      return thrust::make_tuple(i,i);
    }

#endif 

  };


  // ---- Templates ------------------------------------------------------------------------------------------


  template<typename OP, typename ARR>
  ARR inner(const ARR& x, const ARR& y){
    ARR r(x,dims(1),fill::zero);
    InnerCmap(OP(),r,x,y);
    return r;
  }

  template<typename OP, typename ARR>
  void add_inner(ARR& r, const ARR& x, const ARR& y){
    InnerCmap(OP(),r,x,y,1);
  }

}

#endif 



