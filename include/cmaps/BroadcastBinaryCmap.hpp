//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _BroadcastBinaryCmap
#define _BroadcastBinaryCmap

#include "Cmaps2.hpp"

namespace cnine{

  class BroadcastBinaryCmap: public Direct_cmap{ // public Cmap_base, 
  public:

    int I;
    
    template<typename OP, typename ARR>
    BroadcastBinaryCmap(const OP& op, ARR& r, const decltype(r.get_cell(0))& x, const ARR& y, const int add_flag=0){
      I=r.aasize;
      assert(y.aasize==I);
      if(r.dev==0){
	for(int i=0; i<r.aasize; i++){
	  decltype(r.get_cell(0)) t=r.cell(i);
	  op.apply(t,x,y.cell(i),add_flag);
	  //if(add_flag==0) op.apply(t,x,y.cell(i));
	  //else op.add(t,x,y.cell(i));
	}
      }
      if(r.dev==1){
	op.apply(*this,r,ARR(x),y,add_flag);
      }
    }

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I);
    }

    __device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(i,0,i);
    }

#endif 

  };


  // ---- Templates ------------------------------------------------------------------------------------------


  template<typename OP, typename OBJ, typename ARR>
  ARR broadcast(const OBJ& x, const ARR& y){
    ARR r(x,y.adims,fill::raw);
    BroadcastBinaryCmap(OP(),r,x,y);
    return r;
  }

  template<typename OP, typename OBJ, typename ARR, typename ARG0>
  ARR broadcast(const OBJ& x, const ARR& y, const ARG0& arg0){
    ARR r(x,y.adims,fill::raw);
    BroadcastBinaryCmap(OP(arg0),r,x,y);
    return r;
  }


  template<typename OP, typename OBJ, typename ARR>
  void add_broadcast(ARR& r, const OBJ& x, const ARR& y){
    BroadcastBinaryCmap(OP(),r,x,y,1);
  }

  template<typename OP, typename OBJ, typename ARR, typename ARG0>
  void add_broadcast(ARR& r, const OBJ& x, const ARR& y, const ARG0& arg0){
    BroadcastBinaryCmap(OP(arg0),r,x,y,1);
  }

}

#endif


