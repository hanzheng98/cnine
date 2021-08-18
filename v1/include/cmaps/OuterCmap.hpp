//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _OuterCmap
#define _OuterCmap

#include "Cmaps2.hpp"


namespace cnine{

  class OuterCmap: public Direct_cmap{ // public Cmap_base, 
  public:

    int I,J;
    int n;

    //__host__ __device__ OuterCmap(const OuterCmap& x): I(x.I), J(x.J), n(x.n){
    //printf("copied\n");
    //}

    template<typename OP, typename ARR>
    OuterCmap(const OP& op, ARR& r, const ARR& x, const ARR& y, const int add_flag=0){
      I=x.aasize;
      J=y.aasize;
      n=y.aasize;
      if(r.dev==0){
	assert(r.aasize==I*J);
       	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    decltype(x.get_cell(0)) t=r.cell(i*J+j);
	    op.apply(t,x.cell(i),y.cell(j),add_flag);
	    //if(add_flag==0) op.apply(t,x.cell(i),y.cell(j));
	    //else op.add(t,x.cell(i),y.cell(j));
	  }
      }
      if(r.dev==1){
	op.apply(*this,r,x,y,add_flag);
      }
    }

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I,J);
    }

    __device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
      //printf("%d\n",n);
      return thrust::make_tuple(i*n+j,i,j);
    }

#endif 

  };



  // ---- Templates ------------------------------------------------------------------------------------------


  template<typename OP, typename ARR>
  ARR outer(const ARR& x, const ARR& y){
    ARR r(x,dims(x.get_adim(0),y.get_adim(0)),fill::raw);
    OuterCmap(OP(),r,x,y);
    return r;
  }

  template<typename OP, typename ARR, typename ARG0>
  ARR outer(const ARR& x, const ARR& y, const ARG0& arg0){
    ARR r(x,dims(x.get_adim(0),y.get_adim(0)),fill::raw);
    OuterCmap(OP(arg0),r,x,y);
    return r;
  }


  template<typename OP, typename ARR>
  void add_outer(ARR& r, const ARR& x, const ARR& y){
    OuterCmap(OP(),r,x,y,1);
  }

  template<typename OP, typename ARR, typename ARG0>
  void add_outer(ARR& r, const ARR& x, const ARR& y, const ARG0& arg0){
    OuterCmap(OP(arg0),r,x,y,1);
  }


}

#endif 

