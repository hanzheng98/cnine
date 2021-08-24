//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _MVprodCmap
#define _MVprodCmap

#include "Cmaps2.hpp"


namespace cnine{

  class MVprodCmap: public Masked2_cmap{
  public:
    
    int I;
    int J;

    template<typename OP, typename ARR>
    MVprodCmap(const OP& op, ARR& r, const ARR& x, const ARR& y, const int add_flag=0){
      I=x.get_adim(0);
      J=x.get_adim(1);
      assert(r.aasize==I);
      assert(y.aasize==J);
      if(r.dev==0){
	if(J==0) return;
	for(int i=0; i<I; i++){
	  decltype(r.get_cell(0)) t=r.cell(i);
	  if(add_flag==0) op.apply(t,x.cell({i,0}),y.cell(0),false);
	  for(int j=1-add_flag; j<J; j++){
	    op.apply(t,x.cell({i,j}),y.cell(j),true);
	  }
	}
      }
      if(r.dev==1){
	op.accumulate(*this,r,x,y);
      }
    }

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I);
    }

    __device__ int n_accum(const int b) const{
      return J;
    }

    __device__ int target(const int b) const{
      return b;
    }

    __device__ int lst_ptr(const int b) const{
      return 0;
    }

    __device__ thrust::tuple<int,int> source(const int lst, const int b, const int i) const{
      return thrust::make_tuple(b*J+i,i);
    }

#endif 

  };


  // ---- Templates ------------------------------------------------------------------------------------------


  template<typename OP, typename ARR>
  ARR MVprod(const ARR& x, const ARR& y){
    ARR r(x,x.get_adim(0),fill::zero);
    MVprodCmap(OP(),r,x,y);
    return r;
  }

  template<typename OP, typename ARR>
  void add_MVprod(ARR& r, const ARR& x, const ARR& y){
    MVprodCmap(OP(),r,x,y,1);
  }


}

#endif 




  /*
  class MVprod_cmap{
  public:
    
    int I;
    int J;

    template<typename OP, typename ARR>
    MVprod_cmap(const OP& op, ARR& r, const ARR& x, const ARR& y){
      I=x.get_adim(0);
      J=x.get_adim(1);
      assert(r.aasize==I);
      assert(y.aasize==J);
      if(r.dev==0){
	if(J==0) return;
	for(int i=0; i<I; i++){
	  decltype(r.get_cell(0)) t=r.cell(i);
	  op.set(t,x.cell(i,0),y.cell(0));
	  for(int j=1; j<J; j++){
	    op.add(t,x.cell(i,j),y.cell(j));
	  }
	}
      }
      if(r.dev==1){
	//op(*this,r,x,y);
      }
    }

    template<typename OP, typename ARR>
    MVprod_cmap(const OP& op, ARR& r, const ARR& x, const ARR& y, const cmap_add& dummy){
      I=x.get_adim(0);
      J=x.get_adim(1);
      assert(r.aasize==I);
      assert(y.aasize==J);
      if(r.dev==0){
	if(J==0) return;
	for(int i=0; i<I; i++){
	  decltype(r.get_cell(0)) t=r.cell(i);
	  for(int j=0; j<J; j++){
	    op.add(t,x.cell(i,j),y.cell(j));
	  }
	}
      }
      if(r.dev==1){
	//op(*this,r,x,y);
      }
    }

    #ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(I);
    }

    __device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(i,i,i);
    }
    #endif 

  };
  */
