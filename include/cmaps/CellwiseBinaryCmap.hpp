//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CellwiseBinaryCmap
#define _CellwiseBinaryCmap

#include "Cmaps2.hpp"


namespace cnine{

  class CellwiseBinaryCmap: public Direct_cmap{ //public BinaryDirectCmap{ public Cmap_base,
  public:
    
    int I;

    template<typename OP, typename ARR>
    CellwiseBinaryCmap(const OP& op, ARR& r, const ARR& x, const ARR& y, const int add_flag=0){
      I=r.aasize;
      assert(x.aasize==I);
      assert(y.aasize==I);
      if(r.dev==0){
	for(int i=0; i<I; i++){
	  decltype(x.get_cell(0)) t=r.cell(i);
	  op.apply(t,x.cell(i),y.cell(i),add_flag);
	  //if(add_flag==0) op.apply(t,x.cell(i),y.cell(i),0);
	  //else op.apply(t,x.cell(i),y.cell(i),1);
	}
      }
      if(r.dev==1){
	op.apply(*this,r,x,y,add_flag);
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


  // ---- Templates ------------------------------------------------------------------------------------------


  template<typename OP, typename ARR>
  ARR cellwise(const ARR& x, const ARR& y){
    ARR r(x,x.adims,fill::raw);
    CellwiseBinaryCmap(OP(),r,x,y);
    return r;
  }

  template<typename OP, typename ARR, typename ARG0>
  ARR cellwise(const ARR& x, const ARR& y, const ARG0& arg0){
    ARR r(x,x.adims,fill::raw);
    OP op(arg0);
    //ARR r=op.init(x.cell(0),y.cell(0),x.get_adims(),fill::raw);
    CellwiseBinaryCmap(op,r,x,y);
    return r;
  }

  template<typename OP, typename ARR, typename ARG0, typename ARG1>
  ARR cellwise(const ARR& x, const ARR& y, const ARG0& arg0, const ARG1& arg1){
    ARR r(x,x.adims,fill::raw);
    OP op(arg0,arg1);
    //ARR r=op.init(x.cell(0),y.cell(0),x.get_adims(),fill::raw);
    CellwiseBinaryCmap(op,r,x,y);
    return r;
  }



  template<typename OP, typename ARR>
  void add_cellwise(ARR& r, const ARR& x, const ARR& y){
    CellwiseBinaryCmap(OP(),r,x,y,1);
  }

  template<typename OP, typename ARR, typename ARG0>
  void add_cellwise(ARR& r, const ARR& x, const ARR& y, const ARG0& arg0){
    CellwiseBinaryCmap(OP(arg0),r,x,y,1);
  }


}

#endif 


