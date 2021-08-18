// This file is part of cnine, a lightweight C++ tensor library. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GenericOp
#define _GenericOp

#include "CtensorArrayA.hpp"

namespace cnine{

  class CtensorArrayA;

  //inline CtensorArrayA* arrayof(const CtensorA&){
  //return new CtensorArrayA(Gdims({}),Gdims({}));
  //}


  template<typename OBJ>
  class InplaceOp{
  public:
    virtual void operator()(OBJ& R) const=0;
  };


  template<typename OBJ, typename OBJ0>
  class UnaryOp{
  public:
    virtual void operator()(OBJ& R, const OBJ0& x0) const=0;
  };


  //template<typename OBJ>
  //class ARRAY;

  //template<>
  //typedef CtensorArrayA ARRAY<CtensorA>;

  //using ARRAY<CtensorA> =CtensorArrayA;

  template<typename OBJ, typename OBJ0, typename OBJ1>
  class BinaryOp{
  public:

    //typedef decltype(*OBJ().array_type()) OBJARR;
    //typedef decltype(*OBJ0().const_array_type()) COBJARR0;
    //typedef decltype(*OBJ1().const_array_type()) COBJARR1;

    virtual void operator()(OBJ& R, const OBJ0& x0, const OBJ1& x1) const=0;

    virtual void execG(CtensorArrayA& r, const CtensorArrayA& x, const CtensorArrayA& y,
      const int rn, const int xn, const int yn, const int rs, const int rx, const int ry) const{
      CNINE_UNIMPL();
    }

  };


}

#endif


    //virtual void map(cnine::CtensorArrayA& R, const cnine::CtensorArrayA& x0, const cnine::CtensorArrayA& x1) const{
