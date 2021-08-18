//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineRtensorFunctions
#define _CnineRtensorFunctions

#include "Cnine_base.hpp"
#include "ExprTemplates.hpp"
#include "RscalarObj.hpp"
#include "RtensorObj.hpp"
//#include "ArrayOf.hpp"


namespace cnine{



  // -------------------------------------------------------------------------------------------------------- 


  RscalarObj norm2(const RtensorObj& x){
    RscalarObj r(x.get_nbu(),fill::zero);
    x.add_norm2_into(r);
    return r;
  }

  RscalarObj inp(const RtensorObj& x, const RtensorObj& y){
    RscalarObj r(x.get_nbu(),fill::zero);
    x.add_inp_into(r,y);
    return r;
  }

  RtensorObj ReLU(const RtensorObj& x, const float c=0){
    RtensorObj R(x.get_dims(),x.get_nbu(),fill::zero);
    R.add_ReLU(x,c);
    return R;
  }



  // ---- Scalar Multiplication -----------------------------------------------------------------------------

  
  class Rtensor_Rscalar_prod_expr: public Printable{
  public:
    const RscalarObj& c;
    const RtensorObj& x;
    Rtensor_Rscalar_prod_expr(const RscalarObj& _c, const RtensorObj& _x):
      c(_c), x(_x){}
    operator RtensorObj() const{
      RtensorObj R(x.get_dims(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;
    }
    string str(const string indent="") const{
      return RtensorObj(*this).str(indent);
    }
  };

  
  Rtensor_Rscalar_prod_expr operator*(const RscalarObj& c, const RtensorObj& x){
    return Rtensor_Rscalar_prod_expr(c,x); 
  }

  Rtensor_Rscalar_prod_expr operator*(const RtensorObj& x, const RscalarObj& c){
    return Rtensor_Rscalar_prod_expr(c,x); 
  }

  RtensorObj& operator+=(RtensorObj& R, const Rtensor_Rscalar_prod_expr& expr){
    R.add(expr.x,expr.c);
    return R;
  }


  // ---- Matrix multiplication -----------------------------------------------------------------------------


  class Rtensor_Mprod_expr: public Printable{
  public:
    const RtensorObj& x;
    const RtensorObj& y;
    Rtensor_Mprod_expr(const RtensorObj& _x, const RtensorObj& _y):
      x(_x), y(_y){}
    operator RtensorObj() const{
      RtensorObj R(x.get_dims().Mprod(y.get_dims()),x.get_nbu(),fill::zero,x.get_dev());
      R.add_mprod(x,y);
      return R;
    }
    string str(const string indent="") const{
      return RtensorObj(*this).str(indent);
    }
  };
  
  Rtensor_Mprod_expr operator*(const RtensorObj& x, const RtensorObj& y){
    return Rtensor_Mprod_expr(x,y); 
  }

  RtensorObj& operator+=(RtensorObj& R, const Rtensor_Mprod_expr& expr){
    R.add_mprod(expr.x,expr.y);
    return R;
  }


  RtensorObj operator*(const Transpose<RtensorObj>& x, const RtensorObj& y){
    int I=x.obj.get_dims().combined(1,x.obj.get_dims().k());
    int J=y.get_dims().combined(1,y.get_dims().k());
    RtensorObj R(dims(I,J),fill::zero);
    R.add_mprod_TA(x.obj,y);
    return R;
  }

  
  // ---- Other functions ----------------------------------------------------------------------------------

  /*
    template<typename... Args>
    RtensorObj stack(const int ix, const RtensorObj& x, Args...args){
    return RtensorObj(fill::stack,ix,x,args...);
    }
    
    template<typename... Args>
    RtensorObj cat(const int ix, const RtensorObj& x, Args...args){
    return RtensorObj(fill::cat,ix,x,args...);
    }
  */




}

#endif 
  //Broadcast<RtensorObj> broadcast(const RtensorObj& x){
  //return Broadcast<RtensorObj>(x);
  //}

  //Scatter<RtensorObj> scatter(const RtensorObj& x){
  //return Scatter<RtensorObj>(x);
  //}

  /*
  Transpose<RtensorObj> transp(const RtensorObj& x){
    return Transpose<RtensorObj>(x);
  }

  Conjugate<RtensorObj> conj(const RtensorObj& x){
    return Conjugate<RtensorObj>(x);
  }

  Hermitian<RtensorObj> herm(const RtensorObj& x){
    return Hermitian<RtensorObj>(x);
  }
  */
