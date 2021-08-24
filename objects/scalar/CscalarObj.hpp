//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineCscalarObj
#define _CnineCscalarObj

#include "CscalarA.hpp"
#include "RscalarObj.hpp"
#include "ExprTemplates.hpp"

namespace cnine{

  class CscalarObj;

  class CscalarObjExpr{
  public:
    virtual operator CscalarObj() const=0;
  };

  class CscalarObj: public CNINE_CSCALAR_IMPL{
  public:

    using CNINE_CSCALAR_IMPL::CNINE_CSCALAR_IMPL; 


  public: // ---- Named constructors -------------------------------------------------------------------------


    static CscalarObj zero(){
      return CscalarObj(fill::zero);}
    static CscalarObj zero(const int nbd=-1){
      return CscalarObj(nbd,fill::zero);}

    static CscalarObj gaussian(){
      return CscalarObj(fill::gaussian);}
    static CscalarObj gaussian(const int nbd=-1){
      return CscalarObj(nbd,fill::gaussian);}


  public: // ---- Copying ------------------------------------------------------------------------------------


    CscalarObj(const CscalarObj& x):
      CNINE_CSCALAR_IMPL(x){}

    CscalarObj(CscalarObj&& x):
      CNINE_CSCALAR_IMPL(std::move(x)){}

    CscalarObj& operator=(const CscalarObj& x){
      CNINE_CSCALAR_IMPL::operator=(x);
      return *this;
    }

    CscalarObj& operator=(CscalarObj&& x){
      CNINE_CSCALAR_IMPL::operator=(std::move(x));
      return *this;
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    CscalarObj(const CNINE_CSCALAR_IMPL& x):
      CNINE_CSCALAR_IMPL(x){};

    CscalarObj(const Conjugate<CscalarObj>& x):
      CNINE_CSCALAR_IMPL(x.obj.conj()){}


  public: // ---- Access -------------------------------------------------------------------------------------


    RscalarObj real() const{
      return CNINE_CSCALAR_IMPL::real();
    }

    RscalarObj imag() const{
      return CNINE_CSCALAR_IMPL::imag();
    }


  public: // ---- In-place operations ------------------------------------------------------------------------


    void clear(){
      set_zero();
    }


  public: // ---- Non-inplace operations ---------------------------------------------------------------------


    CscalarObj conj() const{
      return CNINE_CSCALAR_IMPL::conj();
    }

    CscalarObj plus(const CscalarObj& x){
      CscalarObj R(*this);
      R.add(x);
      return R;
    }

    CscalarObj apply(std::function<complex<float>(const complex<float>)> fn){
      return CscalarObj(CNINE_CSCALAR_IMPL::apply(fn));
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


 

  public: // ---- Into operations -----------------------------------------------------------------------------


    void inp_into(CscalarObj& R, const CscalarObj& y) const{
      R.add_prodc(*this,y);
    }

    void norm2_into(CscalarObj& R) const{
      R.add_prodc(*this,*this);
    }


  public: // ---- In-place operators --------------------------------------------------------------------------


    CscalarObj& operator+=(const CscalarObj& y){
      add(y);
      return *this;
    }

    CscalarObj& operator-=(const CscalarObj& y){
      subtract(y);
      return *this;
    }


  // ---- Binary operators -----------------------------------------------------------------------------------


    CscalarObj operator+(const CscalarObj& y) const{
      CscalarObj R(*this);
      R.add(y);
      return R;
    }

    CscalarObj operator-(const CscalarObj& y) const{
      CscalarObj R(*this);
      R.subtract(y);
      return R;
    }

    CscalarObj operator*(const CscalarObj& y) const{
      CscalarObj R(fill::zero);
      R.add_prod(*this,y);
      return R;
    }

    CscalarObj operator/(const CscalarObj& y) const{
      CscalarObj R(fill::zero);
      R.add_div(*this,y);
      return R;
    }

    CscalarObj operator*(const float c) const{
      CscalarObj R(fill::zero);
      R.add(*this,c);
      return R;
    }

    CscalarObj operator*(const complex<float> c) const{
      CscalarObj R(fill::zero);
      R.add(*this,c);
      return R;
    }

  public: // ---- I/O ----------------------------------------------------------------------------------------

    
    string classname() const{
      return "GEnet::CscalarObj";
    }

    string describe() const{
      return "Rscalar";
    } 

    friend ostream& operator<<(ostream& stream, const CscalarObj& x){
      stream<<x.str(); return stream;}

  };

}

#endif
