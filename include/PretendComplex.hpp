// This file is part of cnine, a lightweight C++ tensor library. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CninePretendComplex
#define _CninePretendComplex

#include "Cnine_base.hpp"

namespace cnine{

  template<typename TYPE>
  class pretend_complex{
  public:
    TYPE* realp;
    TYPE* imagp;
    pretend_complex(TYPE* _realp, TYPE* _imagp): 
      realp(_realp), imagp(_imagp){}
  public:
    operator complex<TYPE>() const{
      return complex<TYPE>(*realp,*imagp);
    }
    complex<TYPE> conj() const{
      return complex<TYPE>(*realp,-*imagp);
    }
    pretend_complex& operator=(const complex<TYPE> x){
      *realp=std::real(x);
      *imagp=std::imag(x);
      return *this;
    }
  public:
    pretend_complex& operator+=(const complex<TYPE> x){
      *realp+=std::real(x);
      *imagp+=std::imag(x);
      return *this;
    }
    pretend_complex& operator-=(const complex<TYPE> x){
      *realp-=std::real(x);
      *imagp-=std::imag(x);
      return *this;
    }
    pretend_complex& operator*=(const complex<TYPE> x){
      *realp=std::real(x)*(*realp)-std::imag(x)*(*imagp);
      *imagp=std::real(x)*(*imagp)+std::imag(x)*(*realp);
      return *this;
    }
    pretend_complex& operator/=(const complex<TYPE> x){
      complex<float> t=complex<float>(*realp,*imagp)/x;
      *realp=std::real(t);
      *imagp=std::imag(t);
      return *this;
    }
  public:
    complex<float> operator*(const pretend_complex& y) const{
      return complex<TYPE>(*this)*complex<TYPE>(y);
    }
  public:
    friend ostream& operator<<(ostream& stream, const pretend_complex& x){
      stream<<complex<TYPE>(x); return stream;}
  };

}

#endif 

