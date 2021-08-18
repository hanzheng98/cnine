//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CtensorA_add_times_c_cop
#define _CtensorA_add_times_c_cop

#include "GenericCop.hpp"
#include "CtensorA.hpp"


namespace cnine{

  class CtensorArrayA;


  class CtensorA_add_times_c_cop: public Unary1Cop<CtensorA,CtensorArrayA,complex<float> >{
  public:

    CtensorA_add_times_c_cop(){}
    
    virtual void operator()(CtensorA& r, const CtensorA& y, const complex<float>& c) const{
      r.add(y,c);
    }

    template<typename IMAP>
    void operator()(const IMAP& map, CtensorArrayA& r) const{
      CNINE_UNIMPL();
    }

  };


  class CtensorA_add_div_c_cop: public Unary1Cop<CtensorA,CtensorArrayA,complex<float> >{
  public:

    CtensorA_add_div_c_cop(){}
    
    virtual void operator()(CtensorA& r, const CtensorA& y, const complex<float>& c) const{
      r.add(y,complex<float>(1.0)/c);
    }

    template<typename IMAP>
    void operator()(const IMAP& map, CtensorArrayA& r) const{
      CNINE_UNIMPL();
    }

  };


  class CtensorA_inplace_times_c_cop: public Inplace1Cop<CtensorA,CtensorArrayA,complex<float> >{
  public:

    CtensorA_inplace_times_c_cop(){}
    
    virtual void operator()(CtensorA& r, const complex<float>& c) const{
      r.inplace_times(c);
    }

    template<typename IMAP>
    void operator()(const IMAP& map) const{
      CNINE_UNIMPL();
    }

  };


  class CtensorA_inplace_div_c_cop: public Inplace1Cop<CtensorA,CtensorArrayA,complex<float> >{
  public:

    CtensorA_inplace_div_c_cop(){}
    
    virtual void operator()(CtensorA& r, const complex<float>& c) const{
      r.inplace_times(c);
    }

    template<typename IMAP>
    void operator()(const IMAP& map) const{
      CNINE_UNIMPL();
    }

  };

}

#endif
