// This file is part of cnine, a lightweight C++ tensor library. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineOperationTemplates
#define _CnineOperationTemplates


namespace cnine{

  template<typename TENSORTYPE, typename = typename std::enable_if<std::is_base_of<CtensorA, TENSORTYPE>::value, TENSORTYPE>::type>
  TENSORTYPE operator*(complex<float> c, const TENSORTYPE& x){
    return x*c;
  }

  template<typename TENSORTYPE, typename = typename std::enable_if<std::is_base_of<CtensorA, TENSORTYPE>::value, TENSORTYPE>::type>
  TENSORTYPE operator*(CscalarObj& c, const TENSORTYPE& x){
    return x*c;
  }

  //template<typename TENSORTYPE, typename = typename std::enable_if<std::is_base_of<CtensorA, TENSORTYPE>::value, TENSORTYPE>::type>
  //TENSORTYPE operator*(CscalarObj& c, const TENSORTYPE& x){
  //return x*c;
  //}

}

#endif 
