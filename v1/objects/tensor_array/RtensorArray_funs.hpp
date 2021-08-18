//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineRtensorArrayFunctions
#define _CnineRtensorArrayFunctions

#include "Cnine_base.hpp"
#include "ExprTemplates.hpp"
#include "CscalarObj.hpp"
#include "RtensorObj.hpp"
#include "RtensorArray.hpp"


namespace cnine{


  // ---- Broadcast operations ------------------------------------------------------------------------------ 


  //RtensorArray broadcast(const Gdims& adims, const RtensorObj& x){
  //return RtensorArray(adims,x);
  //}

  
  RtensorArray operator*(const RtensorArray& x, const Broadcast<RtensorObj>& _y){
    const RtensorObj& y=_y.obj;
    RtensorArray R(x.adims,x.cdims.Mprod(y.dims),x.nbu,fill::zero,x.dev);
    R.broadcast_add_mprod(x,y);
    return R;
  }




}

#endif 
