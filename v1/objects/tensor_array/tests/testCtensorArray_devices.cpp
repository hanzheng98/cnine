//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.cpp"
#include "CtensorObj_funs.hpp"
#include "CtensorArray.hpp"

#include "CnineSession.hpp"


using namespace cnine;

typedef CscalarObj cscalar;
typedef CtensorObj ctensor;
typedef CtensorArray ctensora;


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  ctensora A=ctensora::zero(dims(2,2),dims(4,4),deviceid::GPU0);
  printl("A",A);

  ctensora B=ctensora::sequential(dims(2,2),dims(4,4),deviceid::GPU0);
  printl("B",B);


  ctensora T=ctensora::sequential(dims(2,2),dims(4,4));
  ctensora T1=T.to(deviceid::GPU0);
  printl("T1",T1);

  ctensora T2=T1.to(deviceid::CPU);
  printl("T2",T2);

}
