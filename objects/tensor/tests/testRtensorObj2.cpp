//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.cpp"
#include "RtensorObj_funs.hpp"
#include "CnineSession.hpp" 

using namespace cnine;

typedef RscalarObj rscalar; 
typedef RtensorObj rtensor; 


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  rtensor A(dims(4,4),fill::sequential);


  // Getting/setting to/from RscalarObj

  rscalar a=A.get(2,3);
  cout<<a<<endl<<endl;

  rscalar b(19.0);
  A.set(1,1,b);
  cout<<A<<endl;


  // Getting/setting to/from float

  float aval=A.get_value(2,3);
  cout<<aval<<endl<<endl;

  float bval(17);
  A.set_value(1,1,bval);
  cout<<A<<endl;


  // Getting/setting with expression template

  a=A(3,3);
  A(0,2)=b;
  cout<<a<<endl<<endl;
  cout<<A<<endl;


  const rtensor constA=A;
  a=constA(3,3);
  aval=constA(3,3);

  A(0,0)=A(3,3);

}
