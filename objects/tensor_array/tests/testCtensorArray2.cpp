//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.cpp"
#include "CtensorObj_funs.hpp"
#include "CtensorArray_funs.hpp"
#include "CtensorA_add_Mprod_cop.hpp"

#include "CnineSession.hpp"


using namespace cnine;

typedef CscalarObj cscalar;
typedef CtensorObj ctensor;
typedef CtensorArray ctensor_array;


int main(int argc, char** argv){

  cnine_session genet;

  cout<<endl;

  ctensor_array A=ctensor_array::sequential(dims(2,2),dims(4,4));
  cout<<"A="<<endl<<A<<endl<<endl;

  ctensor_array B=ctensor_array::sequential(dims(2,2),dims(4,4));
  cout<<"B="<<endl<<B<<endl<<endl;

  //A.map_binary(CtensorA_add_Mprod_cop(),A,B);
  //cout<<"A="<<endl<<A<<endl<<endl;

  //ctensor_array C=A*B;
  //cout<<"C="<<endl<<C<<endl<<endl;

  printl("A*B",A*B);

}
