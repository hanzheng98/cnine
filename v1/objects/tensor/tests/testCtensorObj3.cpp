//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.cpp"
#include "CtensorObj_funs.hpp"
#include "CnineSession.hpp"

using namespace cnine;

typedef CscalarObj cscalar; 
typedef CtensorObj ctensor; 


int main(int argc, char** argv){

  cnine_session session;

  cout<<endl;

  Gtensor<complex<float> > Gin(dims(4,4),fill::sequential);

  ctensor A(Gin);
  ctensor B(Gin);
  cout<<A<<endl;

  ctensor C=A*B;


  Gtensor<complex<float> > Gout=C.gtensor();
  cout<<Gout<<endl;
 
}
