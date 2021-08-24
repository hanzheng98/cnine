//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.cpp"
#include "RscalarObj_funs.hpp"

using namespace cnine;

typedef RscalarObj rscalar;


int main(int argc, char** argv){

  rscalar A=2.0;
  rscalar B=3.0;
  rscalar C=-1.0;

  cout<<endl; 
  cout<<" A+B = "<<A+B<<endl;
  cout<<" A-B = "<<A-B<<endl;
  cout<<" A*B = "<<A*B<<endl;
  cout<<" A/B = "<<A/B<<endl;
  cout<<endl; 

  cout<<" abs(A) = "<<abs(A)<<endl;
  cout<<" inp(A,B) = "<<inp(A,B)<<endl;
  cout<<" ReLU(C) = "<<ReLU(C,0.1)<<endl;

  cout<<" fn(A) = "<<A.apply([](const float x){return x*x+3;})<<endl; 

  cout<<endl; 

}
