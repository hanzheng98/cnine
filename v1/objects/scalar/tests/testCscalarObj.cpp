//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.cpp"
#include "CscalarObj_funs.hpp"

using namespace cnine;

typedef CscalarObj cscalar;


int main(int argc, char** argv){
  cout<<endl;

  cscalar A=2.0;
  cscalar B=3.0;
  cscalar C=complex<float>(1.0,-1.0);

  print(" A",A);
  print(" B",B);
  print(" C",C);

  cout<<endl; 
  cout<<" A+B = "<<A+B<<endl;
  cout<<" A-B = "<<A-B<<endl;
  cout<<" A*B = "<<A*B<<endl;
  cout<<" A/B = "<<A/B<<endl;
  cout<<endl; 

  cout<<" real(C) = "<<real(C)<<endl;
  cout<<" imag(C) = "<<imag(C)<<endl;
  cout<<" conj(A) = "<<conj(A)<<endl<<endl;

  cout<<" abs(A) = "<<abs(A)<<endl;
  cout<<" inp(A,B) = "<<inp(A,B)<<endl;
  cout<<" ReLU(C) = "<<ReLU(C,0.1)<<endl<<endl;

  cout<<"  fn(C) = "<<C.apply([](const complex<float> x){return x*x+complex<float>(3.0);})<<endl; 

}

