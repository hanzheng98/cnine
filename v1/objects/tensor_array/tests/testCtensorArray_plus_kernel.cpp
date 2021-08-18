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

#include "CtensorA_plus_cop.hpp"

#include "CellwiseBinaryCmap.hpp"
#include "InnerCmap.hpp"
#include "OuterCmap.hpp"
#include "MVprodCmap.hpp"
#include "VMprodCmap.hpp"
#include "BroadcastBinaryCmap.hpp"
#include "Convolve2Cmap.hpp"

#include "CnineSession.hpp"


using namespace cnine;

typedef CtensorObj Ctensor;

typedef CtensorA_plus_cop Ctensor_plus;


int main(int argc, char** argv){
  cnine_session genet;
  cout<<endl;

  CtensorArray A(dims(2,2),dims(2,2),-1,[](const Gindex& aix, const Gindex& cix){return aix(0);});
  CtensorArray B(dims(2,2),dims(2,2),-1,[](const Gindex& aix, const Gindex& cix){return complex<float>(0,aix(1));});
  CtensorArray C(dims(2),dims(2,2),-1,[](const Gindex& aix, const Gindex& cix){return aix(0);});
  CtensorArray D(dims(2),dims(2,2),-1,[](const Gindex& aix, const Gindex& cix){return complex<float>(0,aix(0));});
  CtensorArray E(dims(5,5),dims(2,2),-1,[](const Gindex& aix, const Gindex& cix){return complex<float>(aix(0),aix(1));});
  CtensorArray F(dims(2,2),dims(2,2),-1,[](const Gindex& aix, const Gindex& cix){return 0;});

  Ctensor M(dims(2,2),[](const int i, const int j){return 3;});


  printl("cellwise",cellwise<Ctensor_plus>(A,B));
  printl("broadcast",broadcast<Ctensor_plus>(M,A));
  printl("inner",inner<Ctensor_plus>(A,B)); // TODO 
  printl("outer",outer<Ctensor_plus>(C,D));
  printl("MVprod",MVprod<Ctensor_plus>(A,D));
  printl("VMprod",VMprod<Ctensor_plus>(C,B));
  printl("convolve2",convolve2<Ctensor_plus>(E,F));


}
