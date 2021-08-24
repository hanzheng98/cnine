// This file is part of cnine, a lightweight C++ tensor library. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.hpp"

#ifdef _WITH_CENGINE
#include "Cengine_base.cpp"
#endif 

std::default_random_engine rndGen;

mutex cnine::CoutLock::mx;


#ifdef _WITH_CENGINE
#include "RscalarM.hpp"
#include "CscalarM.hpp"
#include "CtensorM.hpp"

/*
int Cengine::ctensor_add_op::_batcher_id=0; 
 int Cengine::ctensor_add_op::_rbatcher_id=0; 

 int Cengine::ctensor_add_prod_c_A_op::_rbatcher_id=0; 

 int Cengine::ctensor_add_inp_op::_rbatcher_id=0; 

*/

namespace cnine{

 template<> int ctensor_add_Mprod_op<0,0>::_batcher_id=0; 
 template<> int ctensor_add_Mprod_op<0,1>::_batcher_id=0; 
 template<> int ctensor_add_Mprod_op<0,2>::_batcher_id=0; 
 template<> int ctensor_add_Mprod_op<0,3>::_batcher_id=0; 

 template<> int ctensor_add_Mprod_op<1,0>::_batcher_id=0; 
 template<> int ctensor_add_Mprod_op<1,1>::_batcher_id=0; 
 template<> int ctensor_add_Mprod_op<1,2>::_batcher_id=0; 
 template<> int ctensor_add_Mprod_op<1,3>::_batcher_id=0; 

 template<> int ctensor_add_Mprod_op<2,0>::_batcher_id=0; 
 template<> int ctensor_add_Mprod_op<2,1>::_batcher_id=0; 
 template<> int ctensor_add_Mprod_op<2,2>::_batcher_id=0; 
 template<> int ctensor_add_Mprod_op<2,3>::_batcher_id=0; 

 template<> int ctensor_add_Mprod_op<0,0>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<0,1>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<0,2>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<0,3>::_rbatcher_id=0; 

 template<> int ctensor_add_Mprod_op<1,0>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<1,1>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<1,2>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<1,3>::_rbatcher_id=0; 

 template<> int ctensor_add_Mprod_op<2,0>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<2,1>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<2,2>::_rbatcher_id=0; 
 template<> int ctensor_add_Mprod_op<2,3>::_rbatcher_id=0; 

}

#endif
