
//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <torch/torch.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "Gdims.hpp"
#include "Gindex.hpp"

#include "RscalarObj.hpp"
#include "RtensorObj.hpp"
#include "RtensorArray.hpp"

#include "CscalarObj.hpp"
#include "CtensorObj.hpp"
#include "CtensorArray.hpp"

//std::default_random_engine rndGen;
#include "Cnine_base.cpp"

#include "RtensorA_add_plus_cop.hpp"
#include "CtensorA_add_plus_cop.hpp"

#include "CellwiseBinaryCmap.hpp"


//  using namespace cnine;
//namespace py=pybind11;

#include "cmaps_py.cpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  using namespace cnine;
  namespace py=pybind11;


  pybind11::class_<fill_raw>(m,"fill_raw")
    .def(pybind11::init<>());
  pybind11::class_<fill_zero>(m,"fill_zero")
    .def(pybind11::init<>());
  pybind11::class_<fill_ones>(m,"fill_ones")
    .def(pybind11::init<>());
  pybind11::class_<fill_identity>(m,"fill_identity")
    .def(pybind11::init<>());
  pybind11::class_<fill_gaussian>(m,"fill_gaussian")
    .def(pybind11::init<>());
  pybind11::class_<fill_sequential>(m,"fill_sequential")
    .def(pybind11::init<>());


  pybind11::class_<Gdims>(m,"gdims")
    .def(pybind11::init<vector<int> >())
    .def(pybind11::init<vector<int> >(),"Initialize a Gdims  object from a list of integers.")

    .def("__len__",&Gdims::k)
    .def("__getitem__",&Gdims::operator())
    .def("__setitem__",&Gdims::set)

    .def("str",&Gdims::str)
    .def("__str__",&Gdims::str)
    .def("__repr__",&Gdims::repr);



  pybind11::class_<Gindex>(m,"gindex")
    .def(pybind11::init<vector<int> >())

    .def("__len__",&Gindex::size)
    .def("__getitem__",[](const Gindex& obj, const int i){return obj(i);})
    .def("__setitem__",&Gindex::set)

    .def("str",&Gindex::str)
    .def("__str__",&Gindex::str)
    .def("__repr__",&Gindex::repr);



  pybind11::class_<RscalarObj>(m,"rscalar")
    .def(pybind11::init<>())
    .def("str",&RscalarObj::str,py::arg("indent")="")
    .def("__str__",&RscalarObj::str,py::arg("indent")="");



  pybind11::class_<CscalarObj>(m,"cscalar")
    .def(pybind11::init<>())
    .def("str",&CscalarObj::str,py::arg("indent")="")
    .def("__str__",&CscalarObj::str,py::arg("indent")="");
  

#include "rtensor_py.cpp"
#include "ctensor_py.cpp"
  
#include "rtensorarr_py.cpp"
#include "ctensorarr_py.cpp"

  //#include "cmaps_py.cpp"
    
  def_inner<RtensorA_add_plus_cop,RtensorArray>(m);
  def_outer<RtensorA_add_plus_cop,RtensorArray>(m);
  def_cellwise<RtensorA_add_plus_cop,RtensorArray>(m);
  def_mprod<RtensorA_add_plus_cop,RtensorArray>(m);
  def_convolve1<RtensorA_add_plus_cop,RtensorArray>(m);
  def_convolve2<RtensorA_add_plus_cop,RtensorArray>(m);

  def_inner<CtensorA_add_plus_cop,CtensorArray>(m);
  def_outer<CtensorA_add_plus_cop,CtensorArray>(m);
  def_cellwise<CtensorA_add_plus_cop,CtensorArray>(m);
  def_mprod<CtensorA_add_plus_cop,CtensorArray>(m);
  def_convolve1<CtensorA_add_plus_cop,CtensorArray>(m);
  def_convolve2<CtensorA_add_plus_cop,CtensorArray>(m);

}


/*
namespace pybind11{
  namespace detail{ 

    template <> struct type_caster<cnine::RtensorObj>{

      static handle cast(cnine::RtensorObj obj){
	vector<int64_t> v(2); for(int i=0; i<2; i++) v[i]=4;
	at::Tensor R(at::zeros(v,torch::CPU(at::kFloat))); 
	return R;
      }
    };

  }

}
*/
