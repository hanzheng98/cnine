
//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//#include <torch/torch.h>
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

    .def("__str__",&Gdims::str);



  pybind11::class_<Gindex>(m,"gindex")
    .def(pybind11::init<vector<int> >())
    .def("str",&Gindex::str)
    .def("__str__",&Gindex::str);



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


}
