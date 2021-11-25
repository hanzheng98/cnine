pybind11::class_<CtensorObj>(m,"ctensor")

  .def(pybind11::init<const Gdims&>())
  .def(pybind11::init<const Gdims&, const fill_raw&>())
  .def(pybind11::init<const Gdims&, const fill_zero&>())
  .def(pybind11::init<const Gdims&, const fill_ones&>())
  .def(pybind11::init<const Gdims&, const fill_identity&>())
  .def(pybind11::init<const Gdims&, const fill_gaussian&>())
  .def(pybind11::init<const Gdims&, const fill_sequential&>())

  .def_static("raw",static_cast<CtensorObj (*)(const Gdims&, const int, const int)>(&CtensorObj::raw))
  .def_static("raw",[](const Gdims& dims, const int dev){return CtensorObj::raw(dims,-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("raw",[](const vector<int>& v, const int dev){return CtensorObj::raw(Gdims(v),-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("raw",[](const int i0){return CtensorObj::raw(Gdims({i0}));})
  .def_static("raw",[](const int i0, const int i1){return CtensorObj::raw(Gdims({i0,i1}));})
  .def_static("raw",[](const int i0, const int i1, const int i2){return CtensorObj::raw(Gdims({i0,i1,i2}));})

  .def_static("zero",static_cast<CtensorObj (*)(const Gdims&, const int, const int)>(&CtensorObj::zero))
  .def_static("zero",[](const Gdims& dims, const int dev){return CtensorObj::zero(dims,-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("zero",[](const vector<int>& v, const int dev){return CtensorObj::zero(Gdims(v),-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("zero",[](const int i0){return CtensorObj::zero(Gdims({i0}));})
  .def_static("zero",[](const int i0, const int i1){return CtensorObj::zero(Gdims({i0,i1}));})
  .def_static("zero",[](const int i0, const int i1, const int i2){return CtensorObj::zero(Gdims({i0,i1,i2}));})

  .def_static("ones",static_cast<CtensorObj (*)(const Gdims&,const int, const int)>(&CtensorObj::ones))
  .def_static("ones",[](const Gdims& dims, const int dev){return CtensorObj::ones(dims,-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("ones",[](const vector<int>& v, const int dev){return CtensorObj::ones(Gdims(v),-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("ones",[](const int i0){return CtensorObj::ones(Gdims({i0}));})
  .def_static("ones",[](const int i0, const int i1){return CtensorObj::ones(Gdims({i0,i1}));})
  .def_static("ones",[](const int i0, const int i1, const int i2){return CtensorObj::ones(Gdims({i0,i1,i2}));})

  .def_static("identity",static_cast<CtensorObj (*)(const Gdims&,const int, const int)>(&CtensorObj::identity))
  .def_static("identity",[](const Gdims& dims, const int dev){return CtensorObj::identity(dims,-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("identity",[](const vector<int>& v, const int dev){return CtensorObj::identity(Gdims(v),-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("identity",[](const int i0){return CtensorObj::identity(Gdims({i0}));})
  .def_static("identity",[](const int i0, const int i1){return CtensorObj::identity(Gdims({i0,i1}));})
  .def_static("identity",[](const int i0, const int i1, const int i2){return CtensorObj::identity(Gdims({i0,i1,i2}));})

  .def_static("gaussian",static_cast<CtensorObj (*)(const Gdims&,const int, const int)>(&CtensorObj::gaussian))
  .def_static("gaussian",[](const Gdims& dims, const int dev){return CtensorObj::gaussian(dims,-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("gaussian",[](const vector<int>& v, const int dev){return CtensorObj::gaussian(Gdims(v),-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("gaussian",[](const int i0){return CtensorObj::gaussian(Gdims({i0}));})
  .def_static("gaussian",[](const int i0, const int i1){return CtensorObj::gaussian(Gdims({i0,i1}));})
  .def_static("gaussian",[](const int i0, const int i1, const int i2){return CtensorObj::gaussian(Gdims({i0,i1,i2}));})

  .def_static("sequential",static_cast<CtensorObj (*)(const Gdims&,const int, const int)>(&CtensorObj::sequential))
  .def_static("sequential",[](const Gdims& dims, const int dev){return CtensorObj::sequential(dims,-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("sequential",[](const vector<int>& v, const int dev){return CtensorObj::sequential(Gdims(v),-1,dev);},
    py::arg("dims"),py::arg("device")=0)
  .def_static("sequential",[](const int i0){return CtensorObj::sequential(Gdims({i0}));})
  .def_static("sequential",[](const int i0, const int i1){return CtensorObj::sequential(Gdims({i0,i1}));})
  .def_static("sequential",[](const int i0, const int i1, const int i2){return CtensorObj::sequential(Gdims({i0,i1,i2}));})

  .def("copy",[](const RtensorObj& obj){return obj;})

  .def("get_k",&CtensorObj::get_k)
  .def("getk",&CtensorObj::get_k)
  .def("get_dims",&CtensorObj::get_dims)
  .def("get_dim",&CtensorObj::get_dim)

  .def("get",static_cast<CscalarObj(CtensorObj::*)(const Gindex& )const>(&CtensorObj::get))
  .def("get",static_cast<CscalarObj(CtensorObj::*)(const int)const>(&CtensorObj::get))
  .def("get",static_cast<CscalarObj(CtensorObj::*)(const int, const int)const>(&CtensorObj::get))
  .def("get",static_cast<CscalarObj(CtensorObj::*)(const int, const int, const int)const>(&CtensorObj::get))

  .def("set",static_cast<CtensorObj&(CtensorObj::*)(const Gindex&, const CscalarObj&)>(&CtensorObj::set))
  .def("set",static_cast<CtensorObj&(CtensorObj::*)(const int, const CscalarObj&)>(&CtensorObj::set))
  .def("set",static_cast<CtensorObj&(CtensorObj::*)(const int, const int, const CscalarObj&)>(&CtensorObj::set))
  .def("set",static_cast<CtensorObj&(CtensorObj::*)(const int, const int, const int, const CscalarObj&)>(&CtensorObj::set))

  .def("get_value",static_cast<complex<float>(CtensorObj::*)(const Gindex& )const>(&CtensorObj::get_value))
  .def("get_value",static_cast<complex<float>(CtensorObj::*)(const int)const>(&CtensorObj::get_value))
  .def("get_value",static_cast<complex<float>(CtensorObj::*)(const int, const int)const>(&CtensorObj::get_value))
  .def("get_value",static_cast<complex<float>(CtensorObj::*)(const int, const int, const int)const>(&CtensorObj::get_value))

  .def("__call__",static_cast<complex<float>(CtensorObj::*)(const Gindex& )const>(&CtensorObj::get_value))
  .def("__call__",[](const CtensorObj& obj, const vector<int> v){return obj.get_value(Gindex(v));})

  .def("__getitem__",static_cast<complex<float>(CtensorObj::*)(const Gindex& )const>(&CtensorObj::get_value))
  .def("__getitem__",[](const CtensorObj& obj, const vector<int> v){return obj.get_value(Gindex(v));})

  .def("set_value",static_cast<CtensorObj&(CtensorObj::*)(const Gindex&, const complex<float>)>(&CtensorObj::set_value))
  .def("set_value",static_cast<CtensorObj&(CtensorObj::*)(const int, const complex<float>)>(&CtensorObj::set_value))
  .def("set_value",static_cast<CtensorObj&(CtensorObj::*)(const int, const int, const complex<float>)>(&CtensorObj::set_value))
  .def("set_value",static_cast<CtensorObj&(CtensorObj::*)(const int, const int, const int, const complex<float>)>(&CtensorObj::set_value))

  .def("__setitem__",[](CtensorObj& obj, const Gindex& ix, const complex<float> x){obj.set_value(ix,x);})
  .def("__setitem__",[](CtensorObj& obj, const vector<int> v, const complex<float> x){obj.set_value(Gindex(v),x);})

  .def("__add__",[](const CtensorObj& x, const CtensorObj& y){return x.plus(y);})
  .def("__subtract__",[](const CtensorObj& x, const CtensorObj& y){return x-y;})
  .def("__mul__",[](const CtensorObj& x, const complex<float> c){
      CtensorObj R(x.get_dims(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;})
  .def("__mul__",[](const complex<float> c, const CtensorObj& x){
      CtensorObj R(x.get_dims(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;})
  .def("__mul__",[](const CtensorObj& x, const CtensorObj& y){
      CtensorObj R(x.get_dims().Mprod(y.get_dims()),x.get_nbu(),fill::zero,x.get_dev());
      R.add_mprod(x,y);
      return R;})

  .def("slice",&CtensorObj::slice)
  .def("chunk",&CtensorObj::chunk)

  .def("reshape",&CtensorObj::reshape)
  .def("reshape",[](CtensorObj& obj, const vector<int>& v){obj.reshape(Gindex(v));})

  .def("transp",&CtensorObj::transp)
  .def("conj",&CtensorObj::conj)
  .def("herm",&CtensorObj::herm)

//.def("__getitem__",static_cast<CscalarObj(CtensorObj::*)(const int, const int)const>(&CtensorObj::get))

  .def("str",&CtensorObj::str,py::arg("indent")="")
  .def("__str__",&CtensorObj::str,py::arg("indent")="");



m.def("inp",[](const CtensorObj& x, const CtensorObj& y){return x.inp(y);});
//m.def("odot",[](const CtensorObj& x, const CtensorObj& y){return x.odot(y);});
m.def("norm2",[](const CtensorObj& x){return x.norm2();});


m.def("ReLU",[](const CtensorObj& x){
    CtensorObj R(x.get_dims(),x.get_nbu(),fill::zero);
    R.add_ReLU(x);
    return R;});
m.def("ReLU",[](const CtensorObj& x, const float c){
    CtensorObj R(x.get_dims(),x.get_nbu(),fill::zero);
    R.add_ReLU(x,c);
    return R;});

