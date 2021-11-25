pybind11::class_<CtensorArray>(m,"ctensor_arr")

  .def(pybind11::init<const Gdims&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_raw&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_zero&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_ones&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_identity&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_gaussian&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_sequential&>())

  .def_static("raw",static_cast<CtensorArray (*)(const Gdims&, const Gdims&, const int, const int)>(&CtensorArray::raw))
  .def_static("raw",[](const Gdims& adims, const Gdims& dims, const int dev){
      return CtensorArray::raw(dims,-1,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("raw",[](const vector<int>& av, const vector<int>& v, const int dev){
      return CtensorArray::raw(Gdims(av),Gdims(v),-1,dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)

  .def_static("zero",static_cast<CtensorArray (*)(const Gdims&, const Gdims&, const int, const int)>(&CtensorArray::zero))
  .def_static("zero",[](const Gdims& adims, const Gdims& dims, const int dev){
      return CtensorArray::zero(dims,-1,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("zero",[](const vector<int>& av, const vector<int>& v, const int dev){
      return CtensorArray::zero(Gdims(av),Gdims(v),-1,dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)

  .def_static("ones",static_cast<CtensorArray (*)(const Gdims&, const Gdims&, const int, const int)>(&CtensorArray::ones))
  .def_static("ones",[](const Gdims& adims, const Gdims& dims, const int dev){
      return CtensorArray::ones(dims,-1,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("ones",[](const vector<int>& av, const vector<int>& v, const int dev){
      return CtensorArray::ones(Gdims(av),Gdims(v),-1,dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)

/*
  .def_static("identity",static_cast<CtensorArray (*)(const Gdims&, const Gdims&, const int, const int)>(&CtensorArray::identity))
  .def_static("identity",[](const Gdims& adims, const Gdims& dims, const int dev){
  return CtensorArray::identity(dims,-1,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("identity",[](const vector<int>& av, const vector<int>& v, const int dev){
  return CtensorArray::identity(Gdims(av),Gdims(v),-1,dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)
*/

  .def_static("sequential",static_cast<CtensorArray (*)(const Gdims&, const Gdims&, const int, const int)>(&CtensorArray::sequential))
  .def_static("sequential",[](const Gdims& adims, const Gdims& dims, const int dev){
      return CtensorArray::sequential(dims,-1,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("sequential",[](const vector<int>& av, const vector<int>& v, const int dev){
      return CtensorArray::sequential(Gdims(av),Gdims(v),-1,dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)

  .def_static("gaussian",static_cast<CtensorArray (*)(const Gdims&, const Gdims&, const int, const int)>(&CtensorArray::gaussian))
  .def_static("gaussian",[](const Gdims& adims, const Gdims& dims, const int dev){
      return CtensorArray::gaussian(dims,-1,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("gaussian",[](const vector<int>& av, const vector<int>& v, const int dev){
      return CtensorArray::gaussian(Gdims(av),Gdims(v),-1,dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)

//.def("get_k",&RtensorObj::get_k)
//    .def("getk",&RtensorObj::get_k)
  
//.def("get_nadims",&CtensorArray::get_adims)
  .def("get_adims",&CtensorArray::get_adims)
  .def("get_adim",&CtensorArray::get_adim)

//.def("get_ncdims",&CtensorArray::get_cdims)
  .def("get_cdims",&CtensorArray::get_cdims)
  .def("get_cdim",&CtensorArray::get_cdim)

  .def("get_cell",&CtensorArray::get_cell)
  .def("get_cell",[](const CtensorArray& obj, const vector<int> v){return obj.get_cell(Gindex(v));})
  .def("__call__",&CtensorArray::get_cell)
  .def("__call__",[](const CtensorArray& obj, const vector<int> v){return obj.get_cell(Gindex(v));})
  .def("__getitem__",&CtensorArray::get_cell)
  .def("__getitem__",[](const CtensorArray& obj, const vector<int> v){return obj.get_cell(Gindex(v));})

  .def("__setitem__",[](CtensorArray& obj, const Gindex& ix, const CtensorObj& x){
      obj.set_cell(ix,x);})
  .def("__setitem__",[](CtensorArray& obj, const vector<int> v, const CtensorObj& x){
      obj.set_cell(Gindex(v),x);})

  .def("__add__",[](const CtensorArray& x, const CtensorArray& y){return x.plus(y);})
  .def("__sub__",[](const CtensorArray& x, const CtensorArray& y){return x-y;})
  .def("__mul__",[](const CtensorArray& x, const float c){
      CtensorArray R(x.get_dims(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;})
  .def("__rmul__",[](const CtensorArray& x, const float c){
      CtensorArray R(x.get_dims(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;})
  .def("__mul__",[](const CtensorArray& x, const CtensorArray& y){
      CtensorArray R(x.get_dims().Mprod(y.get_dims()),x.get_nbu(),fill::zero,x.get_dev());
      R.add_mprod(x,y);
      return R;})

  .def("__iadd__",[](CtensorArray& x, const CtensorArray& y){x.add(y); return x;})
  .def("__isub__",[](CtensorArray& x, const CtensorArray& y){x.subtract(y); return x;})

  .def("__add__",[](const CtensorArray& x, const CtensorObj& y){return x.broadcast_plus(y);})

  .def("widen",&CtensorArray::widen)
  .def("reduce",&CtensorArray::reduce)

  .def("to",&CtensorArray::to_device)

  .def("str",&CtensorArray::str,py::arg("indent")="")
  .def("__str__",&CtensorArray::str,py::arg("indent")="")
  .def("__repr__",&CtensorArray::repr);

