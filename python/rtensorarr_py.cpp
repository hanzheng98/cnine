pybind11::class_<RtensorArray>(m,"rtensor_arr")

  .def(pybind11::init<const Gdims&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_raw&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_zero&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_ones&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_identity&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_gaussian&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_sequential&>())

  .def_static("zero",static_cast<RtensorArray (*)(const Gdims&, const Gdims&, const int, const int)>(&RtensorArray::zero))
  .def_static("zero",[](const Gdims& adims, const Gdims& dims, const int dev){
      return RtensorArray::zero(dims,-1,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("zero",[](const vector<int>& av, const vector<int>& v, const int dev){
      return RtensorArray::zero(Gdims(av),Gdims(v),-1,dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)

  .def_static("ones",static_cast<RtensorArray (*)(const Gdims&, const Gdims&, const int, const int)>(&RtensorArray::ones))
  .def_static("ones",[](const Gdims& adims, const Gdims& dims, const int dev){
      return RtensorArray::ones(dims,-1,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("ones",[](const vector<int>& av, const vector<int>& v, const int dev){
      return RtensorArray::ones(Gdims(av),Gdims(v),-1,dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)

/*
  .def_static("identity",static_cast<RtensorArray (*)(const Gdims&, const Gdims&, const int, const int)>(&RtensorArray::identity))
  .def_static("identity",[](const Gdims& adims, const Gdims& dims, const int dev){
  return RtensorArray::identity(dims,-1,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("identity",[](const vector<int>& av, const vector<int>& v, const int dev){
  return RtensorArray::identity(Gdims(av),Gdims(v),-1,dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)
*/

  .def_static("sequential",static_cast<RtensorArray (*)(const Gdims&, const Gdims&, const int, const int)>(&RtensorArray::sequential))
  .def_static("sequential",[](const Gdims& adims, const Gdims& dims, const int dev){
      return RtensorArray::sequential(dims,-1,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("sequential",[](const vector<int>& av, const vector<int>& v, const int dev){
      return RtensorArray::sequential(Gdims(av),Gdims(v),-1,dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)

  .def_static("gaussian",static_cast<RtensorArray (*)(const Gdims&, const Gdims&, const int, const int)>(&RtensorArray::gaussian))
  .def_static("gaussian",[](const Gdims& adims, const Gdims& dims, const int dev){
      return RtensorArray::gaussian(dims,-1,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("gaussian",[](const vector<int>& av, const vector<int>& v, const int dev){
      return RtensorArray::gaussian(Gdims(av),Gdims(v),-1,dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)

//.def("get_k",&RtensorObj::get_k)
//    .def("getk",&RtensorObj::get_k)
  
  .def("get_nadims",&RtensorArray::get_adims)
  .def("get_adims",&RtensorArray::get_adims)
  .def("get_adim",&RtensorArray::get_adim)

  .def("get_ncdims",&RtensorArray::get_cdims)
  .def("get_cdims",&RtensorArray::get_cdims)
  .def("get_cdim",&RtensorArray::get_cdim)

  .def("get_cell",&RtensorArray::get_cell)
  .def("get_cell",[](const RtensorArray& obj, const vector<int> v){return obj.get_cell(Gindex(v));})
  .def("__call__",&RtensorArray::get_cell)
  .def("__call__",[](const RtensorArray& obj, const vector<int> v){return obj.get_cell(Gindex(v));})
  .def("__getitem__",&RtensorArray::get_cell)
  .def("__getitem__",[](const RtensorArray& obj, const vector<int> v){return obj.get_cell(Gindex(v));})

  .def("__setitem__",[](RtensorArray& obj, const Gindex& ix, const RtensorObj& x){
      obj.set_cell(ix,x);})
  .def("__setitem__",[](RtensorArray& obj, const vector<int> v, const RtensorObj& x){
      obj.set_cell(Gindex(v),x);})

  .def("__add__",[](const RtensorArray& x, const RtensorArray& y){return x.plus(y);})
  .def("__sub__",[](const RtensorArray& x, const RtensorArray& y){return x-y;})
  .def("__mul__",[](const RtensorArray& x, const float c){
      RtensorArray R(x.get_dims(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;})
  .def("__rmul__",[](const RtensorArray& x, const float c){
      RtensorArray R(x.get_dims(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;})
  .def("__mul__",[](const RtensorArray& x, const RtensorArray& y){
      RtensorArray R(x.get_dims().Mprod(y.get_dims()),x.get_nbu(),fill::zero,x.get_dev());
      R.add_mprod(x,y);
      return R;})

  .def("__iadd__",[](RtensorArray& x, const RtensorArray& y){x.add(y); return x;})
  .def("__isub__",[](RtensorArray& x, const RtensorArray& y){x.subtract(y); return x;})

  .def("__add__",[](const RtensorArray& x, const RtensorObj& y){return x.broadcast_plus(y);})

  .def("widen",&RtensorArray::widen)
  .def("reduce",&RtensorArray::reduce)

  .def("to",&RtensorArray::to_device)

  .def("str",&RtensorArray::str,py::arg("indent")="")
  .def("__str__",&RtensorArray::str,py::arg("indent")="");
