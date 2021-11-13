

template<typename COP, typename OBJ>
void def_inner(pybind11::module& m){
  m.def(("inner_"+COP::shortname()).c_str(),
    [](OBJ& R, const OBJ& x, const OBJ& y){
      COP op;
      cnine::CellwiseBiCmap(op,R,x,y);
    });
}

template<typename COP, typename OBJ>
void def_cellwise(pybind11::module& m){
  m.def(("cellwise_"+COP::shortname()).c_str(),
    [](OBJ& R, const OBJ& x, const OBJ& y){
      COP op;
      cnine::CellwiseBiCmap(op,R,x,y);
    });
}

template<typename COP, typename OBJ>
void def_cellwise(pybind11::module& m){
  m.def(("cellwise_"+COP::shortname()).c_str(),
    [](OBJ& R, const OBJ& x, const OBJ& y){
      COP op;
      cnine::CellwiseBiCmap(op,R,x,y);
    });
}


