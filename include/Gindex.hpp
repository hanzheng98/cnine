// This file is part of cnine, a lightweight C++ tensor library. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CnineGindex
#define _CnineGindex

#include "Cnine_base.hpp"
#include "Gdims.hpp"


namespace cnine{
    

  class Gindex: public vector<int>{
  public:

    Gindex(){}

    Gindex(const int k, const fill_zero& dummy): vector<int>(k){
    }

    Gindex(const fill_zero& dummy){
    }

    Gindex(const vector<int>& x){
      for(auto p:x) if(p>=0) push_back(p);
    }

    Gindex(const initializer_list<int>& list): vector<int>(list){}

    Gindex(const int i0):
      Gindex({i0}){}

    Gindex(const int i0, const int i1): vector<int>(2){
      (*this)[0]=i0;
      (*this)[1]=i1;
    }

    Gindex(const int i0, const int i1, const int i2): vector<int>(3){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
    }

    Gindex(const int i0, const int i1, const int i2, const int i3): vector<int>(4){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
    }

    Gindex(int a, const Gdims& dims): 
      vector<int>(dims.size()){
      for(int i=size()-1; i>=0; i--){
	(*this)[i]=a%dims[i];
	a=a/dims[i];
      }
    }

    Gindex(int a, vector<int> strides): 
      vector<int>(strides.size()){
      for(int i=size()-1; i>=1; i--){
	(*this)[i]=(a%strides[i-1])/strides[i];
      }
      (*this)[0]=a/strides[0];
    }

    

  public:
    
    int k() const{
      return size();
    }

    int operator()(const int i) const{
      return (*this)[i];
    }

    void set(const int i, const int x){
      (*this)[i]=x;
    }

    int operator()(const vector<int>& strides) const{
      assert(strides.size()>=size());
      int t=0; 
      for(int i=0; i<size(); i++) 
	t+=(*this)[i]*strides[i];
      return t;
    }

    int operator()(const Gdims& dims) const{
      assert(dims.size()>=size());
      int s=1;
      int t=0; 
      for(int i=size()-1; i>=0; i--){
	t+=(*this)[i]*s;
	s*=dims[i];
      }
      return t;
    }

    int to_int(const Gdims& dims) const{
      assert(dims.size()>=size());
      int s=1;
      int t=0; 
      for(int i=size()-1; i>=0; i--){
	t+=(*this)[i]*s;
	s*=dims[i];
      }
      return t;
    }


  public:

    string str() const{
      string str="("; 
      int k=size();
      for(int i=0; i<k; i++){
	str+=std::to_string((*this)[i]);
	if(i<k-1) str+=",";
      }
      return str+")";
    }

    string str_bare() const{
      string str; 
      int k=size();
      for(int i=0; i<k; i++){
	str+=std::to_string((*this)[i]);
	if(i<k-1) str+=",";
      }
      return str;
    }

    string repr() const{
      return "<cnine::Gdims"+str()+">";
    }


  friend ostream& operator<<(ostream& stream, const Gindex& v){
    stream<<v.str(); return stream;}

  };



}


namespace std{
  template<>
  struct hash<cnine::Gindex>{
  public:
    size_t operator()(const cnine::Gindex& ix) const{
      size_t t=hash<int>()(ix[0]);
      for(int i=1; i<ix.size(); i++) t=(t<<1)^hash<int>()(ix[i]);
      return t;
    }
  };
}




#endif

