// This file is part of cnine, a lightweight C++ tensor library. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CellMask2r
#define _CellMask2r

#include <map>

#include "Gdims.hpp"


namespace cnine{


  class CellTlist2: public vector<pair<int,int> >{
  public:
    //vector<pair<int,int> > lst;
  };


  class CellMask2r{
  public:

    map<int,CellTlist2*> lists;
    vector<int> rstrides;
    vector<int> xstrides;
    vector<int> ystrides;

    mutable int n;
    mutable int* arrg=nullptr;
    mutable bool current=false;


    ~CellMask2r(){
      for(auto p:lists) delete p.second;
      if(arrg) CUDA_SAFE(cudaFree(arrg));
    }


  public:
    
    CellMask2r(const Gdims& rdims, const Gdims& xdims, const Gdims& ydims):
      rstrides(rdims.strides()), 
      xstrides(xdims.strides()), 
      ystrides(ydims.strides()){      
    }

     
  public:
 
    void push(const Gindex& rix, const Gindex& xix, const Gindex& yix){
      current=false;
      int r=rix(rstrides);
      CellTlist2* lst;
      auto it=lists.find(r);
      if(it!=lists.end()) lst=it->second;
      else{
	lst=new CellTlist2();
	lists[r]=lst;
      }
      lst->push_back(pair<int,int>(xix(xstrides),yix(ystrides)));
    }

    void prepare(const int dev) const{
      if(current) return;
      n=lists.size();
#ifdef _WITH_CUDA
      if(arrg) CUDA_SAFE(cudaFree(arrg));
      int N=n;
      for(auto p:lists)
	N+=2+2*p.second->size();
      int* arr=new int[N];

      int i=0;
      int lp=n;
      for(auto p:lists){
	arr[i]=lp;
	auto lst=*p.second;
	arr[lp]=p.first;
	arr[lp+1]=lst.size();
	for(int j=0; j<lst.size(); j++){
	  arr[lp+2+2*j]=lst[j].first;
	  arr[lp+2+2*j+1]=lst[j].second;
	}
	lp+=2+2*lst.size();
	i++;
      }

      CUDA_SAFE(cudaMalloc((void **)&arrg, N*sizeof(int)));
      CUDA_SAFE(cudaMemcpy(arrg,arr,N*sizeof(int),cudaMemcpyHostToDevice));
      delete[] arr;
#endif
    }

    string str(const string indent=""){
      ostringstream oss;
      for(auto it: lists){
	oss<<indent<<Gindex(it.first,rstrides)<<"<-(";
	//for(auto p:it.second->lst)
	const vector<pair<int,int> >& lst=*it.second;
	for(int i=0; i<lst.size(); i++){
	  oss<<"("<<Gindex(lst[i].first,xstrides)<<","<<Gindex(lst[i].second,ystrides)<<")";
	  if(i<lst.size()-1) oss<<",";
	}
	oss<<")"<<endl;
      }
      return oss.str();
    }

  };


}

#endif 
