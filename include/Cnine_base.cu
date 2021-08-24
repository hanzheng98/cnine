// This file is part of cnine, a lightweight C++ tensor library. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "Cnine_base.hpp"

// __device__ __constant__ unsigned char cg_cmem[CNINE_CONST_MEM_SIZE];

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
cublasHandle_t cnine_cublas;
//cublasCreate(&Cengine_cublas);
#endif 
