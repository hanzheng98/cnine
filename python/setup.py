import sys,os
import torch 
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

#os.environ['CUDA_HOME']='/usr/local/cuda'
os.environ["CC"] = "clang"

#CUDA_HOME='/usr/local/cuda'
#print(torch.cuda.is_available())

#CXXFLAGS += '-D_GLIBCXX_USE_CXX11_ABI=0'

setup(name='cnine',
ext_modules=[CppExtension('cnine', ['cnine_py.cpp'],
                                 include_dirs=['/usr/local/cuda/include',
                                               '../include','../objects/scalar','../objects/tensor',
                                               '../objects/tensor_array','../objects/tensor_array/cell_ops'],
#                                 library_dirs=['/usr/local/cuda/lib'],
#                                 runtime_libraries=['cudart'],
#                                 libraries=['cudart','dl'],
                                 extra_compile_args = {
#                              'nvcc': ['-D_WITH_CUDA'],
                                                       'cxx': ['-std=c++14',
                                                               '-Wno-sign-compare',
                                                               '-Wno-deprecated-declarations',
                                                               '-Wno-unused-variable',
                                                               '-Wno-reorder-ctor'
#                                                               '-D_GLIBCXX_USE_CXX11_ABI=0'
                                                               ]},
                                 depends=['setup.py','cnine_py.cpp','rtensor_py.cpp','ctensor_py.cpp',
                                          'rtensorarr_py.cpp','ctensorarr_py.cpp'])], 
      cmdclass={'build_ext': BuildExtension}
      )

