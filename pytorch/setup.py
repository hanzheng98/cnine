import sys,os
import torch 
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

os.environ['CUDA_HOME']='/usr/local/cuda'
#os.environ["CC"] = "clang"

CUDA_HOME='/usr/local/cuda'

print(torch.cuda.is_available())

setup(name='cnine',
ext_modules=[CUDAExtension('cnine', ['cnine_py.cpp'],
                                 include_dirs=['/usr/local/cuda/include',
                                               '../include','../objects/scalar','../objects/tensor'],
                                 library_dirs=['/usr/local/cuda/lib'],
                                 runtime_libraries=['cudart'],
                                 libraries=['cudart','dl'],
                                 extra_compile_args = {'nvcc': ['-D_WITH_CUDA'],
                                                       'cxx': ['-std=c++14', '-Wno-sign-compare','-Wno-deprecated-declarations',
                                                               '-D_WITH_CUDA']},
                                 depends=['setup.py'])], 
      cmdclass={'build_ext': BuildExtension}
      )

