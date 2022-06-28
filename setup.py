# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch, os
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from setuptools import setup, find_packages

this_dir = os.path.dirname(os.path.realpath(__file__))
torch_dir = os.path.dirname(torch.__file__)
conda_include_dir = '/'.join(torch_dir.split('/')[:-4]) + '/include'

# extra = {'cxx': ['-std=c++11', '-fopenmp'], 'nvcc': ['-std=c++11', '-Xcompiler', '-fopenmp']}
extra = {'cxx': ['-std=c++11','-g', '-fopenmp', '-DBUILD_WITH_EASY_PROFILER'], 'nvcc': ['-std=c++11', '-Xcompiler', '-fopenmp', '-DBUILD_WITH_EASY_PROFILER']}
#extra = {'cxx': ['-std=c++11', '-fopenmp'], 'nvcc': ['-std=c++11', '-Xcompiler', '-fopenmp']}
print(this_dir)
setup(
    name='sparseconvnet',
    version='0.2',
    description='Submanifold (Spatially) Sparse Convolutional Networks https://arxiv.org/abs/1706.01307',
    author='Facebook AI Research',
    author_email='benjamingraham@fb.com',
    url='https://github.com/facebookresearch/SparseConvNet',
    packages=['sparseconvnet','sparseconvnet.SCN'],
    ext_modules=[
      CUDAExtension('sparseconvnet.SCN',
        [
         'sparseconvnet/SCN/cuda.cu', 'sparseconvnet/SCN/sparseconvnet_cuda.cpp', 'sparseconvnet/SCN/pybind.cpp'],
        include_dirs=[conda_include_dir,this_dir+'/sparseconvnet/SCN/',this_dir+'/extra/easy_profiler/easy_profiler_core/include',this_dir + '/extra/cudpp/include'],
        library_dirs = ['/usr/local/lib', this_dir + '/extra/cudpp/build/lib',this_dir+'/extra/easy_profiler/build/bin'],
        libraries = ['easy_profiler', 'cudpp', 'cudpp_hash'],
        extra_compile_args=extra)
      if torch.cuda.is_available()  else
      CppExtension('sparseconvnet.SCN',
        ['sparseconvnet/SCN/pybind.cpp', 'sparseconvnet/SCN/sparseconvnet_cpu.cpp'],
        include_dirs=[conda_include_dir, this_dir+'/sparseconvnet/SCN/'],
        library_dirs = ['/usr/local/lib', this_dir + '/extra/cudpp/build/lib',this_dir+'/extra/easy_profiler/build/bin',this_dir+'/extra/easy_profiler/easy_profiler_core/include',this_dir + '/extra/cudpp/include'],
        libraries = ['easy_profiler', 'cudpp', 'cudpp_hash'],
        extra_compile_args=extra['cxx'])],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
