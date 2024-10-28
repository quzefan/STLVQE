from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

if torch.cuda.is_available():
# if False:
    print('Including CUDA code.')
    setup(
        name='quadrilinear6d',
        ext_modules=[
            CUDAExtension('quadrilinear6d', [
                'src/quadrilinear6d_cuda.cpp',
                'src/quadrilinear6d_kernel.cu',
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
else:
    print('NO CUDA is found. Fall back to CPU.')
    setup(name='quadrilinear6d',
        ext_modules=[CppExtension(name = 'quadrilinear6d', 
                                  sources= ['src/quadrilinear6d.cpp'],
                                  extra_compile_args=['-fopenmp'])],
        cmdclass={'build_ext': BuildExtension})
