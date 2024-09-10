# define how to build/compile the cpp code 
import glob 
import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
include_dirs = [os.path.join(ROOT_DIR, "include")]
sources = glob.glob("*.cpp") + glob.glob("*.cu") #list of cpp and cu files in dir

setup(
    name='interpolation', 
    version='1.0',
    author='stanleyedward',
    author_email='114278820+stanleyedward@users.noreply.github.com',
    description='cpp_extension',
    long_description='cpp_extension_long_desc',
    ext_modules=[
        CUDAExtension(
            name='interpolation',
            include_dirs=include_dirs,
            sources=sources,
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"]},
            # extra_link_flags=['-Wl,--no-as-needed', '-lm']
            )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)