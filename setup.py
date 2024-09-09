# define how to build/compile the cpp code 
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='interpolation', 
    version='1.0',
    author='stanleyedward',
    author_email='114278820+stanleyedward@users.noreply.github.com',
    description='cpp_extension',
    long_description='cpp_extension_long_desc',
    ext_modules=[
        CppExtension(
            name='interpolation',
            sources=['interpolation.cpp'],
            # extra_compile_args=['-g'],
            # extra_link_flags=['-Wl,--no-as-needed', '-lm']
            )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)