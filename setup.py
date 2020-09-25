import setuptools
from distutils.core import Extension

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

     name='psrmodels',  

     version='1.0.3',

     author="Nestor Sanchez",

     author_email="nestor.sag@gmail.com",

     packages = setuptools.find_namespace_packages(include=['psrmodels.*']),

     description="Models for security of supply in power systems' reliability",

     license = "MIT",

     setup_requires = ['cffi>=1.0.0'],

     install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'seaborn',
        'cffi>=1.0.0',
        'shapely'

    ],

     long_description=long_description,

     long_description_content_type="text/markdown",

     url="https://bitbucket.com/nestorsag/phd",

     cffi_modules=[
       "psrmodels/_c_builders/build_timedependence.py:ffibuilder",
       "psrmodels/_c_builders/build_univarmargins.py:ffibuilder",
       "psrmodels/_c_builders/build_bivarmargins.py:ffibuilder"],

     classifiers=[

         "Programming Language :: Python :: 3",

         "License :: OSI Approved :: MIT License",

         "Operating System :: OS Independent",

     ],

     #ext_package = 'psrmodels',

     #ext_modules = [mcgensim_module]

     download_url = 'https://github.com/nestorSag/psrmodels/archive/1.0.3.tar.gz'

 )