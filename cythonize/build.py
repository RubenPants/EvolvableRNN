"""
Call this file (in Linux environment) to build all the Cython files via setup.py

Note: Cython must installed beforehand (i.e. pip3 install cython)
"""
import os

if __name__ == '__main__':
    os.chdir("..")
    os.system('python3 cythonize/setup.py build_ext --inplace')
