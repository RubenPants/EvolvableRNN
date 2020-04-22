"""
setup.py

Transform the Cython files to their C-counterpart by running this setup-file from root.

Note: This file must be called from build.py, or called via the following command:
      python3 <folder_from_root>setup.py build_ext --inplace
"""
from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(ext_modules=cythonize('utils/cy/vec2d_cy.pyx'),
      )
setup(ext_modules=cythonize('utils/cy/line2d_cy.pyx'),
      )
setup(ext_modules=cythonize('environment/cy/sensors_cy.pyx'),
      include_dirs=[numpy.get_include()],
      )
setup(ext_modules=cythonize('environment/cy/robot_cy.pyx'),
      include_dirs=[numpy.get_include()],
      )
setup(ext_modules=cythonize('environment/cy/game_cy.pyx'),
      include_dirs=[numpy.get_include()],
      )
setup(ext_modules=cythonize('environment/cy/env_multi_cy.pyx'),
      include_dirs=[numpy.get_include()],
      )
setup(ext_modules=cythonize('population/utils/network_util/cy/activations_cy.pyx'),
      include_dirs=[numpy.get_include()],
      )
setup(ext_modules=cythonize('population/utils/network_util/cy/feed_forward_net_cy.pyx'),
      include_dirs=[numpy.get_include()],
      )
setup(ext_modules=cythonize('population/utils/rnn_cell_util/cy/simple_rnn_cy.pyx'),
      include_dirs=[numpy.get_include()],
      )
setup(ext_modules=cythonize('population/utils/rnn_cell_util/cy/berkeley_gru_cy.pyx'),
      include_dirs=[numpy.get_include()],
      )
setup(ext_modules=cythonize('population/utils/rnn_cell_util/cy/berkeley_gru_no_reset_cy.pyx'),
      include_dirs=[numpy.get_include()],
      )
setup(ext_modules=cythonize('population/utils/rnn_cell_util/cy/berkeley_gru_no_update_cy.pyx'),
      include_dirs=[numpy.get_include()],
      )
setup(ext_modules=cythonize('population/utils/rnn_cell_util/cy/lstm_cy.pyx'),
      include_dirs=[numpy.get_include()],
      )
