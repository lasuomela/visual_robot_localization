from setuptools import setup
from setuptools import find_packages
setup(
name='hloc',
packages= find_packages( where = 'Hierarchical-Localization'),
package_dir={'': 'Hierarchical-Localization'},
include_package_data =True,
version='0.0.0'
)
