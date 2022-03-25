from pathlib import Path
from setuptools import setup
from setuptools import find_packages

description = ['Tools and baselines for visual localization and mapping']

root = Path(__file__).parent / 'Hierarchical-Localization'
with open(str(root / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()
with open(str(root / 'hloc/__init__.py'), 'r') as f:
    version = eval(f.read().split('__version__ = ')[1].split()[0])
with open(str(root / 'requirements.txt'), 'r') as f:
    dependencies = f.read().split('\n')

setup(
    name='hloc',
    version=version,
    packages= find_packages( where = 'Hierarchical-Localization'),
    package_dir={'': 'Hierarchical-Localization'},
    include_package_data =True,
    python_requires='>=3.6',
    install_requires=dependencies,
    author='Paul-Edouard Sarlin',
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/cvg/Hierarchical-Localization/',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
