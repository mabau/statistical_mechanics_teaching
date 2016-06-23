__author__ = "Martin Bauer <martin.bauer@fau.de>"
__copyright__ = "Copyright 2016, Martin Bauer"
__license__ = "GPL"
__version__ = "3"

from setuptools import setup, find_packages

print("find_packages()", find_packages())

setup(
    name='statistical_mechanics_teaching',

    version='0.9.0',

    description='Molecular dynamics simulations based on LAMMPS for teaching statistical mechanics',

    # The project's main homepage.
    url='www10.cs.fau.de/~bauer',

    # Author details
    author='Martin Bauer',
    author_email='martin.bauer@fau.de',

    license='GPLv3',

    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    keywords='teaching physics molecular dynamics',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    # This is done by conda
    #install_requires=['lammps', 'numpy', 'scipy', 'matplotlib'], 
)
