# fe-csdl-framework
The fe-csdl-opt module is a general framework for PDE-constrained optimization problems. The code is still under active developement and we expect it to be available to the public for their research applications by 2022 Fall.

Currently, the implementation is based on the latest commits of [csdl](https://github.com/LSDOlab/csdl) and its [new Python backend](https://github.com/LSDOlab/python_csdl_backend).

## Installation

It's recommended to use conda for installing the module and its dependencies.

- Create a conda environment for FEniCSx with a specific Python version 
  ```
  conda create -n fenicsx python=3.9.10
  ```
  - Conda-forge will install Python 3.10 by default, and it would be incompatible with the SNOPT installation in ModOpt, which requires Python 3.9
- Activate the conda enviroment 
  ```
  conda activate fenicsx
  ```
- Install the latest FEniCSx and its add-on packages, mpich for parallel processing and pyvista for visualization
  ```
  conda install -c conda-forge fenics-dolfinx=0.5.1 mpich pyvista
  ```
- Git clone and install [csdl](https://github.com/LSDOlab/csdl), [csdl_om](https://github.com/LSDOlab/csdl_om) (optional), and [python_csdl_backend](https://github.com/LSDOlab/python_csdl_backend) by `pip`
- Git clone and install [fe-csdl-framework](https://github.com/RuruX/fe-csdl-framework) by `pip`
- (optional) Install SNOPT for optimization (you will need to contact the developer of ModOpt for instructions)
- (optional) Install [ModOpt](https://github.com/LSDOlab/modopt) by `pip` and test with modopt/modopt/external_packages/csdl/test_scaler.py
