# femo
**femo** is a general framework for using **F**inite **E**lement in PDE-constrained **M**ultidisciplinary **O**ptimization problems. It relies on [FEniCSx](https://fenicsproject.org/) to provide solutions and partial derivatives of the PDE residuals, and uses [CSDL](https://github.com/LSDOlab/csdl) as the umbrella for coupling and mathematical modeling of the whole problem. The code is still under active developement and we expect it to be available to the public for their research applications by 2023 Winter. 

For modeling and simulation, you need to install `FEniCSx`, `CSDL` and the Python-based backend of `CSDL` - [python_csdl_backend](https://github.com/LSDOlab/python_csdl_backend); for optimization, you will also need [ModOpt](https://github.com/LSDOlab/modopt) on top of them for the black-box optimizers. 

## Installation

It's recommended to use conda for installing the module and its dependencies.

- Create a conda environment for FEniCSx with a specific Python version (Python 3.9) that is compatible with all of the dependencies
  ```
  conda create -n fenicsx python=3.9.10
  ```
  (Python 3.9.7 also works if Python 3.9.10 is unavailable in your conda)
- Activate the conda enviroment 
  ```
  conda activate fenicsx
  ```
- Install FEniCSx
  ```
  conda install -c conda-forge fenics-dolfinx=0.5.1
  ```
- Git clone and install [CSDL](https://github.com/LSDOlab/csdl), and [python_csdl_backend](https://github.com/LSDOlab/python_csdl_backend) by `pip`
- Git clone and install [femo](https://github.com/RuruX/femo) by `pip`
- (optional) Install [SNOPT](https://ccom.ucsd.edu/~optimizers/solvers/snopt/) for optimization (licence required)
- (optional) Install [ModOpt](https://github.com/LSDOlab/modopt) by `pip` and test with `modopt/modopt/external_packages/csdl/test_scaler.py`


## Cite us
@misc{xiang2024,
    author = "Xiang, Ru 
            and van Schie, Sebastiaan P.C.
            and Scotzniovsky, Luca 
            and Yan, Jiayao
            and Kamensky, David 
            and Hwang, John T.",
    title  = "Automating adjoint sensitivity analysis for multidisciplinary models involving partial differential equations",
    howpublished = {Jul 2024, Preprint available at \url{https://doi.org/10.21203/rs.3.rs-4265983/v1}}
}

@misc{scotzniovsky2024,
    author = "Scotzniovsky, Luca 
            and Xiang, Ru 
            and Cheng, Zeyu 
            and Rodriguez, Gabriel 
            and Kamensky, David 
            and Mi, Chris 
            and Hwang, John T.",
    title  = "Geometric Design of Electric Motors Using Adjoint-based Shape Optimization",
    howpublished = {Feb 2024, Preprint available at \url{https://doi.org/10.21203/rs.3.rs-3941981/v1}}
}
