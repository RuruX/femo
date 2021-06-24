 # shell-opt-mpi
 Thickness optimization problem for Reissner-Mindlin cantilever plate that can support parallel run. 
 
 Problem statement:
 Minimizing: the compliance
 subject to: the linear elastic PDE 
                Forces: uniformly distributed loads
                BC: clamped on the left side
 with respect to: the thickness distribution of the shell elements
  
 The benchmark problem for parallelization with FEniCS and OpenMDAO.
