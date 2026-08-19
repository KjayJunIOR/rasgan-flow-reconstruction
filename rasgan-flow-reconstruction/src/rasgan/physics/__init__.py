"""Physics operators and solvers used by physics-informed losses."""

from .operators import grad_central, laplacian, divergence, curl2d
from .poisson import poisson_solve_fft, poisson_solve_jacobi, streamfunction_from_omega
