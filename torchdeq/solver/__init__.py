"""
The `torchdeq.solver` module provides a set of solvers for finding fixed points in Deep Equilibrium Models (DEQs). 
These solvers are used to iteratively refine the predictions of a DEQ model until they reach a stable state, or "equilibrium".

This module includes implementations of several popular fixed-point solvers, including Anderson acceleration (`anderson_solver`), 
Broyden's method (`broyden_solver`), and fixed-point iteration (`fixed_point_iter`). 
It also provides a faster version of fixed-point iteration (`simple_fixed_point_iter`) 
that omits convergence monitoring for speed improvements.

The `get_solver` function allows users to retrieve a specific solver by its key, and the `register_solver` function 
allows users to add their own custom solvers to the module.

Example:
    To retrieve a solver, call this `get_solver` function:

    >>> solver = get_solver('anderson')

    To register a user-developed solver, call this `register_solver` function:

    >>> register_solver('newton', newton_solver)
"""
from .anderson import anderson_solver
from .broyden import broyden_solver
from .fp_iter import fixed_point_iter, simple_fixed_point_iter

from .utils import solver_stat_from_final_step


__all__ = ['register_solver', 'get_solver', 'solver_stat_from_final_step']


_solvers = {
        'anderson': anderson_solver,
        'broyden': broyden_solver,
        'fixed_point_iter': fixed_point_iter,
        'simple_fixed_point_iter': simple_fixed_point_iter,
        }


def get_solver(key):
    """
    Retrieves a fixed point solver from the registered solvers by its key.

    Supported solvers: ``'anderson'``, ``'broyden'``, ``'fixed_point_iter'``, ``'simple_fixed_point_iter'``.

    Args:
        key (str): The key of the solver to retrieve. This should match one of the keys used to register a solver.

    Returns:
        callable: The solver function associated with the provided key.

    Raises:
        AssertionError: If the key does not match any of the registered solvers.

    Example:
        >>> solver = get_solver('anderson')
    """
    assert key in _solvers

    return _solvers[key]


def register_solver(solver_type, solver):
    """
    Registers a user-defined fixed point solver. 
    This solver can be designated using args.f_solver and args.b_solver.

    This method adds a new entry to the solver dict with the key as
    the specified ``solver_type`` and the value as the ``solver``.

    Args:
        solver_type (str): The type of solver to register. This will be used as the key in the solver dict.
        solver_class (callable): The solver function. This will be used as the value in the solver dict.

    Example:
        >>> register_solver('newton', newton_solver)
    """
    _solvers[solver_type] = solver