"""
Dynamics module for spacecraft attitude simulation.
"""

from .quaternion import (
    normalize, conjugate, inverse, multiply,
    rotate_vector, quat_to_rotmat, rotmat_to_quat,
    error_quat, error_quat_to_vec, vec_to_error_quat,
    quat_derivative, box_plus, box_minus, slerp
)
from .rigid_body import RigidBody, inertia_matrix_from_principal

__all__ = [
    'normalize', 'conjugate', 'inverse', 'multiply',
    'rotate_vector', 'quat_to_rotmat', 'rotmat_to_quat',
    'error_quat', 'error_quat_to_vec', 'vec_to_error_quat',
    'quat_derivative', 'box_plus', 'box_minus', 'slerp',
    'RigidBody', 'inertia_matrix_from_principal'
]