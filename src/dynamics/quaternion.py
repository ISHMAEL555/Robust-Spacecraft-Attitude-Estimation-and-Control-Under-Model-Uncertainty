"""
Quaternion algebra for spacecraft attitude representation.

Conventions (Hamilton product, scalar-last):
- q = [x, y, z, w] where w is the scalar part
- The active rotation matrix maps body components to inertial components.
- The filter uses a right-multiplicative error: q_true = q_hat ⊗ delta_q.
- The legacy error_quat/box_plus helpers below use a left-multiplicative error.
"""

import numpy as np
from typing import Tuple, Union


def normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Cannot normalize zero quaternion")
    return q / norm


def is_unit(q: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if quaternion is unit length within tolerance."""
    return abs(np.linalg.norm(q) - 1.0) < tol


def conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate: q* = [-v, w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def inverse(q: np.ndarray) -> np.ndarray:
    """Quaternion inverse (for unit quaternions, same as conjugate)."""
    return conjugate(q) / np.dot(q, q)


def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Quaternion multiplication: q1 ⊗ q2 (Hamilton convention).
    q = [x, y, z, w] (scalar-last)
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    
    return np.array([x, y, z, w])


def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q: v_rot = q * v * q^*."""
    # Convert vector to pure quaternion
    v_q = np.array([v[0], v[1], v[2], 0.0])
    # q * v * q^*
    result = multiply(multiply(q, v_q), conjugate(q))
    return result[:3]


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion to the active body-to-inertial rotation matrix."""
    q = normalize(q)
    x, y, z, w = q
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    return R


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a body-to-inertial rotation matrix to a scalar-last quaternion."""
    trace = np.trace(R)
    
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    
    return normalize(np.array([x, y, z, w]))


def error_quat(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """
    Compute the left-relative quaternion error: delta_q = q_true ⊗ q_est^*.

    This gives q_true = delta_q ⊗ q_est. The MEKF uses right_error_quat instead.
    """
    return multiply(q_true, inverse(q_est))


def right_error_quat(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """Return the right-multiplicative error ``q_est⁻¹ ⊗ q_true``."""
    return multiply(inverse(q_est), q_true)


def error_quat_to_vec(delta_q: np.ndarray) -> np.ndarray:
    """
    Convert an error quaternion to its rotation vector (3x1).
    Returns delta_theta (3-vector).
    """
    delta_q = normalize(delta_q)
    # Ensure positive scalar part for minimal rotation
    if delta_q[3] < 0:
        delta_q = -delta_q
    vector_norm = np.linalg.norm(delta_q[:3])
    if vector_norm < 1e-12:
        return 2.0 * delta_q[:3]
    angle = 2.0 * np.arctan2(vector_norm, delta_q[3])
    return delta_q[:3] * (angle / vector_norm)


def vec_to_error_quat(delta_theta: np.ndarray) -> np.ndarray:
    """
    Convert small error vector to error quaternion.
    delta_q ≈ [0.5 * delta_theta, 1]^T
    """
    norm = np.linalg.norm(delta_theta)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    
    half_theta = 0.5 * delta_theta
    sin_half = np.sin(norm * 0.5)
    cos_half = np.cos(norm * 0.5)
    
    if norm > 0:
        axis = delta_theta / norm
        return np.array([
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            cos_half
        ])
    else:
        return np.array([0.0, 0.0, 0.0, 1.0])


def quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Body-to-inertial quaternion kinematics: q_dot = 0.5 * q ⊗ [omega, 0]
    """
    wx, wy, wz = omega
    x, y, z, w = q
    
    # q_dot = 0.5 * [ w*wx + y*wz - z*wy,
    #                 w*wy + z*wx - x*wz,
    #                 w*wz + x*wy - y*wx,
    #                -x*wx - y*wy - z*wz ]
    q_dot = 0.5 * np.array([
        w*wx + y*wz - z*wy,
        w*wy + z*wx - x*wz,
        w*wz + x*wy - y*wx,
        -x*wx - y*wy - z*wz
    ])
    return q_dot


def box_plus(q: np.ndarray, delta_theta: np.ndarray) -> np.ndarray:
    """
    Legacy left box-plus operation: q_new = Exp(delta_theta) ⊗ q.

    The MEKF uses right_box_plus for its body-frame error-state update.
    """
    delta_q = vec_to_error_quat(delta_theta)
    return multiply(delta_q, q)


def right_box_plus(q: np.ndarray, delta_theta: np.ndarray) -> np.ndarray:
    """Inject a body-frame error on the right: ``q_new = q ⊗ Exp(delta_theta)``."""
    return multiply(q, vec_to_error_quat(delta_theta))


def right_box_minus(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """Return the rotation vector for the right error ``q_est⁻¹ ⊗ q_true``."""
    return error_quat_to_vec(right_error_quat(q_true, q_est))


def box_minus(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Box-minus operation: delta_theta = Log(q1 ⊗ q2^*)
    Returns the error vector from q2 to q1.
    """
    delta_q = error_quat(q1, q2)
    return error_quat_to_vec(delta_q)


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between q1 and q2."""
    q1 = normalize(q1)
    q2 = normalize(q2)
    
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    if dot > 0.9995:
        # Linear interpolation for very close quaternions
        result = q1 + t * (q2 - q1)
        return normalize(result)
    
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    
    return normalize(s1 * q1 + s2 * q2)


# Test functions
def test_quaternion_algebra():
    """Run basic quaternion algebra tests."""
    # Test identity
    q_id = np.array([0.0, 0.0, 0.0, 1.0])
    assert is_unit(q_id)
    
    # Test normalization
    q = np.array([1.0, 2.0, 3.0, 4.0])
    q_norm = normalize(q)
    assert is_unit(q_norm)
    
    # Test conjugate/inverse
    q = np.array([0.1, 0.2, 0.3, 0.9])
    q = normalize(q)
    q_inv = inverse(q)
    q_mult = multiply(q, q_inv)
    assert np.allclose(q_mult, q_id, atol=1e-10)
    
    # Test rotation
    v = np.array([1.0, 0.0, 0.0])
    q_90z = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])  # 90 deg about z
    v_rot = rotate_vector(q_90z, v)
    assert np.allclose(v_rot, [0.0, 1.0, 0.0], atol=1e-10)
    
    # Test rotmat conversion
    R = quat_to_rotmat(q_90z)
    q_back = rotmat_to_quat(R)
    assert np.allclose(np.abs(q_back), np.abs(q_90z), atol=1e-10)
    
    # Test box_plus/box_minus
    q1 = normalize(np.array([0.1, 0.2, 0.3, 0.9]))
    delta = np.array([0.01, -0.02, 0.005])
    q2 = box_plus(q1, delta)
    delta_back = box_minus(q2, q1)
    assert np.allclose(delta, delta_back, atol=1e-10)
    
    # Test error quaternion
    q_true = normalize(np.array([0.2, 0.1, -0.1, 0.97]))
    q_est = normalize(np.array([0.19, 0.11, -0.09, 0.97]))
    delta_q = error_quat(q_true, q_est)
    delta_theta = error_quat_to_vec(delta_q)
    q_reconstructed = box_plus(q_est, delta_theta)
    assert np.allclose(np.abs(q_reconstructed), np.abs(q_true), atol=1e-10)
    
    print("All quaternion tests passed!")


if __name__ == "__main__":
    test_quaternion_algebra()
