"""Spacecraft attitude controller with body-frame quaternion feedback.

The correction quaternion is q_e = q_current^{-1} ⊗ q_desired, so its vector
part points along the body-frame rotation that carries the current attitude
to the desired one. The rate error and commanded torque are both expressed in
the current body frame:

    quaternion_feedback:  tau = +K_q q_e,v - K_omega (omega - omega_d)
    pd:                   tau = 2 K_q q_e,v - K_omega (omega - omega_d)
    rate_profile:         tau = K_omega (omega_cmd - omega_d_body - omega),  see below

For small angles q_e,v ≈ theta_e / 2, so the effective attitude stiffness is
K_q / 2 for 'quaternion_feedback' and K_q for 'pd'. The command is limited
in Euclidean norm, which preserves the commanded torque direction.

'rate_profile' is the saturation-aware alternative. The correction rotation
vector phi (magnitude |phi|, axis n) is turned into a commanded rate along n,

    |omega_cmd| = min(sqrt(2 a |phi|), k_p |phi|),   a = f tau_max / (n' J n),

a deceleration-limited approach (sqrt) outside |phi| = 2a/k_p^2 and a linear
approach inside it; a rate loop with gain K_omega tracks it. The sqrt branch is
the constant-deceleration braking profile of a double integrator, so the
commanded rate never asks for more stopping torque than the fraction f of the
torque limit provides. It needs a nominal inertia (it is not adapted online).
An optional profile_max_rate caps the commanded rate (the slew-rate constraint
of the rate-limited eigenaxis laws, Wie and Lu, JGCD 1995).

'two_phase' is a mode manager for acquisition from a tumble. In 'detumble' it
ignores the attitude and damps the body momentum with a torque-limited law,

    tau = -J w * min(1 / T_d, tau_max / |J w|),

which is at the limit, opposite the momentum, until |J w| < tau_max T_d and
proportional below. It switches to 'acquisition' (the rate_profile law) when the
rate it is given (the estimated rate in the closed loop) satisfies |w| <= w_entry,
and back to 'detumble' only if |w| > w_exit > w_entry. The gap between the two
thresholds is the hysteresis; w_exit is set above every rate the acquisition
law commands, so a noisy rate near w_entry cannot make it chatter. With
switch_filter_time_constant_s set, both thresholds act on a first-order low-pass
of |w| (s_k = s_{k-1} + a (|w_k| - s_{k-1}), a = dt / (T_f + dt)) instead of the
single sample, so one noisy sample cannot switch. The filter starts at w_exit (a
tumble is assumed until the filtered rate shows otherwise), so not even the first
sample can switch on its own.
"""

import numpy as np
from typing import Optional, Literal, Tuple
from dataclasses import dataclass
from src.dynamics.quaternion import normalize, multiply, conjugate, rotate_vector


@dataclass
class ControllerParams:
    """Attitude controller parameters."""
    # Quaternion feedback gain (N*m per unit quaternion vector error)
    K_q: float = 1.0
    # Rate feedback gain (N*m*s/rad)
    K_omega: float = 10.0
    # Maximum torque command norm (N*m)
    max_torque: float = 0.1
    # Control law type
    control_type: Literal['quaternion_feedback', 'pd', 'rate_profile', 'two_phase'] = 'quaternion_feedback'
    # rate_profile only: nominal inertia (3x3), fraction f of the torque limit
    # used for braking, and the linear-region rate gain k_p (1/s).
    inertia: Optional[np.ndarray] = None
    profile_accel_fraction: float = 0.7
    profile_rate_gain: float = 0.05
    # rate_profile and two_phase: optional cap on the commanded rate (rad/s).
    profile_max_rate: Optional[float] = None
    # two_phase only: switch to acquisition at |w| <= detumble_entry_rate, back
    # to detumble at |w| > detumble_exit_rate (rad/s); detumble time constant (s).
    detumble_entry_rate: float = 0.01
    detumble_exit_rate: float = 0.15
    detumble_time_constant_s: float = 20.0
    # two_phase only: low-pass time constant (s) of the switching signal |w|, and
    # the controller sample time (s). None switches on the raw sample.
    switch_filter_time_constant_s: Optional[float] = None
    control_dt_s: float = 0.01


class AttitudeController:
    """
    Quaternion feedback controller.

    The right-relative attitude correction and body-rate error are expressed in
    current-body coordinates so their feedback terms can be combined directly.
    """

    def __init__(self, params: ControllerParams):
        """
        Initialize attitude controller.

        Args:
            params: ControllerParams
        """
        if params.control_type not in ('quaternion_feedback', 'pd', 'rate_profile', 'two_phase'):
            raise ValueError(f"Unknown control_type: {params.control_type!r}")
        if params.control_type in ('rate_profile', 'two_phase'):
            if params.inertia is None:
                raise ValueError(f"control_type {params.control_type!r} requires a nominal inertia")
            if params.profile_max_rate is not None and params.profile_max_rate <= 0.0:
                raise ValueError("profile_max_rate must be positive")
            if not 0.0 < params.profile_accel_fraction <= 1.0:
                raise ValueError("profile_accel_fraction must be in (0, 1]")
            if params.profile_rate_gain <= 0.0:
                raise ValueError("profile_rate_gain must be positive")
        if params.control_type == 'two_phase':
            if not 0.0 < params.detumble_entry_rate < params.detumble_exit_rate:
                raise ValueError("need 0 < detumble_entry_rate < detumble_exit_rate")
            if params.detumble_time_constant_s <= 0.0:
                raise ValueError("detumble_time_constant_s must be positive")
            if (params.switch_filter_time_constant_s is not None
                    and params.switch_filter_time_constant_s <= 0.0):
                raise ValueError("switch_filter_time_constant_s must be positive")
            if params.control_dt_s <= 0.0:
                raise ValueError("control_dt_s must be positive")
            if (params.profile_max_rate is not None
                    and params.profile_max_rate >= params.detumble_exit_rate):
                raise ValueError("profile_max_rate must be below detumble_exit_rate")
        # Mode state (two_phase only): starts in detumble; switches counts every change.
        self.phase = 'detumble' if params.control_type == 'two_phase' else None
        self.mode_switches = 0
        self.switch_signal: Optional[float] = None
        if params.max_torque <= 0.0:
            raise ValueError("max_torque must be positive")
        self.params = params
        self.K_q = params.K_q
        self.K_omega = params.K_omega
        self.max_torque = params.max_torque

        # Desired state
        self.q_desired = np.array([0.0, 0.0, 0.0, 1.0])
        self.omega_desired = np.zeros(3)

    def set_desired(self, q_desired: np.ndarray, omega_desired: Optional[np.ndarray] = None):
        """Set desired attitude and desired-body-frame angular velocity."""
        self.q_desired = normalize(q_desired)
        if omega_desired is not None:
            self.omega_desired = np.array(omega_desired, dtype=float)
        else:
            self.omega_desired = np.zeros(3)

    def compute_control(self, q_current: np.ndarray, omega_current: np.ndarray) -> np.ndarray:
        """
        Compute the saturated feedback torque command.

        Args:
            q_current: Current body-to-inertial quaternion estimate
            omega_current: Current angular velocity estimate (rad/s)

        Returns:
            Commanded torque (N*m) in body frame
        """
        return self._saturate(self._feedback(q_current, omega_current))

    def compute_control_with_feedforward(
        self,
        q_current: np.ndarray,
        omega_current: np.ndarray,
        inertia: np.ndarray,
        tau_dist: Optional[np.ndarray] = None,
        wheel_momentum: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute feedback plus gyroscopic and disturbance feedforward.

        tau = tau_fb + omega x (J omega + h_w) - tau_dist_hat

        The feedforward cancels the gyroscopic torque of the bus and wheels and
        the estimated environmental disturbance. The sum is saturated once, so
        the feedforward is not lost when the feedback alone is at the limit.

        Args:
            q_current: Current quaternion
            omega_current: Current angular velocity
            inertia: Bus inertia matrix used for feedforward
            tau_dist: Estimated disturbance torque acting on the bus (N*m)
            wheel_momentum: Wheel momentum in body axes (N*m*s)

        Returns:
            Commanded torque
        """
        h_w = np.zeros(3) if wheel_momentum is None else np.asarray(wheel_momentum, dtype=float)
        feedback = self._feedback(q_current, omega_current)  # updates the two_phase mode
        if self.phase == 'detumble':
            # Detumbling already uses the full torque against the momentum; the
            # gyroscopic term only rotates the momentum, so it is not fed forward.
            return self._saturate(feedback)
        tau_ff = np.cross(omega_current, inertia @ omega_current + h_w)
        if tau_dist is not None:
            tau_ff = tau_ff - np.asarray(tau_dist, dtype=float)
        return self._saturate(feedback + tau_ff)

    def attitude_error(self, q_current: np.ndarray) -> np.ndarray:
        """Return the shortest-path body-frame correction quaternion q_c^-1 ⊗ q_d."""
        q_error = normalize(multiply(conjugate(normalize(q_current)), self.q_desired))
        return -q_error if q_error[3] < 0 else q_error

    def _feedback(self, q_current: np.ndarray, omega_current: np.ndarray) -> np.ndarray:
        """Unsaturated feedback torque."""
        q_error = self.attitude_error(q_current)
        # Desired rate is given in the desired body frame; express it in the
        # current body frame before forming the rate error.
        omega_desired_body = rotate_vector(q_error, self.omega_desired)
        if self.params.control_type == 'two_phase':
            self._update_phase(omega_current)
            if self.phase == 'detumble':
                return self._detumble_torque(omega_current)
            return self._rate_profile_feedback(q_error, omega_current, omega_desired_body)
        if self.params.control_type == 'rate_profile':
            return self._rate_profile_feedback(q_error, omega_current, omega_desired_body)
        omega_error = omega_current - omega_desired_body
        stiffness = self.K_q if self.params.control_type == 'quaternion_feedback' else 2.0 * self.K_q
        return quaternion_feedback_torque(q_error, omega_error, stiffness, self.K_omega)

    def _rate_profile_feedback(self, q_error: np.ndarray, omega_current: np.ndarray,
                               omega_desired_body: np.ndarray) -> np.ndarray:
        """Torque tracking the deceleration-limited approach rate (see module doc)."""
        vector = q_error[:3]
        sin_half = np.linalg.norm(vector)
        if sin_half < 1e-12:
            omega_cmd = np.zeros(3)
        else:
            axis = vector / sin_half
            angle = 2.0 * np.arctan2(sin_half, q_error[3])
            inertia = np.asarray(self.params.inertia, dtype=float)
            decel = (self.params.profile_accel_fraction * self.max_torque
                     / float(axis @ inertia @ axis))
            speed = min(np.sqrt(2.0 * decel * angle), self.params.profile_rate_gain * angle)
            if self.params.profile_max_rate is not None:
                speed = min(speed, self.params.profile_max_rate)
            omega_cmd = speed * axis
        return self.K_omega * (omega_cmd + omega_desired_body - omega_current)

    def _update_phase(self, omega_current: np.ndarray) -> None:
        """Hysteresis between detumble and acquisition on the rate the controller is given."""
        rate = float(np.linalg.norm(omega_current))
        tau_f = self.params.switch_filter_time_constant_s
        if tau_f is None:
            self.switch_signal = rate
        else:
            if self.switch_signal is None:
                self.switch_signal = self.params.detumble_exit_rate
            alpha = self.params.control_dt_s / (tau_f + self.params.control_dt_s)
            self.switch_signal += alpha * (rate - self.switch_signal)
        rate = self.switch_signal
        if self.phase == 'detumble' and rate <= self.params.detumble_entry_rate:
            self.phase = 'acquisition'
            self.mode_switches += 1
        elif self.phase == 'acquisition' and rate > self.params.detumble_exit_rate:
            self.phase = 'detumble'
            self.mode_switches += 1

    def _detumble_torque(self, omega_current: np.ndarray) -> np.ndarray:
        """Torque-limited momentum damping: at the limit far out, proportional near zero."""
        momentum = np.asarray(self.params.inertia, dtype=float) @ omega_current
        norm = float(np.linalg.norm(momentum))
        if norm == 0.0:
            return np.zeros(3)
        scale = min(1.0 / self.params.detumble_time_constant_s, self.max_torque / norm)
        return -scale * momentum

    def _saturate(self, tau_cmd: np.ndarray) -> np.ndarray:
        """Limit the command norm while preserving direction."""
        tau_norm = np.linalg.norm(tau_cmd)
        if tau_norm > self.max_torque:
            return tau_cmd * self.max_torque / tau_norm
        return tau_cmd


def compute_control_gains(
    settling_time: float,
    damping_ratio: float,
    inertia: np.ndarray,
    control_type: Literal['quaternion_feedback', 'pd'] = 'pd',
) -> Tuple[float, float]:
    """
    Compute linear-design gains for a desired 2% settling time and damping ratio.

    Small-angle closed loop per axis: J theta_ddot + K_omega theta_dot + k theta = 0
    with effective stiffness k = K_q for 'pd' and k = K_q / 2 for
    'quaternion_feedback' (because q_e,v ≈ theta / 2).

    omega_n = 4 / (damping_ratio * settling_time)   (2% criterion)
    k       = J omega_n^2
    K_omega = 2 damping_ratio J omega_n

    The mean principal inertia is used for J; the result is a starting point,
    not a substitute for checking the actuator saturation limits.

    Args:
        settling_time: Desired settling time (s)
        damping_ratio: Desired damping ratio
        inertia: Inertia matrix (diagonal elements used)
        control_type: Control law the gains are intended for

    Returns:
        (K_q, K_omega) gains
    """
    if settling_time <= 0.0 or damping_ratio <= 0.0:
        raise ValueError("settling_time and damping_ratio must be positive")
    omega_n = 4.0 / (damping_ratio * settling_time)
    I_avg = float(np.mean(np.diag(inertia)))
    stiffness = I_avg * omega_n**2
    K_q = 2.0 * stiffness if control_type == 'quaternion_feedback' else stiffness
    K_omega = 2.0 * damping_ratio * I_avg * omega_n
    return K_q, K_omega


def quaternion_feedback_torque(q_error: np.ndarray, omega_error: np.ndarray,
                               K_q: float, K_omega: float) -> np.ndarray:
    """
    Compute quaternion feedback torque tau = K_q q_e,v - K_omega omega_e.

    Args:
        q_error: Correction quaternion q_current^-1 ⊗ q_desired (vector part used)
        omega_error: Angular velocity error omega - omega_desired (rad/s)
        K_q: Quaternion gain
        K_omega: Rate gain

    Returns:
        Torque command (unsaturated)
    """
    return K_q * q_error[:3] - K_omega * omega_error
