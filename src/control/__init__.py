"""
Control module for spacecraft attitude control.
"""

from .attitude_controller import AttitudeController, ControllerParams, compute_control_gains

__all__ = ['AttitudeController', 'ControllerParams', 'compute_control_gains']