"""
Ego vehicle localization module for close-loop evaluation.

Provides multiple strategies to estimate ego pose (position + yaw) from
sensor data (GNSS, IMU, speedometer), without using simulator GT.

Strategies:
  - 'raw':          Raw GPS converted to CARLA world XY.  (~0.7m mean error)
  - 'dead_reckoning': Speed + compass integration, reset by GPS periodically.
  - 'complementary': Complementary filter blending dead-reckoning (high-freq)
                     with GPS (low-freq drift correction).  (recommended)
  - 'ukf_tuned':    Re-tuned UKF that trusts GPS more than bicycle model.

All strategies return (position_xy, yaw) in CARLA world coordinates.

Usage in agent:
    from team_code.ego_localizer import EgoLocalizer

    # In setup():
    self.localizer = EgoLocalizer(strategy='complementary', dt=1/20)

    # In tick():
    gps_xy = ...   # raw GPS converted to CARLA XY
    compass = ...  # preprocessed compass (CARLA yaw)
    speed = ...    # from speedometer
    imu = input_data['IMU'][1]  # [ax, ay, az, gx, gy, gz, compass_raw]

    ego_xy, ego_yaw = self.localizer.update(
        gps_xy=gps_xy,
        compass=compass,
        speed=speed,
        steer=self.control.steer,
        throttle=self.control.throttle,
        brake=float(self.control.brake > 0.5),
        imu_data=imu,  # optional, used by ukf_tuned
    )
"""

import math
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF


def _normalize_angle(x):
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi
    return x


# ---------------------------------------------------------------------------
# Bicycle model (same as ukf_utils.py, inlined to avoid import issues)
# ---------------------------------------------------------------------------
def _bicycle_model_forward(x, dt, steer, throttle, brake):
    front_wb = -0.090769015
    rear_wb = 1.4178275
    steer_gain = 0.36848336
    brake_accel = -4.952399
    throt_accel = 0.5633837

    locs_0 = float(x[0])
    locs_1 = float(x[1])
    yaw = x[2]
    speed = x[3]

    accel = brake_accel if brake else throt_accel * throttle
    wheel = steer_gain * steer
    beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))

    next_locs_0 = locs_0 + speed * math.cos(yaw + beta) * dt
    next_locs_1 = locs_1 + speed * math.sin(yaw + beta) * dt
    next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
    next_speed = max(speed + accel * dt, 0.0)

    return np.array([next_locs_0, next_locs_1, next_yaws, next_speed])


# ---------------------------------------------------------------------------
# UKF helpers
# ---------------------------------------------------------------------------
def _hx(x):
    return x

def _state_mean(state, wm):
    x = np.zeros(4)
    x[0] = np.dot(state[:, 0], wm)
    x[1] = np.dot(state[:, 1], wm)
    x[2] = math.atan2(np.dot(np.sin(state[:, 2]), wm),
                       np.dot(np.cos(state[:, 2]), wm))
    x[3] = np.dot(state[:, 3], wm)
    return x

def _residual(a, b):
    y = a - b
    y[2] = _normalize_angle(y[2])
    return y


class EgoLocalizer:
    """
    Fuses GPS, compass, speed, and optionally IMU to produce a smooth,
    accurate ego pose estimate.
    """

    STRATEGIES = ('raw', 'dead_reckoning', 'complementary', 'ukf_tuned',
                  'ema', 'ema_dr')

    def __init__(self, strategy='complementary', dt=0.05,
                 gps_alpha=0.15, latency_compensation=True,
                 ema_window=20):
        """
        Args:
            strategy: one of STRATEGIES
            dt: expected time step (1/fps)
            gps_alpha: blending weight for GPS in complementary filter.
                       Higher = trust GPS more (range 0-1).
            latency_compensation: if True, extrapolate GPS by 1 frame using
                                  speed+heading to compensate sensor delay.
            ema_window: window size for EMA GPS smoothing (used by 'ema' and
                        'ema_dr'). Effective alpha = 2/(window+1).
        """
        assert strategy in self.STRATEGIES, f"Unknown strategy: {strategy}"
        self.strategy = strategy
        self.dt = dt
        self.gps_alpha = gps_alpha
        self.latency_compensation = latency_compensation

        # Internal state
        self.initialized = False
        self.pos = np.zeros(2, dtype=np.float64)  # estimated [x, y]
        self.yaw = 0.0
        self.step = 0

        # For dead_reckoning: periodic GPS reset
        self.dr_reset_interval = 20  # reset every N steps to prevent drift
        self.dr_step_since_reset = 0

        # For EMA-based strategies
        self.ema_alpha = 2.0 / (ema_window + 1)  # EMA smoothing factor
        self.gps_ema = None  # smoothed GPS position

        # For ukf_tuned
        self.ukf = None
        if strategy == 'ukf_tuned':
            self._init_ukf(dt)

    def _init_ukf(self, dt):
        points = MerweScaledSigmaPoints(
            n=4, alpha=0.001, beta=2, kappa=0,
            subtract=_residual
        )
        self.ukf = UKF(
            dim_x=4, dim_z=4,
            fx=_bicycle_model_forward,
            hx=_hx,
            dt=dt,
            points=points,
            x_mean_fn=_state_mean,
            z_mean_fn=_state_mean,
            residual_x=_residual,
            residual_z=_residual,
        )
        # --- Key difference from original: trust GPS much more ---
        # GPS noise in CARLA: stddev ~0.55m in XY after conversion
        # Compass: near-perfect.  Speed: near-perfect.
        self.ukf.P = np.diag([0.3, 0.3, 0.001, 0.01])
        self.ukf.R = np.diag([0.05, 0.05, 1e-10, 1e-10])  # trust GPS (was 0.5!)
        self.ukf.Q = np.diag([0.3, 0.3, 0.005, 0.01])     # don't trust bicycle model much

    def _compensate_latency(self, gps_xy, speed, yaw):
        """Extrapolate GPS forward by 1 frame to compensate sensor delay."""
        if not self.latency_compensation or speed < 0.5:
            return gps_xy
        dx = speed * self.dt * math.cos(yaw)
        dy = speed * self.dt * math.sin(yaw)
        return gps_xy + np.array([dx, dy], dtype=np.float64)

    def update(self, gps_xy, compass, speed,
               steer=0.0, throttle=0.0, brake=0.0,
               imu_data=None):
        """
        Update ego pose estimate with new sensor readings.

        Args:
            gps_xy: np.array([x, y]) — raw GPS in CARLA world coords
            compass: float — preprocessed yaw in CARLA coords (rad)
            speed: float — from speedometer (m/s)
            steer/throttle/brake: control inputs (for UKF prediction)
            imu_data: optional np.array(7) from IMU sensor

        Returns:
            (position_xy, yaw): estimated ego pose
                position_xy: np.array([x, y], dtype=float64)
                yaw: float (rad)
        """
        gps_xy = np.asarray(gps_xy, dtype=np.float64)
        yaw = float(_normalize_angle(compass))

        if self.strategy == 'raw':
            return self._update_raw(gps_xy, yaw, speed)
        elif self.strategy == 'dead_reckoning':
            return self._update_dead_reckoning(gps_xy, yaw, speed)
        elif self.strategy == 'complementary':
            return self._update_complementary(gps_xy, yaw, speed)
        elif self.strategy == 'ema':
            return self._update_ema(gps_xy, yaw, speed)
        elif self.strategy == 'ema_dr':
            return self._update_ema_dr(gps_xy, yaw, speed)
        elif self.strategy == 'ukf_tuned':
            return self._update_ukf(gps_xy, yaw, speed, steer, throttle, brake)

    def _update_raw(self, gps_xy, yaw, speed):
        """Just pass through raw GPS with optional latency compensation."""
        pos = self._compensate_latency(gps_xy, speed, yaw)
        self.pos = pos
        self.yaw = yaw
        self.step += 1
        return pos.copy(), yaw

    def _update_dead_reckoning(self, gps_xy, yaw, speed):
        """
        Dead reckoning using speed + compass.
        Periodically resets to GPS to prevent unbounded drift.
        """
        if not self.initialized:
            self.pos = self._compensate_latency(gps_xy, speed, yaw)
            self.yaw = yaw
            self.initialized = True
            self.dr_step_since_reset = 0
            self.step += 1
            return self.pos.copy(), self.yaw

        # Integrate: pos += speed * dt * [cos(yaw), sin(yaw)]
        # Use current compass for heading (essentially noiseless)
        dx = speed * self.dt * math.cos(yaw)
        dy = speed * self.dt * math.sin(yaw)
        self.pos += np.array([dx, dy], dtype=np.float64)
        self.yaw = yaw
        self.dr_step_since_reset += 1

        # Periodic GPS reset
        if self.dr_step_since_reset >= self.dr_reset_interval:
            self.pos = self._compensate_latency(gps_xy, speed, yaw)
            self.dr_step_since_reset = 0

        self.step += 1
        return self.pos.copy(), self.yaw

    def _update_complementary(self, gps_xy, yaw, speed):
        """
        Complementary filter:
          pos = (1 - alpha) * dead_reckoning_pos + alpha * gps_compensated

        Dead reckoning is accurate short-term (speed+compass are near-perfect).
        GPS corrects long-term drift.
        alpha controls the blend — higher trusts GPS more.
        """
        gps_compensated = self._compensate_latency(gps_xy, speed, yaw)

        if not self.initialized:
            self.pos = gps_compensated.copy()
            self.yaw = yaw
            self.initialized = True
            self.step += 1
            return self.pos.copy(), self.yaw

        # Dead reckoning step
        dx = speed * self.dt * math.cos(yaw)
        dy = speed * self.dt * math.sin(yaw)
        dr_pos = self.pos + np.array([dx, dy], dtype=np.float64)

        # Blend
        alpha = self.gps_alpha
        self.pos = (1.0 - alpha) * dr_pos + alpha * gps_compensated
        self.yaw = yaw  # compass is near-perfect, use directly

        self.step += 1
        return self.pos.copy(), self.yaw

    def _update_ema(self, gps_xy, yaw, speed):
        """
        Exponential moving average on raw GPS, then extrapolate forward
        using speed + compass to compensate the smoothing lag.

        EMA removes high-freq noise (stddev shrinks by ~sqrt(alpha/2))
        but introduces lag proportional to (1-alpha)/alpha frames.
        The extrapolation step cancels this lag using speed + heading.
        """
        if self.gps_ema is None:
            self.gps_ema = gps_xy.copy()

        # Update EMA
        a = self.ema_alpha
        self.gps_ema = a * gps_xy + (1 - a) * self.gps_ema

        # EMA lag in frames ≈ (1-a)/a, compensate by extrapolating forward
        lag_frames = (1.0 - a) / a
        lag_dt = lag_frames * self.dt
        dx = speed * lag_dt * math.cos(yaw)
        dy = speed * lag_dt * math.sin(yaw)
        self.pos = self.gps_ema + np.array([dx, dy], dtype=np.float64)
        self.yaw = yaw

        self.step += 1
        return self.pos.copy(), self.yaw

    def _update_ema_dr(self, gps_xy, yaw, speed):
        """
        Dead reckoning with EMA-smoothed GPS anchor correction.

        1. Dead-reckon from last position using speed + compass (near-perfect).
        2. EMA-smooth the GPS stream (removes noise).
        3. Blend: tiny alpha pulls DR toward smoothed GPS to prevent drift.

        This combines the best of both:
        - DR gives zero-noise frame-to-frame motion
        - EMA GPS gives a low-noise absolute reference
        """
        if self.gps_ema is None:
            self.gps_ema = gps_xy.copy()

        # Update GPS EMA
        ea = self.ema_alpha
        self.gps_ema = ea * gps_xy + (1 - ea) * self.gps_ema

        if not self.initialized:
            self.pos = gps_xy.copy()
            self.yaw = yaw
            self.initialized = True
            self.step += 1
            return self.pos.copy(), self.yaw

        # Dead reckoning step
        dx = speed * self.dt * math.cos(yaw)
        dy = speed * self.dt * math.sin(yaw)
        dr_pos = self.pos + np.array([dx, dy], dtype=np.float64)

        # Pull toward EMA-smoothed GPS with small alpha
        alpha = self.gps_alpha
        self.pos = (1.0 - alpha) * dr_pos + alpha * self.gps_ema
        self.yaw = yaw

        self.step += 1
        return self.pos.copy(), self.yaw

    def _update_ukf(self, gps_xy, yaw, speed, steer, throttle, brake):
        """UKF with re-tuned parameters that trust GPS more."""
        gps_compensated = self._compensate_latency(gps_xy, speed, yaw)

        if not self.initialized:
            self.ukf.x = np.array([
                gps_compensated[0], gps_compensated[1],
                yaw, speed
            ])
            self.initialized = True
            self.step += 1
            self.pos = gps_compensated.copy()
            self.yaw = yaw
            return self.pos.copy(), self.yaw

        self.ukf.predict(steer=steer, throttle=throttle, brake=brake)
        z = np.array([gps_compensated[0], gps_compensated[1], yaw, speed])
        self.ukf.update(z)

        self.pos = self.ukf.x[0:2].copy()
        self.yaw = _normalize_angle(self.ukf.x[2])
        self.step += 1
        return self.pos.copy(), self.yaw

    def get_debug_info(self):
        """Return dict of internal state for logging to meta."""
        info = {
            'localizer_strategy': self.strategy,
            'localizer_pos': self.pos.tolist(),
            'localizer_yaw': float(self.yaw),
            'localizer_step': self.step,
        }
        if self.strategy == 'ukf_tuned' and self.ukf is not None:
            info['localizer_ukf_P_diag'] = np.diag(self.ukf.P).tolist()
        return info
