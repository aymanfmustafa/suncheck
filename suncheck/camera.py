"""Pinhole camera in the world: pixels to sky directions and back, given the camera's heading, pitch and roll.

World frame: East, North, Up. Camera frame: x right, y down, z forward (the optical axis). Yaw is the compass heading of the
optical axis (clockwise from north), pitch its elevation above the horizontal, roll the rotation about it (positive =
clockwise as seen by the photographer).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FULL_FRAME_DIAGONAL_MM = 43.27


def f_px_from_f35(f35_mm: float, width: int, height: int) -> float:
    return f35_mm * float(np.hypot(width, height)) / FULL_FRAME_DIAGONAL_MM


def f_px_for_frame(f35_mm: float, width: int, height: int, sensor_aspect: float = 4.0 / 3.0) -> float:
    """Focal length in pixels for a frame that may be a crop of the 4:3 sensor the 35 mm-equivalent refers to.

    A 4:3 frame is the whole sensor, so the 35 mm diagonal applies. A wider frame (16:9 video, a camera stream in the
    browser) keeps the sensor's full width and crops its height, so its long side spans the sensor's 35 mm-equivalent
    width (34.62 mm for a 4:3 sensor).
    """
    long, short = max(width, height), min(width, height)
    if long / short > sensor_aspect + 0.02:
        sensor_width_mm = FULL_FRAME_DIAGONAL_MM * sensor_aspect / float(np.hypot(sensor_aspect, 1.0))
        return f35_mm * long / sensor_width_mm
    return f_px_from_f35(f35_mm, width, height)


def enu_from_azel(az_deg, el_deg) -> np.ndarray:
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], axis=-1)


def azel_from_enu(d: np.ndarray):
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    az = np.degrees(np.arctan2(d[..., 0], d[..., 1])) % 360.0
    el = np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1)))
    return az, el


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_world_from_camera(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """R such that d_world = R @ d_camera.

    Built from a reference camera looking north and level (x->East, y->Down, z->North), then rolled about its own axis,
    pitched up about its own x axis, and yawed clockwise about Up.
    """
    base = np.array([[1.0, 0.0, 0.0],   # camera x -> East
                     [0.0, 0.0, 1.0],   # camera z -> North
                     [0.0, -1.0, 0.0]])  # camera y -> -Up
    # roll about camera z (positive clockwise for the photographer; image y is down, so a positive angle in the x-y plane
    # with y down is clockwise), pitch about camera x (positive tilts the axis up: moves z toward -y), yaw about world Up.
    roll = _rot_z(np.radians(roll_deg))
    pitch = _rot_x(np.radians(pitch_deg))
    cam = pitch @ roll
    yaw = np.radians(yaw_deg)
    yaw_world = np.array([[np.cos(yaw), np.sin(yaw), 0.0], [-np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    return yaw_world @ base @ cam


@dataclass
class CameraPose:
    f_px: float
    width: int
    height: int
    yaw: float
    pitch: float
    roll: float = 0.0

    @property
    def cx(self):
        return self.width / 2.0

    @property
    def cy(self):
        return self.height / 2.0

    @property
    def R(self) -> np.ndarray:
        return rotation_world_from_camera(self.yaw, self.pitch, self.roll)

    def pixel_to_azel(self, u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        d_cam = np.stack([(u - self.cx) / self.f_px, (v - self.cy) / self.f_px, np.ones_like(u)], axis=-1)
        d_world = d_cam @ self.R.T
        return azel_from_enu(d_world)

    def azel_to_pixel(self, az, el):
        """Pixel coordinates (u, v) and a visibility mask (in front of the camera, inside the frame)."""
        d_world = enu_from_azel(np.asarray(az, dtype=float), np.asarray(el, dtype=float))
        d_cam = d_world @ self.R
        z = d_cam[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx + self.f_px * d_cam[..., 0] / z
            v = self.cy + self.f_px * d_cam[..., 1] / z
        visible = (z > 1e-6) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return u, v, visible

    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.width / (2 * self.f_px))))

    def vfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.height / (2 * self.f_px))))


def solve_yaw_pitch_from_sun(f_px: float, width: int, height: int, sun_uv: tuple[float, float], sun_az: float,
                             sun_el: float, roll: float = 0.0, yaw0: float | None = None, pitch0: float | None = None,
                             iters: int = 30) -> CameraPose:
    """The camera's heading and pitch from where the sun appears in the frame, given its true direction at that time.

    Two unknowns, two constraints; solved by Gauss-Newton on the angular error with finite differences.
    """
    u, v = sun_uv
    pose = CameraPose(f_px, width, height, yaw0 if yaw0 is not None else sun_az, pitch0 if pitch0 is not None else 0.0, roll)
    target = enu_from_azel(sun_az, sun_el)

    def err(yaw, pitch):
        p = CameraPose(f_px, width, height, yaw, pitch, roll)
        az, el = p.pixel_to_azel(u, v)
        d = enu_from_azel(az, el)
        return np.degrees(np.arccos(np.clip(float(d @ target), -1, 1)))

    def residual(yaw, pitch):
        p = CameraPose(f_px, width, height, yaw, pitch, roll)
        az, el = p.pixel_to_azel(u, v)
        daz = (float(az) - sun_az + 180) % 360 - 180
        return np.array([daz * np.cos(np.radians(sun_el)), float(el) - sun_el])

    yaw, pitch = pose.yaw, pose.pitch
    for _ in range(iters):
        r = residual(yaw, pitch)
        if np.linalg.norm(r) < 1e-4:
            break
        h = 1e-3
        J = np.stack([(residual(yaw + h, pitch) - r) / h, (residual(yaw, pitch + h) - r) / h], axis=1)
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        yaw = (yaw + float(step[0])) % 360.0
        pitch = float(np.clip(pitch + step[1], -89.0, 89.0))
    out = CameraPose(f_px, width, height, yaw, pitch, roll)
    out.sun_error_deg = err(yaw, pitch)  # type: ignore[attr-defined]
    return out


def solve_yaw_f_from_sun(f0_px: float, width: int, height: int, sun_uv: tuple[float, float], sun_az: float,
                         sun_el: float, pitch: float, roll: float, yaw0: float | None = None, iters: int = 40) -> CameraPose:
    """Heading and focal length from the sun, when the tilt (pitch and roll) is known from the phone's accelerometer.

    The sun's two image coordinates fix the two unknowns; the lens is constrained by how far off-axis the sun sits, so
    callers should only trust the focal length when the sun is well away from the frame centre.
    """
    u, v = sun_uv
    target = enu_from_azel(sun_az, sun_el)

    def residual(yaw, logf):
        p = CameraPose(float(np.exp(logf)), width, height, yaw, pitch, roll)
        az, el = p.pixel_to_azel(u, v)
        daz = (float(az) - sun_az + 180) % 360 - 180
        return np.array([daz * np.cos(np.radians(sun_el)), float(el) - sun_el])

    yaw = yaw0 if yaw0 is not None else sun_az
    logf = float(np.log(f0_px))
    lo, hi = np.log(f0_px / 2.0), np.log(f0_px * 2.0)
    for _ in range(iters):
        r = residual(yaw, logf)
        if np.linalg.norm(r) < 1e-4:
            break
        h = 1e-3
        J = np.stack([(residual(yaw + h, logf) - r) / h, (residual(yaw, logf + h) - r) / h], axis=1)
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        yaw = (yaw + float(step[0])) % 360.0
        logf = float(np.clip(logf + step[1], lo, hi))
    out = CameraPose(float(np.exp(logf)), width, height, yaw, pitch, roll)
    az, el = out.pixel_to_azel(u, v)
    out.sun_error_deg = float(np.degrees(np.arccos(np.clip(float(enu_from_azel(float(az), float(el)) @ target), -1, 1))))  # type: ignore[attr-defined]
    return out


def yaw_pitch_roll_from_axes(axis: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    """Our heading, pitch and roll from the optical axis and the image's right-hand direction, both in East-North-Up."""
    up = np.array([0.0, 0.0, 1.0])
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    yaw = float(np.degrees(np.arctan2(axis[0], axis[1])) % 360.0)
    pitch = float(np.degrees(np.arcsin(np.clip(axis[2], -1.0, 1.0))))
    right0 = np.cross(axis, up)
    n = np.linalg.norm(right0)
    if n < 1e-9:  # looking straight up or down: roll is undefined
        return yaw, pitch, 0.0
    right0 /= n
    down0 = np.cross(axis, right0)
    right = np.asarray(right, dtype=float)
    roll = float(np.degrees(np.arctan2(right @ down0, right @ right0)))
    return yaw, pitch, roll


def pose_from_device_orientation(alpha_deg: float, beta_deg: float, gamma_deg: float, screen_angle_deg: float = 0.0):
    """Heading, pitch and roll of the back camera from the W3C deviceorientation angles.

    alpha must be absolute (on iOS, alpha = 360 - webkitCompassHeading); beta and gamma come straight from the event.
    The device frame is x to the right, y to the top of the phone, z out of the screen; the back camera looks along -z.
    The matrix is the specification's Z-X'-Y'' composition, device to Earth (East, North, Up). screen_angle_deg is
    window.orientation (0 portrait, 90 with the phone turned counter-clockwise, -90 clockwise), which says where the
    frame's right-hand edge lies on the phone's body.
    """
    a, b, g = np.radians(alpha_deg), np.radians(beta_deg), np.radians(gamma_deg)
    R = _rot_z(a) @ _rot_x(b) @ _rot_y(g)
    s = np.radians(screen_angle_deg)
    axis = R @ np.array([0.0, 0.0, -1.0])
    right = R @ np.array([np.cos(s), -np.sin(s), 0.0])
    return yaw_pitch_roll_from_axes(axis, right)
