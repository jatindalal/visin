from enum import Enum

import numpy as np

from visin.core.math import MatrixUtils


class Camera:
    class ProjectionMode(Enum):
        Perspective = 0
        Orthographic = 1

    def __init__(
        self,
        position=(0.0, 0.0, 50.0),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov=45.0,
        near=0.0001,
        far=100000.0,
    ):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.world_up = MatrixUtils._normalize(np.array(up, dtype=np.float32))
        self.up = self.world_up.copy()

        self.fov = float(fov)
        self.near = float(near)
        self.far = float(far)
        self.aspect_ratio = 16 / 9

        self.ortho_zoom = 50.0
        self.projection_mode = self.ProjectionMode.Perspective

        self.move_speed = 3.0
        self.orbit_speed = 1.0
        self.pan_speed = 1.5
        self.min_distance = 0.1
        self.max_distance = 1000.0

        self.forward = np.zeros(3, dtype=np.float32)
        self.right = np.zeros(3, dtype=np.float32)
        self.distance = 1.0

        self._update_orientation()

    def set_aspect_ratio(self, width, height):
        self.aspect_ratio = width / height if height > 0 else 16 / 9

    def toggle_projection(self):
        self.projection_mode = (
            self.ProjectionMode.Orthographic
            if self.projection_mode == self.ProjectionMode.Perspective
            else self.ProjectionMode.Perspective
        )

    def arcball_rotate(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        viewport_width: float,
        viewport_height: float,
    ):
        start = self._map_to_arcball(start_x, start_y, viewport_width, viewport_height)
        end = self._map_to_arcball(end_x, end_y, viewport_width, viewport_height)
        axis_view = np.cross(end, start)
        axis_length = np.linalg.norm(axis_view)
        if axis_length < 1e-6:
            return

        dot = np.clip(np.dot(start, end), -1.0, 1.0)
        angle = np.arccos(dot) * self.orbit_speed
        axis_view = axis_view / axis_length
        axis_world = MatrixUtils._normalize(
            self.right * axis_view[0]
            + self.up * axis_view[1]
            - self.forward * axis_view[2]
        )
        if np.linalg.norm(axis_world) < 1e-6:
            return

        offset = self.position - self.target
        self.position = self.target + self._rotate_vector(offset, axis_world, angle)
        self.up = self._rotate_vector(self.up, axis_world, angle)
        self._update_orientation()

    def pan(self, dx: float, dy: float, viewport_width: float, viewport_height: float):
        units_per_pixel_x, units_per_pixel_y = self._world_units_per_pixel(
            viewport_width, viewport_height
        )
        offset = (-self.right * dx * units_per_pixel_x) + (
            self.up * dy * units_per_pixel_y
        )
        offset *= self.pan_speed
        self.position += offset
        self.target += offset
        self._update_orientation()

    def zoom(self, offset: float):
        zoom_scale = np.power(0.9, offset)
        if self.projection_mode == self.ProjectionMode.Perspective:
            distance = np.clip(
                self.distance * zoom_scale,
                self.min_distance,
                self.max_distance,
            )
            view_direction = MatrixUtils._normalize(self.position - self.target)
            if np.linalg.norm(view_direction) < 1e-6:
                view_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            self.position = self.target + view_direction * distance
            self._update_orientation()
            return

        self.ortho_zoom = np.clip(self.ortho_zoom * zoom_scale, 0.1, 1000.0)

    def move(self, delta_time: float, move_forward: float = 0.0, move_right: float = 0.0):
        move_delta = self.move_speed * delta_time
        translation = (
            self.forward * (move_forward * move_delta)
            + self.right * (move_right * move_delta)
        )
        self.position += translation
        self.target += translation
        self._update_orientation()

    def get_view_matrix(self):
        return MatrixUtils.look_at(self.position, self.target, self.up)

    def get_projection_matrix(self):
        if self.projection_mode == self.ProjectionMode.Perspective:
            return MatrixUtils.perspective_projection(
                self.fov, self.aspect_ratio, self.near, self.far
            )

        half_zoom = self.ortho_zoom * 0.5
        return MatrixUtils.orthographic_projection(
            left=-half_zoom * self.aspect_ratio,
            right=half_zoom * self.aspect_ratio,
            bottom=-half_zoom,
            top=half_zoom,
            near=self.near,
            far=self.far,
        )

    def get_mvp(self, model=None):
        projection = self.get_projection_matrix()
        view = self.get_view_matrix()
        if model is None:
            return MatrixUtils.create_mvp(projection, view)

        return MatrixUtils.create_mvp(projection, view, model)

    def _update_orientation(self):
        self.forward = MatrixUtils._normalize(self.target - self.position).astype(
            np.float32
        )
        right = np.cross(self.forward, self.up)
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(self.forward, self.world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        self.right = MatrixUtils._normalize(right).astype(np.float32)
        self.up = MatrixUtils._normalize(np.cross(self.right, self.forward)).astype(
            np.float32
        )
        self.distance = max(
            float(np.linalg.norm(self.position - self.target)), self.min_distance
        )

    def _map_to_arcball(
        self, x: float, y: float, viewport_width: float, viewport_height: float
    ) -> np.ndarray:
        viewport_width = max(float(viewport_width), 1.0)
        viewport_height = max(float(viewport_height), 1.0)

        point = np.array(
            [
                (2.0 * x - viewport_width) / viewport_width,
                (viewport_height - 2.0 * y) / viewport_height,
            ],
            dtype=np.float32,
        )
        length_sq = float(np.dot(point, point))
        if length_sq <= 1.0:
            z = np.sqrt(1.0 - length_sq)
            return np.array([point[0], point[1], z], dtype=np.float32)

        point /= np.sqrt(length_sq)
        return np.array([point[0], point[1], 0.0], dtype=np.float32)

    def _rotate_vector(
        self, vector: np.ndarray, axis: np.ndarray, angle_radians: float
    ) -> np.ndarray:
        axis = MatrixUtils._normalize(np.asarray(axis, dtype=np.float32))
        vector = np.asarray(vector, dtype=np.float32)
        cos_theta = np.cos(angle_radians)
        sin_theta = np.sin(angle_radians)
        return (
            vector * cos_theta
            + np.cross(axis, vector) * sin_theta
            + axis * np.dot(axis, vector) * (1.0 - cos_theta)
        ).astype(np.float32)

    def _world_units_per_pixel(
        self, viewport_width: float, viewport_height: float
    ) -> tuple[float, float]:
        viewport_width = max(float(viewport_width), 1.0)
        viewport_height = max(float(viewport_height), 1.0)
        if self.projection_mode == self.ProjectionMode.Orthographic:
            visible_height = self.ortho_zoom
            visible_width = visible_height * self.aspect_ratio
            return visible_width / viewport_width, visible_height / viewport_height

        visible_height = 2.0 * self.distance * np.tan(np.radians(self.fov) * 0.5)
        visible_width = visible_height * self.aspect_ratio
        return visible_width / viewport_width, visible_height / viewport_height


class CameraController:
    class Interaction(Enum):
        Idle = 0
        Orbit = 1
        Pan = 2

    def __init__(self, camera: Camera):
        self.camera = camera
        self.interaction = self.Interaction.Idle
        self.last_pointer = None

    def set_interaction(self, interaction: Interaction, x: float, y: float):
        if interaction == self.Interaction.Idle:
            self.end_interaction()
            return

        if self.interaction != interaction or self.last_pointer is None:
            self.interaction = interaction
            self.last_pointer = (x, y)

    def end_interaction(self):
        self.interaction = self.Interaction.Idle
        self.last_pointer = None

    def drag_to(self, x: float, y: float, viewport_width: float, viewport_height: float):
        if self.interaction == self.Interaction.Idle or self.last_pointer is None:
            self.last_pointer = (x, y)
            return

        last_x, last_y = self.last_pointer
        dx = x - last_x
        dy = y - last_y
        self.last_pointer = (x, y)

        if self.interaction == self.Interaction.Pan:
            self.camera.pan(dx, dy, viewport_width, viewport_height)
            return

        self.camera.arcball_rotate(
            last_x,
            last_y,
            x,
            y,
            viewport_width,
            viewport_height,
        )

    def zoom(self, offset: float):
        self.camera.zoom(offset)
