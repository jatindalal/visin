import numpy as np

class MatrixUtils:
    """
    Utility class for generating Model-View-Projection matrices.
    Uses only numpy, no external dependencies.
    """

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        """Helper function to normalize a vector."""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    @staticmethod
    def perspective_projection(fov_y_degrees: float, aspect_ratio: float, 
                               near: float, far: float) -> np.ndarray:
        """
        Create a perspective projection matrix.
        """
        fov_y_rad = np.radians(fov_y_degrees)
        f = 1.0 / np.tan(fov_y_rad / 2.0)
        matrix = np.zeros((4, 4), dtype=np.float32)
        matrix[0, 0] = f / aspect_ratio
        matrix[1, 1] = f
        matrix[2, 2] = (far + near) / (near - far)
        matrix[2, 3] = (2 * far * near) / (near - far)
        matrix[3, 2] = -1.0
        return matrix

    @staticmethod
    def orthographic_projection(left: float, right: float, 
                                bottom: float, top: float, 
                                near: float, far: float) -> np.ndarray:
        """
        Create an orthographic projection matrix.
        """
        matrix = np.zeros((4, 4), dtype=np.float32)
        matrix[0, 0] = 2.0 / (right - left)
        matrix[1, 1] = 2.0 / (top - bottom)
        matrix[2, 2] = -2.0 / (far - near)
        matrix[0, 3] = -(right + left) / (right - left)
        matrix[1, 3] = -(top + bottom) / (top - bottom)
        matrix[2, 3] = -(far + near) / (far - near)
        matrix[3, 3] = 1.0
        return matrix

    @staticmethod
    def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """
        Create a view matrix using lookAt.
        """
        eye = np.asarray(eye, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)

        # Normalize axes manually
        z_axis = MatrixUtils._normalize(eye - target)
        x_axis = MatrixUtils._normalize(np.cross(up, z_axis))
        y_axis = np.cross(z_axis, x_axis)  # Already normalized if x and z are orthogonal

        matrix = np.eye(4, dtype=np.float32)
        matrix[0, :3] = x_axis
        matrix[1, :3] = y_axis
        matrix[2, :3] = z_axis
        matrix[0, 3] = -np.dot(x_axis, eye)
        matrix[1, 3] = -np.dot(y_axis, eye)
        matrix[2, 3] = -np.dot(z_axis, eye)
        return matrix

    @staticmethod
    def translate(x: float, y: float, z: float) -> np.ndarray:
        """Create a translation matrix."""
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 3] = x
        matrix[1, 3] = y
        matrix[2, 3] = z
        return matrix

    @staticmethod
    def rotate_x(angle_degrees: float) -> np.ndarray:
        """Create rotation matrix around X-axis."""
        angle = np.radians(angle_degrees)
        c, s = np.cos(angle), np.sin(angle)
        matrix = np.eye(4, dtype=np.float32)
        matrix[1, 1] = c
        matrix[1, 2] = -s
        matrix[2, 1] = s
        matrix[2, 2] = c
        return matrix

    @staticmethod
    def rotate_y(angle_degrees: float) -> np.ndarray:
        """Create rotation matrix around Y-axis."""
        angle = np.radians(angle_degrees)
        c, s = np.cos(angle), np.sin(angle)
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0] = c
        matrix[0, 2] = s
        matrix[2, 0] = -s
        matrix[2, 2] = c
        return matrix

    @staticmethod
    def rotate_z(angle_degrees: float) -> np.ndarray:
        """Create rotation matrix around Z-axis."""
        angle = np.radians(angle_degrees)
        c, s = np.cos(angle), np.sin(angle)
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0] = c
        matrix[0, 1] = -s
        matrix[1, 0] = s
        matrix[1, 1] = c
        return matrix

    @staticmethod
    def scale(x: float, y: float, z: float) -> np.ndarray:
        """Create a scaling matrix."""
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0] = x
        matrix[1, 1] = y
        matrix[2, 2] = z
        return matrix

    @staticmethod
    def create_mvp(projection: np.ndarray, view: np.ndarray, 
                   model: np.ndarray = None) -> np.ndarray:
        """
        Combine projection, view, and optional model matrices.
        """
        if model is None:
            model = np.eye(4, dtype=np.float32)
        mvp = projection @ view @ model
        return mvp.astype(np.float32)
