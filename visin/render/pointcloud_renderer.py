import moderngl
import numpy as np

from visin.core.math import MatrixUtils

VERTEX_SHADER = """

#version 330
uniform mat4 mvp;
uniform float pointsize;
in vec3 in_vertex;

void main() {
    gl_Position = mvp * vec4(in_vertex, 1.0);
    gl_PointSize = pointsize;
}

"""

FRAGMENT_SHADER = """

#version 330
uniform vec4 color;
out vec4 f_color;
void main() {
    f_color = color;
}

"""


class PointCloudRenderer:
    def __init__(self, ctx):
        self.ctx: moderngl.Context = ctx
        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER
        )
        self.num_points = 0
        self.capacity_bytes = 0
        self.vbo = None
        self.vao = None

    def update_points(self, points):
        self._validate_points(points)
        needed_bytes = points.nbytes
        # vbo is a gpu buffer, reallocating it is pricey
        if self.vbo is None or needed_bytes > self.capacity_bytes:
            if self.vbo:
                self.vbo.release()
                self.vao = None

            new_capacity = max(needed_bytes, int(self.capacity_bytes * 1.5))
            self.vbo = self.ctx.buffer(reserve=new_capacity)
            self.capacity_bytes = new_capacity
            self.vao = self.ctx.vertex_array(
                self.program, [(self.vbo, "3f", "in_vertex")]
            )

        self.vbo.write(points.tobytes())
        self.num_points = points.shape[0]

    def render(self, mvp, pointsize=2.0, color=(1.0, 1.0, 1.0, 1.0)):
        self._validate_mvp(mvp)

        # render the updated points on the opengl surface
        self.program["mvp"].write(mvp.astype(np.float32).tobytes())
        self.program["pointsize"].value = pointsize
        self.program["color"].value = color
        self.vao.render(moderngl.POINTS, vertices=self.num_points)

    def _validate_mvp(self, mvp: np.ndarray):
        valid_type = type(mvp) is np.ndarray
        valid_dimension = mvp.ndim == 2
        valid_shape = mvp.shape == (4, 4)

        if not (valid_type and valid_shape and valid_dimension):
            raise ValueError("Invalid mvp matrix")

    def _validate_points(self, points: np.ndarray):
        valid_type = type(points) is np.ndarray
        valid_dimension = points.ndim == 2
        valid_shape = points.shape[1] == 3

        if not (valid_type and valid_shape and valid_dimension):
            raise ValueError("Invalid points array")
