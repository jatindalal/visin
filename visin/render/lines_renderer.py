import moderngl
import numpy as np

from visin.core.math import MatrixUtils

VERTEX_SHADER = """

#version 330
uniform mat4 mvp;
uniform float pointsize;
in vec3 in_vertex;
in vec4 in_color;

out vec4 v_color;
void main() {
    v_color = in_color;
    gl_Position = mvp * vec4(in_vertex, 1.0);
    gl_PointSize = pointsize;
}

"""

FRAGMENT_SHADER = """

#version 330
in vec4 v_color;
out vec4 f_color;
void main() {
    f_color = v_color;
}

"""


class LinesRenderer:
    def __init__(self, ctx):
        self.ctx: moderngl.Context = ctx
        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER
        )
        self.capacity_bytes = 0
        self.vbo = None
        self.vao = None

    def _validate_lines(self, lines: np.ndarray, colors: np.ndarray):
        valid_type = type(lines) is np.ndarray and type(colors) is np.ndarray
        valid_dimension = lines.ndim == 3 and colors.ndim == 2
        valid_shape = lines.shape[1] == 2 and lines.shape[2] == 3
        valid_number = 2 * lines.shape[0] == colors.shape[0]

        if not (valid_type and valid_shape and valid_dimension and valid_number):
            print("lines", lines.shape)
            print("colors", colors.shape)
            raise ValueError("Invalid lines and colors")

    def update_lines(self, lines, colors):
        self._validate_lines(lines, colors)
        pos = lines.reshape(-1, 3).astype(np.float32)
        col = colors.astype(np.float32)
        print(pos.shape)
        print(col.shape)
        data = np.ascontiguousarray(np.concatenate([pos, col], axis=1))
        needed_bytes = data.nbytes
        # vbo is a gpu buffer, reallocating it is pricey
        if self.vbo is None or needed_bytes > self.capacity_bytes:
            if self.vbo:
                self.vbo.release()
                self.vao = None

            new_capacity = max(needed_bytes, int(self.capacity_bytes * 1.5))
            self.vbo = self.ctx.buffer(reserve=new_capacity)
            self.capacity_bytes = new_capacity
            self.vao = self.ctx.vertex_array(self.program, self.vbo, "in_vertex", "in_color")

        self.vbo.write(data.tobytes())

    def render(
        self,
        mvp,
        pointsize=2.0,
    ):

        # NumPy produces row-major matrices, while OpenGL uniforms expect
        # column-major data for mat4 uploads.
        self.program["mvp"].write(
            np.ascontiguousarray(mvp.T, dtype=np.float32).tobytes()
        )
        self.program["pointsize"].value = pointsize
        self.vao.render(moderngl.LINES)
