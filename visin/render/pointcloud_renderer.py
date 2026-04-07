import moderngl
import numpy as np

from visin.core.math import MatrixUtils

VERTEX_SHADER = """

#version 330
uniform mat4 mvp;
uniform float pointsize;
in vec3 in_vertex;
out float out_height;

void main() {
    out_height = in_vertex.z;
    gl_Position = mvp * vec4(in_vertex, 1.0);
    gl_PointSize = pointsize;
}

"""

FRAGMENT_SHADER = """

#version 330
uniform vec4 color_underground;
uniform vec4 color_ground;
uniform vec4 color_min_height;
uniform vec4 color_max_height;
uniform float max_height;
uniform float ground_value;
in float out_height;
out vec4 f_color;
void main() {
    vec4 final_color;
    if (out_height < ground_value - 0.05) {
        final_color = color_underground;
    } else if (out_height <= ground_value + 0.05) {
        final_color = color_ground;
    } else {
        float relative_height = out_height - ground_value;
        float max_relative = max_height - ground_value;

        float intensity = log(relative_height + 1.0) / log(max_relative + 1.0);
        intensity = clamp(intensity, 0.0, 1.0);

        final_color = mix(color_min_height, color_max_height, intensity);
    }
    f_color = final_color;
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

    def render(
        self,
        mvp,
        pointsize=2.0,
        max_height=100,
        ground_value=0.0,
        color_underground=(0.3, 0.15, 0.0, 1.0),
        color_ground=(0.0, 0.5, 0.2, 1.0),
        color_min_height=(0.0, 0.8, 0.8, 1.0),
        color_max_height=(1.0, 1.0, 1.0, 1.0),
    ):
        self._validate_mvp(mvp)

        # NumPy produces row-major matrices, while OpenGL uniforms expect
        # column-major data for mat4 uploads.
        self.program["mvp"].write(
            np.ascontiguousarray(mvp.T, dtype=np.float32).tobytes()
        )
        self.program["pointsize"].value = pointsize
        self.program["max_height"].value = max_height
        self.program["ground_value"].value = ground_value
        self.program["color_underground"].value = color_underground
        self.program["color_ground"].value = color_ground
        self.program["color_min_height"].value = color_min_height
        self.program["color_max_height"].value = color_max_height
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
