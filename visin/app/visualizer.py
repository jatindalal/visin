import glfw
import moderngl
import numpy as np
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

from visin.core.math import MatrixUtils
from visin.render.pointcloud_renderer import PointCloudRenderer


class Visualizer:
    def __init__(self, width=800, height=600, title="Point Cloud Visualizer"):
        self.width = width
        self.height = height
        self.title = title

        self.window = None
        self.ctx = None
        self.imgui_io = None
        self.imgui_glfw = None

        self.pointcloud_renderer = None

    def run(self):
        self._init_window()
        self._init_imgui()
        self._init_renderers()

        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.imgui_glfw.process_inputs()

            self._render()

        self._shutdown()

    def _render(self):
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        win_width, win_height = glfw.get_window_size(self.window)
        if win_width == 0 or win_height == 0:
            return

        self.ctx.viewport = (0, 0, fb_width, fb_height)
        self.imgui_io.display_size = (win_width, win_height)
        self.imgui_io.display_framebuffer_scale = (
            fb_width / win_width,
            fb_height / win_height,
        )

        self.ctx.clear(0.3, 0.3, 0.3)

        # create a random array of shape (N, 3) and send them to renderer
        # points = np.random.rand(100, 3) * 5
        p_range = (-5.0, 5.0)
        points = []
        for i in range(1000):
            points.append(
                (
                    np.random.uniform(*p_range),
                    np.random.uniform(*p_range),
                    np.random.uniform(*p_range),
                )
            )
        points = np.array(points)

        self.pointcloud_renderer.update_points(points)
        projection = MatrixUtils.perspective_projection(45.0, 16 / 9, 0.1, 100.0)
        view = MatrixUtils.look_at(eye=[0, 0, 5], target=[0, 0, 0], up=[0, 1, 0])
        mvp = MatrixUtils.create_mvp(projection, view)
        self.pointcloud_renderer.render(mvp, pointsize=2.0)

        imgui.new_frame()
        self._render_ui()
        imgui.render()
        self.imgui_glfw.render(imgui.get_draw_data())

        glfw.swap_buffers(self.window)

    def _render_ui(self):
        imgui.show_demo_window()

    def _init_imgui(self):
        imgui.create_context()

        self.imgui_io = imgui.get_io()
        self.imgui_io.config_flags |= imgui.ConfigFlags_.docking_enable
        self.imgui_glfw = GlfwRenderer(self.window, attach_callbacks=True)

    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")

        # OS X supports only forward-compatible core profiles from 3.2

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        # glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.TRUE)

        # create a window
        self.window = glfw.create_window(
            self.width, self.height, self.title, None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeWarning("Could not create Window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    def _init_renderers(self):
        self.pointcloud_renderer = PointCloudRenderer(self.ctx)

    def _on_resize(self, window, width, height):
        self._render()

    def _shutdown(self):
        glfw.terminate()
