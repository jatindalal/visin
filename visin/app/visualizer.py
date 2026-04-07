import glfw
import moderngl
import numpy as np
import pandas as pd
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
from pypcd4.pypcd4 import PointCloud

from visin.core.camera import Camera, CameraController
from visin.render.pointcloud_renderer import PointCloudRenderer


class InputStateMachine:
    def __init__(self):
        self.keys_down = set()
        self.mouse_buttons_down = set()

        self.pointsize = 2.0
        self.use_orthographic_projection = False

    def on_key(self, key, action):
        if action == glfw.RELEASE:
            self.keys_down.discard(key)
            return

        if action in (glfw.PRESS, glfw.REPEAT):
            self.keys_down.add(key)

    def on_mouse_button(self, button, action):
        if button not in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT):
            return

        if action == glfw.RELEASE:
            self.mouse_buttons_down.discard(button)
            return

        if action == glfw.PRESS:
            self.mouse_buttons_down.add(button)

    def is_key_down(self, key) -> bool:
        return key in self.keys_down

    def resolve_camera_interaction(self) -> CameraController.Interaction:
        left_down = glfw.MOUSE_BUTTON_LEFT in self.mouse_buttons_down
        right_down = glfw.MOUSE_BUTTON_RIGHT in self.mouse_buttons_down
        shift_down = self._is_shift_down()

        if right_down or (left_down and shift_down):
            return CameraController.Interaction.Pan
        if left_down:
            return CameraController.Interaction.Orbit
        return CameraController.Interaction.Idle

    def _is_shift_down(self) -> bool:
        return (
            glfw.KEY_LEFT_SHIFT in self.keys_down
            or glfw.KEY_RIGHT_SHIFT in self.keys_down
        )


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
        self.camera = Camera()
        self.camera_controller = CameraController(self.camera)
        self.camera.set_aspect_ratio(width, height)

        self.input_state = InputStateMachine()

        self.last_time = glfw.get_time()

    def run(self):
        self._init_window()
        self._init_imgui()
        self._init_renderers()

        assert self.window is not None
        assert self.imgui_glfw is not None

        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.imgui_glfw.process_inputs()

            self._render()

        self._shutdown()

    def _render(self):
        if (
            self.window is None
            or self.ctx is None
            or self.imgui_io is None
            or self.imgui_glfw is None
            or self.pointcloud_renderer is None
        ):
            return

        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        win_width, win_height = glfw.get_window_size(self.window)
        if win_width == 0 or win_height == 0:
            return

        self._update_display_state(fb_width, fb_height, win_width, win_height)

        self.ctx.clear(0.02, 0.02, 0.02)

        current_time = glfw.get_time()
        delta_time = current_time - self.last_time
        self.last_time = current_time

        self._update_camera(delta_time)
        self._render_scene()

        imgui.new_frame()
        self._render_ui()
        imgui.render()
        assert self.imgui_glfw is not None
        self.imgui_glfw.render(imgui.get_draw_data())

        glfw.swap_buffers(self.window)

    def _render_ui(self):
        imgui.begin("PointCloud Viewer")

        _, self.input_state.pointsize = imgui.slider_float(
            "Pointsize", self.input_state.pointsize, 0.5, 10.0
        )

        clicked, self.input_state.use_orthographic_projection = imgui.checkbox(
            "Use orthographic camera", self.input_state.use_orthographic_projection
        )
        if clicked:
            self.camera.toggle_projection()

        imgui.end()

    def _update_display_state(self, fb_width, fb_height, win_width, win_height):
        assert self.ctx is not None
        assert self.imgui_io is not None
        self.ctx.viewport = (0, 0, fb_width, fb_height)
        self.imgui_io.display_size = imgui.ImVec2(win_width, win_height)
        self.imgui_io.display_framebuffer_scale = imgui.ImVec2(
            fb_width / win_width,
            fb_height / win_height,
        )
        self.camera.set_aspect_ratio(fb_width, fb_height)

    def _render_scene(self):
        assert self.pointcloud_renderer is not None
        mvp = self.camera.get_mvp()
        self.pointcloud_renderer.render(mvp, pointsize=self.input_state.pointsize)

    def _init_imgui(self):
        imgui.create_context()

        self.imgui_io = imgui.get_io()
        self.imgui_io.config_flags |= imgui.ConfigFlags_.docking_enable
        self.imgui_glfw = GlfwRenderer(self.window, attach_callbacks=False)

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
        glfw.set_window_size_callback(self.window, self._on_window_size)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_char_callback(self.window, self._on_char)

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    def _init_renderers(self):
        self.pointcloud_renderer = PointCloudRenderer(self.ctx)

        points = self._load_points("/Users/jd/Downloads/samp24-utm-ground.pcd")
        self.pointcloud_renderer.update_points(points)

    def _on_resize(self, window, width, height):
        self._render()

    def _on_window_size(self, window, width, height):
        if self.imgui_glfw is None:
            return

        self.imgui_glfw.resize_callback(window, width, height)

    def _on_mouse_move(self, window, xpos, ypos):
        if self.imgui_glfw is None or self.imgui_io is None:
            return

        self.imgui_glfw.mouse_callback(window, xpos, ypos)

        self._sync_camera_interaction(xpos, ypos)

        if self.imgui_io.want_capture_mouse:
            return

        width, height = glfw.get_window_size(window)
        self.camera_controller.drag_to(xpos, ypos, width, height)

    def _on_mouse_button(self, window, button, action, mods):
        if self.imgui_glfw is None or self.imgui_io is None:
            return

        self.imgui_glfw.mouse_button_callback(window, button, action, mods)
        self.input_state.on_mouse_button(button, action)
        xpos, ypos = glfw.get_cursor_pos(window)
        self._sync_camera_interaction(xpos, ypos)

    def _on_scroll(self, window, xoffset, yoffset):
        if self.imgui_glfw is None or self.imgui_io is None:
            return

        self.imgui_glfw.scroll_callback(window, xoffset, yoffset)

        if not self.imgui_io.want_capture_mouse:
            self.camera_controller.zoom(yoffset)

    def _on_key(self, window, key, scancode, action, mods):
        if self.imgui_glfw is None or self.imgui_io is None:
            return

        self.imgui_glfw.keyboard_callback(window, key, scancode, action, mods)
        self.input_state.on_key(key, action)

        xpos, ypos = glfw.get_cursor_pos(window)
        self._sync_camera_interaction(xpos, ypos)

        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        if (
            key == glfw.KEY_P
            and action == glfw.PRESS
            and not self.imgui_io.want_capture_keyboard
        ):
            self.camera.toggle_projection()

    def _on_char(self, window, char):
        if self.imgui_glfw is None:
            return

        self.imgui_glfw.char_callback(window, char)

    def _update_camera(self, delta_time):
        if self.imgui_io is None:
            return

        if self.imgui_io.want_capture_keyboard:
            return

        move_forward = 0.0
        move_right = 0.0

        if self.input_state.is_key_down(glfw.KEY_W):
            move_forward += 1.0
        if self.input_state.is_key_down(glfw.KEY_S):
            move_forward -= 1.0
        if self.input_state.is_key_down(glfw.KEY_D):
            move_right += 1.0
        if self.input_state.is_key_down(glfw.KEY_A):
            move_right -= 1.0

        self.camera.move(
            delta_time,
            move_forward=move_forward,
            move_right=move_right,
        )

    def _sync_camera_interaction(self, xpos, ypos):
        if self.window is None or self.imgui_io is None:
            return

        if self.imgui_io.want_capture_mouse:
            self.camera_controller.end_interaction()
            return

        interaction = self.input_state.resolve_camera_interaction()
        self.camera_controller.set_interaction(interaction, xpos, ypos)

    def _load_points(self, path):
        points = PointCloud.from_path(path).numpy(("x", "y", "z"))
        finite_points = points[np.isfinite(points).all(axis=1)]
        if finite_points.size == 0:
            raise RuntimeError("Point cloud does not contain any finite xyz points")

        return np.ascontiguousarray(finite_points, dtype=np.float32)

    def _shutdown(self):
        glfw.terminate()
