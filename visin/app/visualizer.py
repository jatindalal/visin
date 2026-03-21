import glfw
import moderngl
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer



class Visualizer:
    def __init__(self, width=800, height=600, title="Point Cloud Visualizer"):
        self.width = width
        self.height = height
        self.title = title

        self.window = None
        self.ctx = None

    def run(self):
        self._init_window()
        self._init_imgui()

        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.imgui_glfw.process_inputs()

            self.ctx.clear(0.3, 0.3, 0.3)

            imgui.new_frame()
            self._render_ui()
            imgui.render()

            self.imgui_glfw.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

        self._shutdown()

    def _render_ui(self):
        imgui.show_demo_window()

    def _init_imgui(self):
        imgui.create_context()

        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.docking_enable
        self.imgui_glfw = GlfwRenderer(self.window, attach_callbacks=True)

    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")

        # OS X supports only forward-compatible core profiles from 3.2

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        # create a window
        self.window = glfw.create_window(
            self.width, self.height, self.title, None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeWarning("Could not create Window")

        glfw.make_context_current(self.window)

        glfw.swap_interval(1)

        self.ctx = moderngl.create_context()

    def _shutdown(self):
        glfw.terminate()
