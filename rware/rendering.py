"""
2D rendering of the Robotic's Warehouse
environment using pyglet
"""

import math
import os
import sys

from gymnasium import error
import numpy as np
import six

from rware.warehouse import Direction

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
except ImportError:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)
_ORANGE = (255, 165, 0)
_DARKORANGE = (255, 140, 0)
_DARKSLATEBLUE = (72, 61, 139)
_TEAL = (0, 128, 128)
_STEELBLUE = (70, 130, 180)
_CRIMSON = (220, 20, 60)
_GOLD = (212, 175, 55)
_GRAY = (120, 120, 120)

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK
_SHELF_COLOR = _DARKSLATEBLUE
_SHELF_REQ_COLOR = _TEAL
_AGENT_COLOR = _DARKORANGE
_AGENT_LOADED_COLOR = _RED
_AGENT_DIR_COLOR = _BLACK
_GOAL_COLOR = (60, 60, 60)
_HUMAN_COLOR = _STEELBLUE
_DANGER_COLOR = _CRIMSON
_DEADLOCK_COLOR = _GOLD

_SHELF_PADDING = 2


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, world_size):
        display = get_display(None)
        self.rows, self.cols = world_size

        self.grid_size = 30
        self.icon_size = 20

        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 2 + self.rows * (self.grid_size + 1)
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def render(self, env, return_rgb_array=False):
        glClearColor(*_BACKGROUND_COLOR, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_danger_zones(env)
        self._draw_blocked_cells(env)
        self._draw_goals(env)
        self._draw_shelfs(env)
        self._draw_agents(env)
        self._draw_humans(env)
        self._draw_blockers(env)
        self._draw_hud(env)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        # HORIZONTAL LINES
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,  # LEFT X
                        (self.grid_size + 1) * r + 1,  # Y
                        (self.grid_size + 1) * self.cols,  # RIGHT X
                        (self.grid_size + 1) * r + 1,  # Y
                    ),
                ),
                ("c3B", (*_GRID_COLOR, *_GRID_COLOR)),
            )

        # VERTICAL LINES
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * c + 1,  # X
                        0,  # BOTTOM Y
                        (self.grid_size + 1) * c + 1,  # X
                        (self.grid_size + 1) * self.rows,  # TOP Y
                    ),
                ),
                ("c3B", (*_GRID_COLOR, *_GRID_COLOR)),
            )
        batch.draw()

    def _draw_shelfs(self, env):
        batch = pyglet.graphics.Batch()

        for shelf in env.shelfs:
            x, y = shelf.x, shelf.y
            y = self.rows - y - 1  # pyglet rendering is reversed
            shelf_color = (
                _SHELF_REQ_COLOR if shelf in env.request_queue else _SHELF_COLOR
            )

            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * x + _SHELF_PADDING + 1,  # TL - X
                        (self.grid_size + 1) * y + _SHELF_PADDING + 1,  # TL - Y
                        (self.grid_size + 1) * (x + 1) - _SHELF_PADDING,  # TR - X
                        (self.grid_size + 1) * y + _SHELF_PADDING + 1,  # TR - Y
                        (self.grid_size + 1) * (x + 1) - _SHELF_PADDING,  # BR - X
                        (self.grid_size + 1) * (y + 1) - _SHELF_PADDING,  # BR - Y
                        (self.grid_size + 1) * x + _SHELF_PADDING + 1,  # BL - X
                        (self.grid_size + 1) * (y + 1) - _SHELF_PADDING,  # BL - Y
                    ),
                ),
                ("c3B", 4 * shelf_color),
            )
        batch.draw()

    def _draw_danger_zones(self, env):
        debug = getattr(env, "_visual_debug", {})
        danger_cells = debug.get("danger_cells", [])
        if not danger_cells:
            return

        batch = pyglet.graphics.Batch()
        for x, y in danger_cells:
            draw_y = self.rows - y - 1
            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * x + 1,
                        (self.grid_size + 1) * draw_y + 1,
                        (self.grid_size + 1) * (x + 1),
                        (self.grid_size + 1) * draw_y + 1,
                        (self.grid_size + 1) * (x + 1),
                        (self.grid_size + 1) * (draw_y + 1),
                        (self.grid_size + 1) * x + 1,
                        (self.grid_size + 1) * (draw_y + 1),
                    ),
                ),
                ("c4B", 4 * (*_DANGER_COLOR, 48)),
            )
        batch.draw()

    def _draw_blocked_cells(self, env):
        debug = getattr(env, "_visual_debug", {})
        blocked_cells = debug.get("blocked_cells", [])
        if not blocked_cells:
            return

        batch = pyglet.graphics.Batch()
        for x, y in blocked_cells:
            draw_y = self.rows - y - 1
            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * x + 1,
                        (self.grid_size + 1) * draw_y + 1,
                        (self.grid_size + 1) * (x + 1),
                        (self.grid_size + 1) * draw_y + 1,
                        (self.grid_size + 1) * (x + 1),
                        (self.grid_size + 1) * (draw_y + 1),
                        (self.grid_size + 1) * x + 1,
                        (self.grid_size + 1) * (draw_y + 1),
                    ),
                ),
                ("c4B", 4 * (*_GRAY, 110)),
            )
        batch.draw()

    def _draw_goals(self, env):
        batch = pyglet.graphics.Batch()

        # draw goal boxes
        for goal in env.goals:
            x, y = goal
            y = self.rows - y - 1  # pyglet rendering is reversed
            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * x + 1,  # TL - X
                        (self.grid_size + 1) * y + 1,  # TL - Y
                        (self.grid_size + 1) * (x + 1),  # TR - X
                        (self.grid_size + 1) * y + 1,  # TR - Y
                        (self.grid_size + 1) * (x + 1),  # BR - X
                        (self.grid_size + 1) * (y + 1),  # BR - Y
                        (self.grid_size + 1) * x + 1,  # BL - X
                        (self.grid_size + 1) * (y + 1),  # BL - Y
                    ),
                ),
                ("c3B", 4 * _GOAL_COLOR),
            )
        batch.draw()

        # draw goal labels
        for goal in env.goals:
            x, y = goal
            y = self.rows - y - 1
            label_x = x * (self.grid_size + 1) + (1 / 2) * (self.grid_size + 1)
            label_y = (self.grid_size + 1) * y + (1 / 2) * (self.grid_size + 1)
            label = pyglet.text.Label(
                "G",
                font_name="Calibri",
                font_size=18,
                bold=False,
                x=label_x,
                y=label_y,
                anchor_x="center",
                anchor_y="center",
                color=(*_WHITE, 255),
            )
            label.draw()

    def _draw_agents(self, env):
        batch = pyglet.graphics.Batch()
        debug = getattr(env, "_visual_debug", {})
        deadlock_agents = set(debug.get("deadlock_agents", []))

        radius = self.grid_size / 3

        resolution = 6

        for agent in env.agents:
            col, row = agent.x, agent.y
            row = self.rows - row - 1  # pyglet rendering is reversed

            # make a circle
            verts = []
            for i in range(resolution):
                angle = 2 * math.pi * i / resolution
                x = (
                    radius * math.cos(angle)
                    + (self.grid_size + 1) * col
                    + self.grid_size // 2
                    + 1
                )
                y = (
                    radius * math.sin(angle)
                    + (self.grid_size + 1) * row
                    + self.grid_size // 2
                    + 1
                )
                verts += [x, y]
            circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))

            draw_color = _AGENT_LOADED_COLOR if agent.carrying_shelf else _AGENT_COLOR

            glColor3ub(*draw_color)
            circle.draw(GL_POLYGON)
            if agent.id in deadlock_agents:
                glColor3ub(*_DEADLOCK_COLOR)
                circle.draw(GL_LINE_LOOP)

        for agent in env.agents:
            col, row = agent.x, agent.y
            row = self.rows - row - 1  # pyglet rendering is reversed

            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * col
                        + self.grid_size // 2
                        + 1,  # CENTER X
                        (self.grid_size + 1) * row
                        + self.grid_size // 2
                        + 1,  # CENTER Y
                        (self.grid_size + 1) * col
                        + self.grid_size // 2
                        + 1
                        + (
                            radius if agent.dir.value == Direction.RIGHT.value else 0
                        )  # DIR X
                        + (
                            -radius if agent.dir.value == Direction.LEFT.value else 0
                        ),  # DIR X
                        (self.grid_size + 1) * row
                        + self.grid_size // 2
                        + 1
                        + (
                            radius if agent.dir.value == Direction.UP.value else 0
                        )  # DIR Y
                        + (
                            -radius if agent.dir.value == Direction.DOWN.value else 0
                        ),  # DIR Y
                    ),
                ),
                ("c3B", (*_AGENT_DIR_COLOR, *_AGENT_DIR_COLOR)),
            )
        batch.draw()

    def _draw_humans(self, env):
        radius = self.grid_size / 4
        resolution = 14

        for human in getattr(env, "humans", []):
            col, row = human.x, human.y
            row = self.rows - row - 1
            verts = []
            for i in range(resolution):
                angle = 2 * math.pi * i / resolution
                x = (
                    radius * math.cos(angle)
                    + (self.grid_size + 1) * col
                    + self.grid_size // 2
                    + 1
                )
                y = (
                    radius * math.sin(angle)
                    + (self.grid_size + 1) * row
                    + self.grid_size // 2
                    + 1
                )
                verts += [x, y]
            circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
            glColor3ub(*_HUMAN_COLOR)
            circle.draw(GL_POLYGON)
            glColor3ub(*_BLACK)
            circle.draw(GL_LINE_LOOP)

    def _draw_blockers(self, env):
        debug = getattr(env, "_visual_debug", {})
        blockers = debug.get("blockers", [])
        if not blockers:
            return

        batch = pyglet.graphics.Batch()
        for blocker in blockers:
            agent = env.agents[blocker["agent_id"] - 1]
            start_x = (self.grid_size + 1) * agent.x + self.grid_size // 2 + 1
            start_y = (
                (self.grid_size + 1) * (self.rows - agent.y - 1)
                + self.grid_size // 2
                + 1
            )
            target_x = (self.grid_size + 1) * blocker["target"][0] + self.grid_size // 2 + 1
            target_y = (
                (self.grid_size + 1) * (self.rows - blocker["target"][1] - 1)
                + self.grid_size // 2
                + 1
            )
            color = _DANGER_COLOR if blocker["reason"] == "human" else _BLACK
            batch.add(
                2,
                gl.GL_LINES,
                None,
                ("v2f", (start_x, start_y, target_x, target_y)),
                ("c3B", (*color, *color)),
            )
        batch.draw()

    def _draw_hud(self, env):
        metrics = getattr(env, "_last_info", {}).get("metrics", {})
        if not metrics:
            return

        lines = [
            f"Step: {getattr(env, '_cur_steps', 0)}",
            f"Humans: {metrics.get('human_count', 0)}",
            f"Deadlock: {metrics.get('deadlock_active', False)}",
            f"Blocked(H): {metrics.get('blocked_by_human', 0)}",
            f"Blocked(A): {metrics.get('blocked_by_agent', 0)}",
            f"Blocked(Z): {metrics.get('blocked_by_zone', 0)}",
            f"Avg wait: {metrics.get('avg_wait_steps', 0.0):.2f}",
            f"Throughput: {metrics.get('throughput', 0.0):.2f}",
        ]
        for idx, text in enumerate(lines):
            label = pyglet.text.Label(
                text,
                font_name="Consolas",
                font_size=10,
                x=8,
                y=self.height - 14 - idx * 14,
                anchor_x="left",
                anchor_y="center",
                color=(*_BLACK, 255),
            )
            label.draw()

    def _draw_badge(self, row, col, index):
        resolution = 6
        radius = self.grid_size / 5

        badge_x = col * (self.grid_size + 1) + (3 / 4) * (self.grid_size + 1)
        badge_y = (
            self.height
            - (self.grid_size + 1) * (row + 1)
            + (1 / 4) * (self.grid_size + 1)
        )

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_WHITE)
        circle.draw(GL_POLYGON)
        glColor3ub(*_BLACK)
        circle.draw(GL_LINE_LOOP)
        label = pyglet.text.Label(
            str(index),
            font_name="Times New Roman",
            font_size=9,
            bold=True,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
            color=(*_BLACK, 255),
        )
        label.draw()
