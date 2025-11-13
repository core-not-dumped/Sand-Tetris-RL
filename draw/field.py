import pyglet
from pyglet import shapes
from variables.global_variables import *

class Field:
    def __init__(self):
        self.batch = pyglet.graphics.Batch()
        # When drawing need to consider the line thickness
        l = field_left - field_thickness // 2
        r = field_right + field_thickness // 2
        t = field_top + field_thickness // 2
        b = field_bottom - field_thickness // 2
        self.left_line = shapes.Line(l, t, l, b, thickness=field_thickness, batch=self.batch)
        self.right_line = shapes.Line(r, t, r, b, thickness=field_thickness, batch=self.batch)
        self.top_line = shapes.Line(l, t, r, t, thickness=field_thickness, batch=self.batch)
        self.bottom_line = shapes.Line(l, b, r, b, thickness=field_thickness, batch=self.batch)

    def draw(self):
        self.batch.draw()
