import pyglet
from variables.global_variables import *
from pyglet import shapes
from datetime import datetime

class Score:
    def __init__(self):
        self.batch = pyglet.graphics.Batch()
        
        # right
        self.score_ = pyglet.text.Label(
            'Score',
            x=(screen_width+field_right)//2, y=screen_height*2//3,
            anchor_x='center', anchor_y='center',
            font_size=20,
            batch=self.batch
        )
        self.score_num_ = pyglet.text.Label(
            '0',
            x=self.score_.x, y=self.score_.y - 40,
            anchor_x='center', anchor_y='center',
            font_size=20,
            batch=self.batch
        )
        self.time_ = pyglet.text.Label(
            'Time',
            x=self.score_num_.x, y=self.score_num_.y - 80,
            anchor_x='center', anchor_y='center',
            font_size=20,
            batch=self.batch
        )
        self.time_num_ = pyglet.text.Label(
            '0',
            x=self.time_.x, y=self.time_.y - 40,
            anchor_x='center', anchor_y='center',
            font_size=20,
            batch=self.batch
        )
        self.best_score_ = pyglet.text.Label(
            'Best Score',
            x=self.time_num_.x, y=self.time_num_.y - 80,
            anchor_x='center', anchor_y='center',
            font_size=19,
            batch=self.batch
        )
        self.best_score_num_ = pyglet.text.Label(
            '0',
            x=self.best_score_.x, y=self.best_score_.y - 40,
            anchor_x='center', anchor_y='center',
            font_size=20,
            batch=self.batch
        )

        # top
        self.sand_tetris_ = pyglet.text.Label(
            'Sand Tetris',
            x=screen_width//2, y=screen_height - 60,
            anchor_x='center', anchor_y='center',
            font_size=50,
            batch=self.batch
        )
        self.line_ = shapes.Line(0, field_top + 30, screen_width, field_top + 30, thickness=2, color=(255, 255, 255), batch=self.batch)

        self.time_count_max = 10
        self.time_count = self.time_count_max
        self.start_time = datetime.now()
        self.best_score = 0

    def draw(self, score):
        self.time_count -= 1
        if self.time_count == 0:
            self.later_time = datetime.now()
            diff = self.later_time - self.start_time
            seconds = diff.total_seconds()
            self.time_num_.text = str(int(seconds))
            self.time_count = self.time_count_max
            self.score_num_.text = str(int(score))
            self.best_score = max(self.best_score, score)
            self.best_score_num_.text = str(int(self.best_score))
        self.batch.draw()