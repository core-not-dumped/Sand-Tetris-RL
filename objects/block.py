
import random
import pyglet
from pyglet import shapes
from variables.global_variables import *

class Blocks:
    def __init__(self):
        self.blocks_list = []
        self.next_shape_type = random.randint(0, len(block_coor_list)-1)
        self.next_shape = block_coor_list[self.next_shape_type][0]
        self.next_color_type = random.randint(0, len(block_color_list)-1)
        self.next_color = block_color_list[self.next_color_type]
        self.batch = pyglet.graphics.Batch()
        self.generate_next_shape_color()
        

    def generate_next_shape_color(self):
        for px, py in self.next_shape:
            x = block_start_x + px * block_size
            y = block_start_y + py * block_size
            new_Block = Block(x, y, block_size, block_size, self.next_color_type, self.next_color, self.batch)
            self.blocks_list.append(new_Block)
        self.dir = 0
        self.color_type = self.next_color_type
        self.shape_type = self.next_shape_type

        self.next_shape_type = random.randint(0, len(block_coor_list)-1)
        self.next_shape = block_coor_list[self.next_shape_type][0]
        self.next_color_type = random.randint(0, len(block_color_list)-1)
        self.next_color = block_color_list[self.next_color_type]
        

    def add_particle(self, particles):
        for b in self.blocks_list:
            particles.blocks_to_particles(b.position, b.color, b.type)
        self.blocks_list = []

    def particle_collision_check(self, particles):
        for b in self.blocks_list:
            sx, sy = particle_coor_to_idx(b.x, b.y)
            fx = min(sx + block_size // particle_size + 1, particle_map_width)
            fy = min(sy + block_size // particle_size + 1, particle_map_height)
            sx, sy = max(0, sx - 1), max(0, sy - 1)
            for i in range(sx, fx):
                for j in range(sy, fy):
                    if particles.particle_map[i][j]:
                        self.add_particle(particles)
                        return True
        return False
                    
    
    def field_collision_physics(self, collision_flag, particles):
        if collision_flag['left']:
            coor = block_coor_list[self.shape_type][self.dir]
            min_x = min(coor, key=lambda coord: coord[0])[0]
            for b, c in zip(self.blocks_list, coor):
                b.x = field_left + (c[0] - min_x) * block_size

        if collision_flag['right']:
            coor = block_coor_list[self.shape_type][self.dir]
            max_x = max(coor, key=lambda coord: coord[0])[0]
            for b, c in zip(self.blocks_list, coor):
                b.x = field_right + (c[0] - max_x - 1) * block_size

        if collision_flag['bottom']:
            coor = block_coor_list[self.shape_type][self.dir]
            min_y = min(coor, key=lambda coord: coord[1])[1]
            for b, c in zip(self.blocks_list, coor):
                b.y = field_bottom + (c[1] - min_y) * block_size
            self.add_particle(particles)
            return True
        return False
            

    def field_collision_check(self, particles):
        collision_flag = {'left':False, 'right':False, 'bottom':False}
        for b in self.blocks_list:
            if b.x < field_left:    collision_flag['left'] = True
            if b.x + block_size > field_right:   collision_flag['right'] = True
            if b.y < field_bottom:  collision_flag['bottom'] = True
        return self.field_collision_physics(collision_flag, particles)


    def move(self):
        for b in self.blocks_list:  b.y += block_falling_spd * particle_size


    def move_left(self):
        for b in self.blocks_list:  b.x -= block_x_spd * particle_size


    def move_right(self):
        for b in self.blocks_list:  b.x += block_x_spd * particle_size


    def update(self, key_pressed, particles):

        # generate new block
        if not self.blocks_list:
            self.generate_next_shape_color()
            return self.particle_collision_check(particles)

        # move
        self.move()
        if key_pressed['left']:     self.move_left()
        if key_pressed['right']:    self.move_right()


    def block_collision(self, particles):
        # collision_check
        if self.field_collision_check(particles) or self.particle_collision_check(particles):
            return True
        return False

    def rotate_blocks(self):
        if not self.blocks_list:    return

        dir_num = len(block_coor_list[self.shape_type])
        self.dir = (self.dir + 1) % dir_num
        start_x, start_y = self.blocks_list[0].position
        self.blocks_list = []
        for px, py in block_coor_list[self.shape_type][self.dir]:
            x = start_x + px * block_size
            y = start_y + py * block_size
            color = block_color_list[self.color_type]
            new_Block = Block(x, y, block_size, block_size, self.color_type, color, self.batch)
            self.blocks_list.append(new_Block)
        

    def draw(self):
        self.batch.draw()        


    def get_block_x_pos(self):
        x, _ = self.blocks_list[0].position
        return x


class Block(shapes.Rectangle):
    def __init__(self, x, y, w, h, t, c, batch):
        super().__init__(x, y, w, h, c, batch=batch)
        self.type = t

