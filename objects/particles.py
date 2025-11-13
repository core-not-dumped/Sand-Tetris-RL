import pyglet
from pyglet import shapes
from variables.global_variables import *

class Particles:
    def __init__(self):
        self.batch = pyglet.graphics.Batch()
        self.particle_map = [[False for _ in range(particle_map_height)] for _ in range(particle_map_width)]
        self.update_t = 0
        self.state_particle = np.zeros((channel_num, particle_map_width, particle_map_height), dtype=np.float32)
        self.state = np.zeros((channel_num, particle_map_width, particle_map_height), dtype=np.float32)
        self.falling_start_y = 0
        self.total_score = 0

    def blocks_to_particles(self, pos, color, type):
        x, y = pos
        for px in range(block_size // particle_size):
            for py in range(block_size // particle_size):
                if np.random.rand() < remove_prob:  continue

                nx, ny = x + px * particle_size, y + py * particle_size
                idx_x, idx_y = particle_coor_to_idx(nx, ny)
                if not self.particle_map[idx_x][idx_y]:
                    self.particle_map[idx_x][idx_y] = Particle(nx, ny, particle_size, particle_size, type, color, self.batch)
                    self.state_particle[type][idx_x][idx_y] = 1


    def falling(self):
        # if one line is full the don't fall
        for start_y in range(self.falling_start_y, particle_map_height):
            break_check = False
            for idx_x in range(particle_map_width):
                if not self.particle_map[idx_x][start_y]:
                    break_check = True
                    break
            if break_check: break
        self.falling_start_y = start_y

        width_range_list = [range(particle_map_width), range(particle_map_width - 1, -1, -1)]
        falling_x_list = [[0, 1], [0, -1]]
        for width_range, falling_x in zip(width_range_list, falling_x_list):
            for idx_y in range(self.falling_start_y, particle_map_height):
                for idx_x in width_range:
                    p = self.particle_map[idx_x][idx_y]
                    if not p:   continue
                    for px, py in zip(falling_x, [-1, -2]):
                        nx, ny = idx_x + px, idx_y + py
                        if nx < 0 or nx >= particle_map_width:  continue
                        if ny < 0 or ny >= particle_map_height:  continue
                        if not self.particle_map[nx][ny]:
                            self.particle_map[nx][ny] = p
                            self.particle_map[idx_x][idx_y] = False
                            self.state_particle[p.type][nx][ny] = 1
                            self.state_particle[p.type][idx_x][idx_y] = 0
                            p.x, p.y = particle_idx_to_coor(nx, ny)
                            break

    def BFS(self):
        score = 0
        map_check = [[not self.particle_map[x][y] for y in range(particle_map_height)] for x in range(particle_map_width)]
        for y in range(particle_map_height):
            if not self.particle_map[0][y]: continue
            if map_check[0][y]:             continue

            f, r = 0, 1
            list_q = [(0, y)]
            map_check[0][y] = True
            particle_type = self.particle_map[0][y].type
            remove_particle = False
            while f < r:
                x, y = list_q[f]
                for px, py in zip([-1, 0, 1, 0, 1, -1, 1, -1], [0, 1, 0, -1, 1, 1, -1, -1]):
                    nx, ny = x + px, y + py
                    if nx < 0 or nx >= particle_map_width:  continue
                    if ny < 0 or ny >= particle_map_height: continue
                    if map_check[nx][ny]:                   continue
                    if not particle_type == self.particle_map[nx][ny].type:    continue
                    if nx == particle_map_width - 1:   remove_particle = True
                    list_q.append((nx, ny))
                    map_check[nx][ny] = True
                    r += 1
                f += 1

            if remove_particle:
                for x, y in list_q:
                    self.state_particle[self.particle_map[x][y].type][x][y] = 0
                    self.particle_map[x][y] = False
                score += len(list_q) * block_reward if use_block_reward else line_reward
                self.total_score += len(list_q)
                self.falling_start_y = 0
        return score
    

    def state_add_block(self, blocks):
        self.state = np.zeros((channel_num, particle_map_width, particle_map_height), dtype=np.float32)
        self.state[:self.state_particle.shape[0], :, :] = self.state_particle
        if channel_num == num_type:
            for b in blocks.blocks_list:
                x, y = b.position
                for px in range(block_size // particle_size):
                    for py in range(block_size // particle_size):
                        nx, ny = x + px * particle_size, y + py * particle_size
                        idx_x, idx_y = particle_coor_to_idx(nx, ny)
                        self.state[b.type][idx_x][idx_y] = -1
        if channel_num == num_type + 1:
            for b in blocks.blocks_list:
                x, y = b.position
                for px in range(block_size // particle_size):
                    for py in range(block_size // particle_size):
                        nx, ny = x + px * particle_size, y + py * particle_size
                        idx_x, idx_y = particle_coor_to_idx(nx, ny)
                        self.state[channel_num-1][idx_x][idx_y] = 1


    def future_falling_simulate(self, simulate_falling):
        falling_start_y = self.falling_start_y
        particle_map = [[(p.type+1 if p else 0) for p in row] for row in self.particle_map]

        score = 0
        for falling_frame in range(simulate_falling):
            # falling sand
            for start_y in range(falling_start_y, particle_map_height):
                break_check = False
                for idx_x in range(particle_map_width):
                    if not particle_map[idx_x][start_y]:
                        break_check = True
                        break
                if break_check: break
            falling_start_y = start_y

            falling_sand = False
            width_range_list = [range(particle_map_width), range(particle_map_width - 1, -1, -1)]
            falling_x_list = [[0, 1], [0, -1]]
            for width_range, falling_x in zip(width_range_list, falling_x_list):
                for idx_y in range(falling_start_y, particle_map_height):
                    for idx_x in width_range:
                        p = particle_map[idx_x][idx_y]
                        if not p:   continue
                        for px, py in zip(falling_x, [-1, -2]):
                            nx, ny = idx_x + px, idx_y + py
                            if nx < 0 or nx >= particle_map_width:  continue
                            if ny < 0 or ny >= particle_map_height:  continue
                            if not particle_map[nx][ny]:
                                particle_map[nx][ny] = p
                                particle_map[idx_x][idx_y] = False
                                falling_sand = True
                                break
        
            # BFS
            if falling_frame % 5 == 4 or falling_sand == False:
                removed_sand = False
                map_check = [[not particle_map[x][y] for y in range(particle_map_height)] for x in range(particle_map_width)]
                for y in range(particle_map_height):
                    if not particle_map[0][y]:  continue
                    if map_check[0][y]:         continue

                    f, r = 0, 1
                    list_q = [(0, y)]
                    map_check[0][y] = True
                    particle_type = particle_map[0][y]
                    remove_particle = False
                    while f < r:
                        x, y = list_q[f]
                        for px, py in zip([-1, 0, 1, 0, 1, -1, 1, -1], [0, 1, 0, -1, 1, 1, -1, -1]):
                            nx, ny = x + px, y + py
                            if nx < 0 or nx >= particle_map_width:  continue
                            if ny < 0 or ny >= particle_map_height: continue
                            if map_check[nx][ny]:                   continue
                            if not particle_type == particle_map[nx][ny]:    continue
                            if nx == particle_map_width - 1:   remove_particle = True
                            list_q.append((nx, ny))
                            map_check[nx][ny] = True
                            r += 1
                        f += 1

                    if remove_particle:
                        removed_sand = True
                        for x, y in list_q: particle_map[x][y] = 0
                        if use_block_reward:    score += len(list_q) * block_reward
                        else:                   score = line_reward
                        falling_start_y = 0
                if (not removed_sand) and (not falling_sand):  break
            

        top_height = 0
        for y in range(particle_map_height - 1, -1, -1):
            if any(particle_map[x][y] for x in range(particle_map_width)):
                top_height = y
                break
        misaligned_line = [
            all(
                any(particle_map[x][y] == color for y in range(top_height + 1))
                for x in range(particle_map_width)
            )
            for color in range(1, len(block_color_list) + 1)
        ]
        pres_height_reward = -top_height * height_reward_max / particle_map_height
        pres_misaligned_reward = sum(misaligned_line) * misaligned_reward
        state_reward = pres_height_reward + pres_misaligned_reward
        return score, state_reward


    def get_state_reward(self):
        top_height = 0
        for y in range(particle_map_height - 1, -1, -1):
            if any(self.particle_map[x][y] for x in range(particle_map_width)):
                top_height = y
                break
        misaligned_line = [
            all(
                any(self.particle_map[x][y] == color for y in range(top_height + 1))
                for x in range(particle_map_width)
            )
            for color in range(1, len(block_color_list) + 1)
        ]
        pres_height_reward = -top_height * height_reward_max / particle_map_height
        pres_misaligned_reward = sum(misaligned_line) * misaligned_reward
        state_reward = pres_height_reward + pres_misaligned_reward
        return state_reward
    

    def update(self):
        self.update_t += 1
        if not self.update_t % falling_freq:    self.falling()
        if not self.update_t % BFS_freq:        return self.BFS()
        return 0
    
    def return_top_height(self):
        for y in range(particle_map_height - 1, -1, -1):
            for x in range(particle_map_width):
                if self.particle_map[x][y]:
                    return y
        return 0


    def draw(self):
        self.batch.draw()


class Particle(shapes.Rectangle):
    def __init__(self, x, y, w, h, t, c, b):
        super().__init__(x, y, w, h, c, batch=b)
        self.type = t

    def __lt__(self, other):
        if self.y == other.y:   return self.x < other.x
        return self.y < other.y

