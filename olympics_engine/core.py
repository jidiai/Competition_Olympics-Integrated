# Time   :  2021-10
# Author :  Yutongamber, YanSong97

import math
import pygame
import copy
import time
import numpy as np
import random
import matplotlib.pyplot as pyplot
import sys

from pathlib import Path
CURRENT_PATH = str(Path(__file__).resolve().parent)
sys.path.append(CURRENT_PATH)

from olympics_engine.viewer import Viewer, debug
#from olympics.utils import *

#from generator import create_scenario


# color 宏
COLORS = {
    'red': [255, 0, 0],
    'green': [0, 255, 0],
    'blue': [0, 0, 255],
    'yellow': [255, 255, 0],
    'grey':  [176,196,222],
    'purple': [160, 32, 240],
    'black': [0, 0, 0],
    'white': [255, 255, 255],
    'light green': [204, 255, 229],
    'sky blue': [0,191,255]
}

COLOR_TO_IDX = {
    'red': 7,
    'green': 1,
    'sky blue': 2,
    'yellow': 3,
    'grey': 4,
    'purple': 5,
    'black': 6,
    'light green': 0,
    'blue':8
}

# Map of object type to integers in output (partial) observation
OBJECT_TO_IDX = {
    'agent2': 1,
    'cross': 4,
    'agent1': 5,
    'wall': 6,
    'goal': 7,
}


def point2line(l1, l2, point):
    """
    :param l1: coord of line start point
    :param l2: coord of line end point
    :param point: coord of circle center
    :return:
    """

    l1l2 = [l2[0] - l1[0], l2[1]-l1[1]]
    l1c = [point[0]-l1[0], point[1]-l1[1]]

    cross_prod = abs(l1c[0]*l1l2[1] - l1c[1]*l1l2[0])

    l1l2_length = math.sqrt(l1l2[0]**2 + l1l2[1]**2)
    return cross_prod/l1l2_length

def cross_prod(v1, v2):
    return v1[0]*v2[1] - v1[1]*v2[0]

def line_intersect(line1, line2):       #[[x1,y1], [x2,y2]], https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect

    p = line1[0]
    r = [line1[1][0] - line1[0][0] , line1[1][1] - line1[0][1]]

    q = line2[0]
    s = [line2[1][0] - line2[0][0] , line2[1][1] - line2[0][1]]

    rs = cross_prod(r,s)
    if rs == 0:
        return False
    else:
        q_p = [q[0]-p[0], q[1]-p[1]]

        t = cross_prod(q_p, s)/rs
        u = cross_prod(q_p, r)/rs

        if 0<=t<=1 and 0<=u<=1:
            point = [p[0]+t*r[0], p[1]+t*r[1]]
            return True
        else:
            return False

def closest_point(l1, l2, point):
    """
    compute the coordinate of point on the line l1l2 closest to the given point, reference: https://en.wikipedia.org/wiki/Cramer%27s_rule
    :param l1: start pos
    :param l2: end pos
    :param point:
    :return:
    """
    A1 = l2[1] - l1[1]
    B1 = l1[0] - l2[0]
    C1 = (l2[1] - l1[1])*l1[0] + (l1[0] - l2[0])*l1[1]
    C2 = -B1 * point[0] + A1 * point[1]
    det = A1*A1 + B1*B1
    if det == 0:
        cx, cy = point
    else:
        cx = (A1*C1 - B1*C2)/det
        cy = (A1*C2 + B1*C1)/det

    return [cx, cy]


class OlympicsBase(object):
    def __init__(self, map, seed=None):

        self.seed = seed
        self.set_seed()

        self.action_f = [-100, 200]
        self.action_theta = [-30, 30]

        self.agent_num = 0
        self.agent_list = []
        self.agent_init_pos = []
        self.agent_pos = []
        self.agent_previous_pos = []
        self.agent_v = []
        self.agent_accel = []
        self.agent_theta = []

        self.agent_record = []

        self.show_traj = True
        self.draw_obs = True
        self.print_log = False
        self.print_log2 = False
        self.map_object = []
        self.global_wall_ignore = []
        self.global_circle_ignore = []

        # env hyper
        self.tau = 0.1  # delta t
        self.gamma = 0.98  # v衰减系数
        self.wall_restitution = 0.5
        self.circle_restitution = 0.5

        self.step_cnt = 0
        self.done = False
        self.max_step = 500

        self.energy_recover_rate = 200
        self.speed_cap = 500

        #for debugg
        # self.obs_boundary_init = [[80,350], [180,350],[180,450],[80,450]]
        # self.obs_boundary = self.obs_boundary_init
        self.obs_boundary_init = list()
        self.obs_boundary = self.obs_boundary_init

        self.map = map
        #self.check_valid_map()
        self.generate_map(map)
        self.merge_map()

        self.view_setting = map["view"]
        self.map_num = None
        #self.is_render = True
        self.display_mode = False

        self.reset()
        #self.check_overlap()

    def check_valid_map(self):      #not using due to conflicting with arc center repitition...
        object_init_list = [str(self.map['objects'][i].init_pos) for i in range(len(self.map['objects']))]
        temp = self.map['objects']
        print('len1 = {}, len2 = {}'.format(len(object_init_list),len(set(object_init_list))))
        assert len(object_init_list) == len(set(object_init_list)), print('ERROR: There is repitition in your designed map!')

    def generate_map(self, map):
        for index, item in enumerate(map["agents"]):
            self.agent_list.append(item)
            position = item.position_init

            r = item.r
            self.agent_init_pos.append(item.position_init)
            self.agent_num += 1

            if item.type == 'agent':
                visibility = item.visibility
                boundary = self.get_obs_boundaray(position, r, visibility)
            #print("boundary: ", boundary)
                self.obs_boundary_init.append(boundary)
            else:
                self.obs_boundary_init.append(None)
            # self.obs_boundary = self.obs_boundary_init
    def merge_map(self):
        point2wall = {}
        for idx, map_item in enumerate(self.map['objects']):
            if map_item.type != 'wall':
                continue
            l1, l2 = tuple(map_item.l1), tuple(map_item.l2)
            if not map_item.can_pass():
                if l1 not in point2wall.keys():
                    point2wall[l1] = [idx]
                else:
                    point2wall[l1].append(idx)
                if l2 not in point2wall.keys():
                    point2wall[l2] = [idx]
                else:
                    point2wall[l2].append(idx)
        self.point2wall=point2wall

    def get_obs_boundaray(self, init_position, r, visibility):
        # 默认初始视线水平
        boundary = list()
        x_init, y_init = init_position[0], init_position[1]
        for unit in [[0,1], [1,1], [1,-1], [0,-1]]:
            x = x_init + r + visibility * unit[0]
            y = y_init - visibility * unit[1] / 2
            boundary.append([x,y])
        return boundary

    @staticmethod
    def create_seed():
        seed = random.randrange(1000)
        return seed

    def set_seed(self, seed=None):
        pass




    def init_state(self):
        self.agent_pos = []
        self.agent_v = []
        self.agent_accel = []
        self.agent_theta = []
        self.agent_record = []
        random_theta = random.uniform(-180, 180)
        for i in range(len(self.agent_list)):
            self.agent_pos.append(self.agent_init_pos[i])
            self.agent_previous_pos.append(self.agent_init_pos[i])
            self.agent_v.append([0,0])
            self.agent_accel.append([0,0])
            init_obs = self.view_setting["init_obs"][i] if "init_obs" in self.view_setting else 0#random_theta
            self.agent_theta.append([init_obs])
            self.agent_list[i].reset()
            self.agent_list[i].reset_color()

            self.agent_record.append([self.agent_init_pos[i]])

    def check_overlap(self):
        #checking the intialisation of agents
        for agent_idx in range(self.agent_num):
            current_pos = self.agent_pos[agent_idx]
            r = self.agent_list[agent_idx].r
            #check wall overlap
            for object_idx in range(len(self.map['objects'])):
                if self.map['objects'][object_idx].type == 'wall':
                    #distance from point to line
                    l1,l2 = self.map['objects'][object_idx].init_pos
                    distance = point2line(l1,l2,current_pos)
                    if distance < r:
                        raise ValueError("The position of agent {} overlap with object {}. Please reinitialise...".
                                         format(agent_idx, object_idx))
            for rest_idx in range(agent_idx+1, self.agent_num):
                pos1 = self.agent_pos[agent_idx]
                r1 = self.agent_list[agent_idx].r
                pos2 = self.agent_pos[rest_idx]
                r2 = self.agent_list[rest_idx].r

                distance = (pos1[0]-pos2[0])**2 + (pos1[1] - pos2[1])**2
                if distance < (r1+r2)**2:
                    raise ValueError("The position of agent {} overlap with agent {}.".format(agent_idx, rest_idx))


    def reset(self):
        self.set_seed()
        self.init_state()
        self.step_cnt = 0
        self.done = False

        self.viewer = Viewer(self.view_setting)
        self.display_mode=False

        return self.get_obs()

    def theta_decoder(self):
        if self.theta < 0 or self.theta > 360:
            self.theta %= 360

    def _init_view(self):
        self.viewer.draw_ball(self.agent_pos, self.agent_list)
        self.viewer.draw_trajectory(self.agent_record, self.agent_list)
        for w in self.map['objects']:
            self.viewer.draw_map(w)

        pygame.display.flip()

        # self.viewer.draw_ball(self.agent_pos, self.agent_list)
        # if self.show_traj:
        #     self.get_trajectory()
        #     self.viewer.draw_trajectory(self.agent_record, self.agent_list)
        # self.viewer.draw_direction(self.agent_pos, self.agent_accel)
        # # self.viewer.draw_map()
        # for w in self.map['objects']:
        #     self.viewer.draw_map(w)
        #
        # if self.draw_obs:
        #     self.viewer.draw_obs(self.obs_boundary, self.agent_list)
        #     self.viewer.draw_view(self.obs_list, self.agent_list)
        #
        # for event in pygame.event.get():
        #     # 如果单击关闭窗口，则退出
        #     if event.type == pygame.QUIT:
        #         sys.exit()
        # pygame.display.flip()


    def _circle_collision_response(self, coord1, coord2, v1, v2, m1, m2):
        """
        whether the input represents the new or the old need second thoughts
        :param coord1: position of object 1
        :param coord2: position of object 2
        :param v1:  velocity of object 1
        :param v2:  velocity of object 2
        :param m1: mass of object 1
        :param m2: mass of object 2
        :return:
        """
        n_x = coord1[0] - coord2[0]
        n_y = coord1[1] - coord2[1]

        vdiff_x = (v1[0] - v2[0])
        vdiff_y = (v1[1] - v2[1])

        n_vdiff = n_x * vdiff_x + n_y * vdiff_y
        nn = n_x * n_x + n_y * n_y
        b = n_vdiff/nn

        #object 1
        u1_x = v1[0] - 2*(m2/(m1+m2)) * b * n_x
        u1_y = v1[1] - 2*(m2/(m1+m2)) * b * n_y

        #object 2
        u2_x = v2[0] + 2*(m1/(m1+m2)) * b * n_x
        u2_y = v2[1] + 2*(m1/(m1+m2)) * b * n_y

        return [u1_x*self.circle_restitution, u1_y*self.circle_restitution], \
               [u2_x*self.circle_restitution, u2_y*self.circle_restitution]


    def CCD_circle_collision(self, old_pos1, old_pos2, old_v1, old_v2, r1, r2, m1, m2, return_t):
        """
        this is the CCD circle collision for the new SetPhysics function
        """

        relative_pos = [old_pos1[0]-old_pos2[0], old_pos1[1]-old_pos2[1]]
        relative_v = [old_v1[0]-old_v2[0], old_v1[1]-old_v2[1]]
        if (relative_v[0]**2+relative_v[1]**2) == 0:
            return -1

        pos_v = relative_pos[0]*relative_v[0] + relative_pos[1]*relative_v[1]
        K = pos_v/(relative_v[0]**2+relative_v[1]**2)
        l = (relative_pos[0]**2 + relative_pos[1]**2 - (r1+r2)**2)/(relative_v[0]**2+relative_v[1]**2)

        sqrt = (K**2 - l)
        if sqrt <0 and return_t:
            #print('CCD circle no solution')
            return -1

        sqrt = math.sqrt(sqrt)
        t1 = -K - sqrt
        t2 = -K + sqrt
        t = min(t1, t2)

        if return_t:
            return t

        x1,y1 = old_pos1
        x2,y2 = old_pos2
        x1_col = x1 + old_v1[0]*t
        y1_col = y1 + old_v1[1]*t
        x2_col = x2 + old_v2[0]*t
        y2_col = y2 + old_v2[1]*t
        pos_col1, pos_col2 = [x1_col, y1_col], [x2_col, y2_col]

        #handle collision
        v1_col, v2_col = self._circle_collision_response(pos_col1, pos_col2, old_v1, old_v2, m1, m2)      #the position and v at the collision time

        return pos_col1, v1_col, pos_col2, v2_col


    def bounceable_wall_collision_time(self, pos_container, v_container, remaining_t, ignore):

        col_target = None
        col_target_idx = None
        current_idx = None
        current_min_t = remaining_t

        for agent_idx in range(self.agent_num):
            pos = pos_container[agent_idx]
            v = v_container[agent_idx]
            r = self.agent_list[agent_idx].r

            if v[0] == v[1] == 0:
                continue

            for object_idx in range(len(self.map['objects'])):
                object = self.map['objects'][object_idx]

                if object.can_pass():     #cross
                    continue
                if object.ball_can_pass and self.agent_list[agent_idx].type == 'ball':      #for table hockey game
                    continue

                #check the collision time and the collision target (wall and endpoint collision)
                temp_t, temp_col_target = object.collision_time(pos = pos, v = v, radius = r, add_info = [agent_idx, object_idx, ignore])

                if abs(temp_t) < 1e-10:   #the collision time computation has numerical error
                    temp_t = 0


                #if object_idx == 2:
                #    print('agent {}: time on wall {}({}) is = {}, current_min_t = {}'.format(
                #        agent_idx,object_idx,temp_col_target, temp_t, current_min_t))
                    #print('ignore list = ', ignore)

                if 0<= temp_t < current_min_t:
                    if temp_col_target == 'wall' or temp_col_target == 'arc':
                        check = ([agent_idx, object_idx, temp_t] not in ignore)
                    elif temp_col_target == 'l1' or temp_col_target == 'l2':
                        check = ([agent_idx, getattr(object, temp_col_target), temp_t] not in ignore)
                    else:
                        raise NotImplementedError('bounceable_wall_collision_time error')

                    if check:
                        current_min_t = temp_t
                        col_target = temp_col_target
                        col_target_idx = object_idx
                        current_idx = agent_idx

        return current_min_t, col_target, col_target_idx, current_idx

    def wall_response(self, target_idx, col_target, pos, v, r, t):

        object = self.map['objects'][target_idx]

        #t, v_col = object.collision_time(pos, v, r, compute_response = True, response_col_t = t,restitution = self.restitution)
        col_pos, col_v = object.collision_response(pos, v, r, col_target, t, self.wall_restitution)

        return col_pos, col_v

    def update_other(self, pos_container, v_container, t, already_updated):  # should we also contain a???

        for agent_idx in range(self.agent_num):
            if agent_idx in already_updated:
                continue

            old_pos = pos_container[agent_idx]
            old_v = v_container[agent_idx]

            new_x = old_pos[0] + old_v[0] * t
            new_y = old_pos[1] + old_v[1] * t

            pos_container[agent_idx] = [new_x, new_y]
            v_container[agent_idx] = [old_v[0]*self.gamma, old_v[1]*self.gamma]

        return pos_container, v_container

    def update_all(self, pos_container, v_container, t, a):
        for agent_idx in range(self.agent_num):
            accel_x, accel_y = a[agent_idx]

            pos_old = pos_container[agent_idx]
            v_old = v_container[agent_idx]

            vx, vy = v_old
            x, y = pos_old
            x_new = x + vx * t  # update position with t
            y_new = y + vy * t
            pos_new = [x_new, y_new]
            vx_new = self.gamma * vx + accel_x * self.tau  # update v with acceleration
            vy_new = self.gamma * vy + accel_y * self.tau

            #if vx_new ** 2 + vy_new ** 2 > self.max_speed_square:
            #    print('speed limited')
            #    v_new = [vx, vy]
            #else:
            v_new = [vx_new, vy_new]

            pos_container[agent_idx] = pos_new
            v_container[agent_idx] = v_new
        return pos_container, v_container


    def circle_collision_time(self, pos_container, v_container, remaining_t, ignore):   #ignore = [[current_idx, target_idx, collision time]]


        #compute collision time between all circle
        current_idx = None
        target_idx = None
        current_min_t = remaining_t
        col_target = None

        for agent_idx in range(self.agent_num):
            pos1 = pos_container[agent_idx]
            v1 = v_container[agent_idx]
            m1 = self.agent_list[agent_idx].mass
            r1 = self.agent_list[agent_idx].r

            for rest_idx in range(agent_idx + 1, self.agent_num):
                pos2 = pos_container[rest_idx]
                v2 = v_container[rest_idx]
                m2 = self.agent_list[rest_idx].mass
                r2 = self.agent_list[rest_idx].r

                #compute ccd collision time
                collision_t = self.CCD_circle_collision(pos1, pos2, v1, v2, r1, r2, m1, m2, return_t=True)
                # print('agent {}, time on circle {} is  = {}, current_min_t = {}'.format(agent_idx,
                #                                                                         rest_idx,
                #                                                                         collision_t,
                #                                                                         remaining_t))
                # print('ignore list = ', ignore)

                if 0 <= collision_t < current_min_t:# and [agent_idx, rest_idx, collision_t] not in ignore:
                    current_min_t = collision_t
                    current_idx = agent_idx
                    target_idx = rest_idx
                    col_target = 'circle'

        return current_min_t, col_target, current_idx, target_idx

    def actions_to_accel(self, actions_list):
        """
        Convert action(force) to acceleration; if the action is None, the accel is zero; if the agent is fatigue, the
        force will convert to no acceleration;
        :param actions_list:
        :return:
        """
        a_container = [[] for _ in range(self.agent_num)]
        for agent_idx in range(self.agent_num):
            action = actions_list[agent_idx]
            if action is None:
                accel = [0, 0]
            else:
                if self.agent_list[agent_idx].is_fatigue:       #if agent is out of energy, no driving force applies
                    accel = [0,0]
                else:
                    mass = self.agent_list[agent_idx].mass

                    assert self.action_f[0] <= action[0] <= self.action_f[1], print('Continuous driving force needs '
                                                                                    'to be within the range [-100,200]')
                    force = action[0] / mass

                    assert self.action_theta[0] <= action[1] <= self.action_theta[1], print(
                        'Continuous turing angle needs to be within the range [-30deg, 30deg]')
                    theta = action[1]

                    theta_old = self.agent_theta[agent_idx][0]
                    theta_new = theta_old + theta
                    self.agent_theta[agent_idx][0] = theta_new

                    accel_x = force * math.cos(theta_new / 180 * math.pi)
                    accel_y = force * math.sin(theta_new / 180 * math.pi)
                    accel = [accel_x, accel_y]
                    #self.agent_accel[agent_idx] = accel  # update the agent acceleration

            a_container[agent_idx] = accel
        return a_container

    def _add_wall_ignore(self, collision_wall_target, current_agent_idx, target_wall_idx, ignore_wall, if_global=False):

        if collision_wall_target == 'wall' or collision_wall_target ==  'arc':
            if if_global:
                self.global_wall_ignore.append([current_agent_idx, target_wall_idx, 0])
            else:
                ignore_wall.append([current_agent_idx, target_wall_idx, 0])

        elif collision_wall_target == 'l1' or collision_wall_target == 'l2':
            if if_global:
                self.global_wall_ignore.append(
                    [current_agent_idx, getattr(self.map['objects'][target_wall_idx], collision_wall_target), 0])
            else:
                ignore_wall.append(
                    [current_agent_idx, getattr(self.map['objects'][target_wall_idx], collision_wall_target), 0])

        else:
            raise NotImplementedError("ignore list error")

        if not if_global:
            return ignore_wall


    # def handle_wall(self, target_wall_idx, col_target, current_agent_idx, col_t,
    #                 pos_container, v_container, remaining_t, ignore_wall_list):
    #
    #     col_pos, col_v = self.wall_response(target_idx=target_wall_idx, col_target = col_target,
    #                                         pos = pos_container[current_agent_idx], v = v_container[current_agent_idx],
    #                                         r = self.agent_list[current_agent_idx], t = col_t)
    #     pos_container[current_agent_idx] = col_pos
    #     v_container[current_agent_idx] = col_v
    #
    #     pos_container, v_container = self.update_other(pos_container=pos_container, v_container=  v_container, t=col_t,
    #                                                    already_updated=[current_agent_idx])
    #     remaining_t -= col_t
    #     if remaining_t <= 1e-14:
    #         self._add_wall_ignore(col_target, current_agent_idx, target_wall_idx, None, if_global=True)
    #
    #     ignore_wall_list = self._add_wall_ignore(col_target, current_agent_idx, target_wall_idx, ignore_wall_list)
    #
    #     return pos_container, v_container, remaining_t, ignore_wall_list
    def handle_wall(self, target_wall_idx, col_target, current_agent_idx, col_t,
                    pos_container, v_container, remaining_t, ignore_wall_list):

        col_pos, col_v = self.wall_response(target_idx=target_wall_idx, col_target = col_target,
                                            pos = pos_container[current_agent_idx], v = v_container[current_agent_idx],
                                            r = self.agent_list[current_agent_idx], t = col_t)
        pos_container[current_agent_idx] = col_pos
        v_container[current_agent_idx] = col_v

        pos_container, v_container = self.update_other(pos_container=pos_container, v_container=  v_container, t=col_t,
                                                       already_updated=[current_agent_idx])
        remaining_t -= col_t
        if remaining_t <= 1e-14:

            if col_target == 'wall':
                self._add_wall_ignore('wall', current_agent_idx, target_wall_idx, ignore_wall_list,if_global=True)
                self._add_wall_ignore('l1', current_agent_idx, target_wall_idx, ignore_wall_list,if_global=True)
                self._add_wall_ignore('l2', current_agent_idx, target_wall_idx, ignore_wall_list,if_global=True)
            elif col_target == 'l2' or col_target == 'l1':
                self._add_wall_ignore(col_target, current_agent_idx, target_wall_idx,
                                                         ignore_wall_list, if_global=True)
                wall_endpoint = getattr(self.map['objects'][target_wall_idx], col_target)
                connected_wall = self.point2wall[tuple(wall_endpoint)]
                for idx in connected_wall:
                    self._add_wall_ignore('wall', current_agent_idx, idx, ignore_wall_list, if_global=True)
            else:
                self._add_wall_ignore(col_target, current_agent_idx, target_wall_idx, None, if_global=True)         #collision of arc

        if col_target == 'wall':
            ignore_wall_list = self._add_wall_ignore('wall', current_agent_idx, target_wall_idx, ignore_wall_list)

            ignore_wall_list = self._add_wall_ignore('l1', current_agent_idx, target_wall_idx, ignore_wall_list)
            ignore_wall_list = self._add_wall_ignore('l2', current_agent_idx, target_wall_idx, ignore_wall_list)
        elif col_target == 'l2' or col_target == 'l1':
            ignore_wall_list = self._add_wall_ignore(col_target, current_agent_idx, target_wall_idx, ignore_wall_list)
            wall_endpoint = getattr(self.map['objects'][target_wall_idx], col_target)
            connected_wall = self.point2wall[tuple(wall_endpoint)]
            for idx in connected_wall:
                ignore_wall_list = self._add_wall_ignore('wall', current_agent_idx, idx, ignore_wall_list)
        else:
            ignore_wall_list = self._add_wall_ignore(col_target, current_agent_idx, target_wall_idx, ignore_wall_list)


        return pos_container, v_container, remaining_t, ignore_wall_list

    def handle_circle(self, target_circle_idx, col_target, current_circle_idx, col_t,
                      pos_container, v_container, remaining_t, ignore_circle_list):

        pos_col1, v1_col, pos_col2, v2_col = \
            self.CCD_circle_collision(pos_container[current_circle_idx], pos_container[target_circle_idx],
                                      v_container[current_circle_idx], v_container[target_circle_idx],
                                      self.agent_list[current_circle_idx].r, self.agent_list[target_circle_idx].r,
                                      self.agent_list[current_circle_idx].mass, self.agent_list[target_circle_idx].mass,
                                      return_t = False)

        pos_container[current_circle_idx], pos_container[target_circle_idx] = pos_col1, pos_col2
        v_container[current_circle_idx], v_container[target_circle_idx] = v1_col, v2_col

        pos_container, v_container = self.update_other(pos_container, v_container, col_t,
                                                       [current_circle_idx, target_circle_idx])
        remaining_t -= col_t
        if remaining_t <= 1e-14:
            self.global_circle_ignore.append([current_circle_idx, target_circle_idx, 0.0])  # adding instead of defining
        ignore_circle_list.append([current_circle_idx, target_circle_idx, 0.0])

        return pos_container, v_container, remaining_t, ignore_circle_list



    def stepPhysics(self, actions_list, step = None):

        assert len(actions_list) == self.agent_num, print("The number of action needs to match with the number of agents!")
        self.actions_list = actions_list

        #current pos and v
        temp_pos_container = [self.agent_pos[i] for i in range(self.agent_num)]
        temp_v_container = [self.agent_v[i] for i in range(self.agent_num)]
        temp_a_container = self.actions_to_accel(actions_list)
        self.agent_accel = temp_a_container

        remaining_t = self.tau
        ignore_wall = copy.copy(self.global_wall_ignore)
        ignore_circle = copy.copy(self.global_circle_ignore)
        self.global_wall_ignore, self.global_circle_ignore = [], []   #only inherit once


        while True:
            if self.print_log:
                print('Remaining time = ', remaining_t)
                print('The pos = {}, the v = {}'.format(temp_pos_container, temp_v_container))

            earliest_wall_col_t, collision_wall_target, target_wall_idx, current_agent_idx = \
                self.bounceable_wall_collision_time(temp_pos_container, temp_v_container, remaining_t, ignore_wall)    #collision detection with walls

            earliest_circle_col_t, collision_circle_target, current_circle_idx, target_circle_idx = \
                self.circle_collision_time(temp_pos_container, temp_v_container, remaining_t, ignore_circle)

            if self.print_log:
                print('Wall t = {}, collide = {}, agent_idx = {}, wall_idx = {}'.format(
                    earliest_wall_col_t, collision_wall_target, current_agent_idx, target_wall_idx))
                print('Circle t = {}, collide = {}, agent_idx = {}, target_idx = {}'.format(
                    earliest_circle_col_t, collision_circle_target, current_circle_idx, target_circle_idx))


            if collision_wall_target is not None and collision_circle_target is None:
                if self.print_log:
                    print('HIT THE WALL!')

                temp_pos_container, temp_v_container, remaining_t, ignore_wall = \
                    self.handle_wall(target_wall_idx, collision_wall_target, current_agent_idx, earliest_wall_col_t,
                                     temp_pos_container, temp_v_container, remaining_t, ignore_wall)

            elif collision_wall_target is None and collision_circle_target == 'circle':
                if self.print_log:
                    print('HIT THE BALL!')

                temp_pos_container, temp_v_container, remaining_t, ignore_circle = \
                    self.handle_circle(target_circle_idx, collision_circle_target, current_circle_idx,
                                       earliest_circle_col_t, temp_pos_container, temp_v_container, remaining_t,
                                       ignore_circle)

            elif collision_wall_target is not None and collision_circle_target == 'circle':
                if self.print_log:
                    print('HIT BOTH!')

                if earliest_wall_col_t < earliest_circle_col_t:
                    if self.print_log:
                        print('PROCESS WALL FIRST!')

                    temp_pos_container, temp_v_container, remaining_t, ignore_wall = \
                        self.handle_wall(target_wall_idx, collision_wall_target, current_agent_idx, earliest_wall_col_t,
                                         temp_pos_container, temp_v_container, remaining_t, ignore_wall)

                elif earliest_wall_col_t >= earliest_circle_col_t:
                    if self.print_log:
                        print('PROCESS CIRCLE FIRST!')

                    temp_pos_container, temp_v_container, remaining_t, ignore_circle = \
                        self.handle_circle(target_circle_idx, collision_circle_target, current_circle_idx,
                                           earliest_circle_col_t, temp_pos_container, temp_v_container, remaining_t,
                                           ignore_circle)

                else:
                    raise NotImplementedError("collision error")

            else:   #no collision within this time interval
                if self.print_log:
                    print('NO COLLISION!')
                temp_pos_container, temp_v_container = self.update_all(temp_pos_container, temp_v_container, remaining_t, temp_a_container)
                break   #when no collision occurs, break the collision detection loop

        self.agent_pos = temp_pos_container
        self.agent_v = temp_v_container

        #if self.is_render:
        #    debug('Step: ' + str(step), x = 30)
        #    pos = [(float('%.1f' % i[0]), float('%.1f' % i[1])) for i in self.agent_pos]
        #    debug('Agent pos = ' + str(pos),x = 120)
        #    speed = [ float('%.1f' % (math.sqrt(i[0]**2 + i[1]**2))) for i in self.agent_v  ]
        #    debug('Agent speed = ' + str(speed), x = 420)

    def cross_detect2(self):
        """
        check whether the agent has reach the cross(final) line
        :return:
        """
        for agent_idx in range(self.agent_num):

            agent = self.agent_list[agent_idx]
            for object_idx in range(len(self.map['objects'])):
                object = self.map['objects'][object_idx]

                if not object.can_pass():
                    continue
                else:
                    #print('object = ', object.type)
                    if object.color == 'red' and object.check_cross(self.agent_pos[agent_idx], agent.r):

                        agent.color = 'red'
                        agent.finished = True       #when agent has crossed the finished line
                        agent.alive = False         #kill the agent when finishing the task

    def cross_detect(self, previous_pos, new_pos):

        for object_idx in range(len(self.map['objects'])):
            object = self.map['objects'][object_idx]
            if object.can_pass() and object.color == 'red':
                l1,l2 = object.init_pos         #locate the pos of Final
                final = object

        for agent_idx in range(self.agent_num):
            agent = self.agent_list[agent_idx]
            agent_pre_pos, agent_new_pos = previous_pos[agent_idx], new_pos[agent_idx]

            if line_intersect(line1 = [l1, l2], line2 = [agent_pre_pos, agent_new_pos]):
                agent.color = 'red'
                agent.finished = True
                agent.alive = False

            if (point2line(l1, l2, agent_new_pos) <= self.agent_list[agent_idx].r) and final.check_on_line(closest_point(l1,l2,agent_new_pos)):
                #if the center of circle to the line has distance less or equal to the radius, and the closest point is on the line, then cross the final
                agent.color = 'red'
                agent.finished = True
                agent.alive = False





    def get_obs(self):
        """
        POMDP: partial observation
        step1: 将原坐标系上landmark（Cross, Wall...）映射到新坐标系，生成新坐标
        step2: 在新坐标系下，进行栅格化
        step3: 在视野范围内，填值
        """

        ## =========================== helper functions =============================
        def rotate(x, y, theta):
            """
            坐标轴转，物体不转
            formula reference: https://www.cnblogs.com/jiahenhe2/p/10135235.html
            """
            x_n = math.cos(theta * math.pi / 180) * x + math.sin(theta * math.pi / 180) * y
            y_n = -math.sin(theta * math.pi / 180) * x + math.cos(theta * math.pi / 180) * y
            return x_n, y_n

        def rotate2(x, y, theta):
            """
            坐标轴不转，物体转; 坐标点旋转后的点坐标, 逆时针旋转theta
            """
            x_n = math.cos(theta * math.pi / 180) * x + math.sin(theta * math.pi / 180) * y
            y_n = -math.sin(theta * math.pi / 180) * x + math.cos(theta * math.pi / 180) * y
            return x_n, y_n

        # this is for debug
        self.obs_boundary = list()

        def get_distance(AB, vec_OC, AB_length, pixel):
            """
            通过向量叉乘，求点C到线段AB的距离; 通过点乘判断点位置的三种情况，左边、右边和上面
            :param: 两个点AB -> [[].[]]，点C->[]，AB线段长度
            :return:
            formula reference: https://blog.csdn.net/qq_45735851/article/details/114448767
            """
            vec_OA, vec_OB = AB[0],AB[1]
            vec_CA = [vec_OA[0]-vec_OC[0], vec_OA[1]-vec_OC[1]]
            vec_CB = [vec_OB[0]-vec_OC[0], vec_OB[1]-vec_OC[1]]
            vec_AB = [vec_OB[0]-vec_OA[0], vec_OB[1]-vec_OA[1]]

            vec_AC = [-vec_OA[0]+vec_OC[0], -vec_OA[1]+vec_OC[1]]
            vec_BC = [-vec_OB[0]+vec_OC[0], -vec_OB[1]+vec_OC[1]]

            if pixel:
                if dot(vec_AB,vec_AC) < 0:
                    d = distance_2points(vec_AC)
                elif dot(vec_AB,vec_BC) > 0:
                    d = distance_2points(vec_BC)
                else:
                    d = math.ceil(cross(vec_CA, vec_CB) / AB_length)
            else:
                d = math.ceil(cross(vec_CA, vec_CB) / AB_length)
            return d

        def dot(vec_1, vec_2):
            """
            计算点乘，vec_1, vec_2都为向量
            """
            return vec_1[0]*vec_2[0] + vec_1[1]*vec_2[1]

        def cross(vec_1, vec_2):
            """
            计算叉积，vec_1, vec_2都为向量
            """
            return vec_1[0]*vec_2[1]-vec_1[1]*vec_2[0]

        def distance_2points(vec):
            """
            计算两个点直接距离，vec为向量
            """
            return math.sqrt(vec[0]**2 + vec[1]**2)

        def check_radian(start_radian, end_radian, angle):

            if start_radian >= 0:
                if end_radian >= 0 and end_radian >= start_radian:
                    return True if (start_radian <= angle <= end_radian) else False
                elif end_radian >= 0 and end_radian < start_radian:
                    return True if not (start_radian <= angle <= end_radian) else False

                elif end_radian <= 0:
                    if angle >= 0 and angle >= start_radian:
                        return True
                    elif angle < 0 and angle <= end_radian:
                        return True
                    else:
                        return False

            elif start_radian < 0:
                if end_radian >= 0:
                    if angle >= 0 and angle < end_radian:
                        return True
                    elif angle < 0 and angle > start_radian:
                        return True
                    else:
                        return False
                elif end_radian < 0 and end_radian > start_radian:
                    return True if (angle < 0 and start_radian <= angle <= end_radian) else False
                elif end_radian < 0 and end_radian < start_radian:
                    return True if not (end_radian <= angle <= start_radian) else False

        # ===========================================================================

        obs_list = list()

        for agent_idx, agent in enumerate(self.agent_list):
            if self.agent_list[agent_idx].type == 'ball':
                self.obs_boundary.append(None)
                obs_list.append(None)
                continue

            time_s = time.time()
            theta_copy = self.agent_theta[agent_idx][0]
            agent_pos = self.agent_pos
            agent_x, agent_y = agent_pos[agent_idx][0], agent_pos[agent_idx][1]
            theta = theta_copy
            position_init = agent.position_init

            visibility = self.agent_list[agent_idx].visibility
            v_clear = self.agent_list[agent_idx].visibility_clear
            # obs_map = np.zeros((visibility[0], visibility[1]))
            # obs_weight,obs_height = int(visibility[0]/v_clear[0]),int(visibility[1]/v_clear[1])
            obs_size = int(visibility / v_clear)

            # update_obs_boundary()
            agent_current_boundary = list()
            for b in self.obs_boundary_init[agent_idx]:
                m = b[0]
                n = b[1]
                # print("check b orig: ", b_x, b_y)
                vec_oo_ = (-agent_x, agent_y)
                vec = (-position_init[0], position_init[1])
                vec_o_a = (m, -n)
                # vec_oa = (vec_oo_[0]+vec_o_a[0], vec_oo_[1]+vec_o_a[1])
                vec_oa = (vec[0] + vec_o_a[0], vec[1] + vec_o_a[1])
                b_x_ = vec_oa[0]
                b_y_ = vec_oa[1]
                # print("check b: ", b_x_, b_y_)
                x_new, y_new = rotate2(b_x_, b_y_, theta)
                # print("check x_new: ", x_new, y_new)
                # x_new_ = x_new + agent_x
                # y_new_ = -y_new + agent_y
                x_new_ =  x_new - vec_oo_[0]
                y_new_ =  y_new - vec_oo_[1]
                agent_current_boundary.append([x_new_, -y_new_])
            self.obs_boundary.append(agent_current_boundary)

            # 计算GameMap上objects组件，相对于agent的坐标(没有旋转)
            for index_m, item in enumerate(self.map["objects"]):
                if (item.type == "wall") or (item.type == "cross"):
                    pos = item.init_pos
                    item.cur_pos = list()
                    for index, p in enumerate(pos):
                        vec_o_d = (p[0], -p[1])
                        vec_oo_ = (-agent_x, agent_y)
                        vec_od = (vec_o_d[0] + vec_oo_[0], vec_o_d[1] + vec_oo_[1])
                        item.cur_pos.append([vec_od[0], vec_od[1]])
                elif item.type == "arc":
                    pos = item.center
                    item.cur_pos = list()
                    vec_o_d = (pos[0], -pos[1])
                    vec_oo_ = (-agent_x, agent_y)
                    vec_od = (vec_o_d[0] + vec_oo_[0], vec_o_d[1] + vec_oo_[1])
                    item.cur_pos.append([vec_od[0], vec_od[1]])

            # 计算视野中心点到各个边的距离
            # 视野中心点
            vec_oc = (agent.r+visibility/2, 0)
            c_x = vec_oc[0]
            c_y = vec_oc[1]
            c_x_, c_y_ = rotate2(c_x, c_y, theta)
            vec_oc_ = (c_x_, c_y_)

            map_objects = list()
            map_deduced = dict()
            map_deduced["objects"] = list()
            for c in self.map["objects"]:
                if (c.type == "wall") or (c.type == "cross"):
                    distance = abs(get_distance(c.cur_pos, vec_oc_, c.length, pixel=False))
                    if distance <= visibility/2 * 1.415:
                    # if distance <= 50 * 1.415:
                        map_deduced["objects"].append(c)
                        map_objects.append(c)
                elif c.type == "arc":
                    distance = distance_2points([c.cur_pos[0][0]-vec_oc_[0],c.cur_pos[0][1]-vec_oc_[1]])
                    if distance <= visibility/2 * 1.415 + c.R:
                        map_deduced["objects"].append(c)
                        map_objects.append(c)
                else:
                    raise ValueError("No such object type- {}. Please check scenario.json".
                                     format(c.type))

            map_deduced["agents"] = list()

            # 当前agent自己
            agent_self = self.agent_list[agent_idx]
            agent_self.to_another_agent = []
            agent_self.to_another_agent_rotated = []
            temp_idx = 0
            #for a_i, a_other in enumerate(self.map["agents"]):
            for a_i, a_other in enumerate(self.agent_list):
                if a_i == agent_idx:
                    continue
                else:
                    vec_o_b = (self.agent_pos[a_i][0], -self.agent_pos[a_i][1])
                    vec_oo_ = (-agent_x, agent_y)
                    vec_ob = (vec_o_b[0]+vec_oo_[0], vec_o_b[1]+vec_oo_[1])
                    vec_bc_ = (vec_oc_[0]-vec_ob[0], vec_oc_[1]-vec_ob[1])
                    # agent_self.to_another_agent = vec_ob
                    distance = math.sqrt(vec_bc_[0]**2 + vec_bc_[1]**2)
                    threshold = self.agent_list[agent_idx].visibility * 1.415 / 2 + a_other.r # 默认视线为正方形
                    if distance <= threshold:
                        map_deduced["agents"].append(a_i) # todo
                        a_other.temp_idx = temp_idx
                        map_objects.append(a_other)
                        agent_self.to_another_agent.append(vec_ob)
                        temp_idx += 1

            # obs_map = np.zeros(( obs_weight , obs_height))
            obs_map = np.zeros((obs_size,obs_size))
            for obj in map_deduced["objects"]:
                if (obj.type == "wall") or (obj.type == "cross"):
                    points_pos = obj.cur_pos
                    obj.cur_pos_rotated = list()
                    for pos in points_pos:
                        pos_x = pos[0]
                        pos_y = pos[1]
                        theta_obj = - theta
                        pos_x_, pos_y_ = rotate2(pos_x, pos_y, theta_obj)
                        obj.cur_pos_rotated.append([pos_x_, pos_y_])
                elif obj.type == "arc":
                    pos = obj.cur_pos
                    obj.cur_pos_rotated = list()
                    pos_x = pos[0][0]
                    pos_y = pos[0][1]
                    theta_obj = - theta
                    pos_x_, pos_y_ = rotate2(pos_x, pos_y, theta_obj)
                    obj.cur_pos_rotated.append([pos_x_, pos_y_])

            for id, obj in enumerate(map_deduced["agents"]):
                vec_ob = agent_self.to_another_agent[id]  # todo: 现在只有两个agent
                theta_obj = - theta
                x, y = rotate2(vec_ob[0], vec_ob[1], theta_obj)
                agent_self.to_another_agent_rotated.append((x,y))

            time_stamp = time.time()
            #for component in map_objects:
            for component in list(reversed(map_objects)):           #reverse to consider agent first, then wall
                for i in range(obs_size):
                    x = agent.r + visibility - v_clear*i - v_clear / 2
                    for j in range(obs_size):
                        y = visibility/2 - v_clear*j - v_clear /2
                        point = (x, y)
                        if obs_map[i][j] > 0:           #when there is already object on this pixel
                            continue
                        else:
                            if (component.type == "wall") or (component.type == "cross"):
                                distance = abs(get_distance(component.cur_pos_rotated, point, component.length,
                                                            pixel=True))
                                if distance <= v_clear :  # 距离小于等于1个像素点长度
                                    obs_map[i][j] = COLOR_TO_IDX[component.color]
                            elif component.type == "agent" or component.type == "ball":
                                idx = component.temp_idx
                                vec_bc_ = (x-agent_self.to_another_agent_rotated[idx][0], y-agent_self.to_another_agent_rotated[idx][1])
                                distance = math.sqrt(vec_bc_[0] ** 2 + vec_bc_[1] ** 2)
                                if distance <= component.r:
                                    obs_map[i][j] = COLOR_TO_IDX[component.color]
                            elif component.type == "arc":
                                radius = component.R
                                x_2center, y_2center = x-component.cur_pos_rotated[0][0], y-component.cur_pos_rotated[0][1]
                                theta_pixel = theta
                                pos_x_2center, pos_y_2center = rotate2(x_2center,y_2center,theta_pixel)
                                angle = math.atan2(pos_y_2center, pos_x_2center)
                                start_radian, end_radian = component.start_radian, component.end_radian
                                if check_radian(start_radian,end_radian,angle):
                                # if (angle >= start_radian) and (angle <= end_radian):
                                    vec = [x - component.cur_pos_rotated[0][0], y - component.cur_pos_rotated[0][1]]
                                    distance = distance_2points(vec)
                                    if (distance <= radius + v_clear) and (distance >= radius - v_clear):
                                        obs_map[i][j] = COLOR_TO_IDX[component.color]

            obs_list.append(obs_map)
            if self.print_log2:
                print('agent {} get obs duration {}'.format(agent_idx, time.time() - time_stamp))
        self.obs_list = obs_list


        return obs_list

    def change_inner_state(self):

        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].type == 'ball':
                continue

            if self.agent_list[agent_idx].energy < 0:       #once fatigue, the agent died and lose control
                remaining_energy = -1
            else:

                #previous_pos = self.agent_previous_pos[agent_idx]
                #current_pos = self.agent_pos[agent_idx]

                #moved = math.sqrt((previous_pos[0] - current_pos[0])**2 + (previous_pos[1] - current_pos[1])**2)


                #force_idx = 0 if self.actions_list[agent_idx] is None else self.actions_list[agent_idx][0]
                force = self.agent_list[agent_idx].mass * math.sqrt(self.agent_accel[agent_idx][0]**2 +
                                                                    self.agent_accel[agent_idx][1]**2)
                v = math.sqrt(self.agent_v[agent_idx][0]**2 + self.agent_v[agent_idx][1]**2)
                consume_rate = force*v/50
                consume_rate -= self.energy_recover_rate

                remaining_energy = self.agent_list[agent_idx].energy - consume_rate*self.tau
                #force = force_idx

                #energy_used = force * moved/1000          #F*s

                #remaining_energy = self.agent_list[agent_idx].energy - energy_used + self.energy_recover_rate*self.tau
                if remaining_energy <0:
                    remaining_energy = -1
                elif remaining_energy > self.agent_list[agent_idx].energy_cap:
                    remaining_energy = self.agent_list[agent_idx].energy_cap
                else:
                    pass

            #print('remaining energy = ', remaining_energy)
            self.agent_list[agent_idx].energy = remaining_energy
            #self.agent_list[agent_idx].energy -= (energy_used - self.energy_recover_rate*self.tau)

    def speed_limit(self):

        for agent_idx in range(self.agent_num):
            current_v = self.agent_v[agent_idx]
            current_speed = math.sqrt(current_v[0]**2 + current_v[1]**2)

            if current_speed > self.speed_cap:
                factor = self.speed_cap/current_speed
                cap_v = [current_v[0] * factor, current_v[1] * factor]
                self.agent_v[agent_idx] = cap_v

                #print('OVER-SPEED!!!')



    def render(self, info=None):

        if not self.display_mode:
            self.viewer.set_mode()
            self.display_mode=True

        self.viewer.draw_background()
        # 先画map; ball在map之上
        for w in self.map['objects']:
            self.viewer.draw_map(w)

        self.viewer.draw_ball(self.agent_pos, self.agent_list)
        if self.show_traj:
            self.get_trajectory()
            self.viewer.draw_trajectory(self.agent_record, self.agent_list)
        self.viewer.draw_direction(self.agent_pos, self.agent_accel)
        #self.viewer.draw_map()

        if self.draw_obs:
            self.viewer.draw_obs(self.obs_boundary, self.agent_list)
            self.viewer.draw_view(self.obs_list, self.agent_list)

        #draw energy bar
        #debug('agent remaining energy = {}'.format([i.energy for i in self.agent_list]), x=100)
        self.viewer.draw_energy_bar(self.agent_list)
        debug('Agent 0', x=570, y=110)
        debug('Agent 1', x=640, y=110)
        if self.map_num is not None:
            debug('Map {}'.format(self.map_num), x=100)

        # debug('mouse pos = '+ str(pygame.mouse.get_pos()))
        debug('Step: ' + str(self.step_cnt), x=30)
        if info is not None:
            debug(info, x=100)


        for event in pygame.event.get():
            # 如果单击关闭窗口，则退出
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.display.flip()
        #self.viewer.background.fill((255, 255, 255))

    def get_trajectory(self):

        for i in range(len(self.agent_list)):
            pos_r = copy.deepcopy(self.agent_pos[i])
            self.agent_record[i].append(pos_r)

        #pos_r = copy.deepcopy(self.pos)
        #self.record.append(pos_r)
