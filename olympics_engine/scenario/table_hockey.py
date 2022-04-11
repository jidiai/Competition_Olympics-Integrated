from olympics_engine.core import OlympicsBase
from olympics_engine.viewer import Viewer, debug
import pygame
import sys
import random

class table_hockey(OlympicsBase):
    def __init__(self, map, minimap=False):
        self.minimap_mode = minimap

        super(table_hockey, self).__init__(map)
        self.gamma = 1  # v衰减系数
        self.wall_restitution = 0.8
        self.circle_restitution = 1
        self.tau = 0.1

        self.print_log = False

        self.draw_obs = True
        self.show_traj = False
        
        self.speed_cap = 100


    def reset(self):
        self.set_seed()
        self.init_state()
        self.step_cnt = 0
        self.done = False

        self.viewer = Viewer(self.view_setting)
        self.display_mode=False


        init_obs = self.get_obs()
        if self.minimap_mode:
            self._build_minimap()

        output_init_obs = self._build_from_raw_obs(init_obs)
        return output_init_obs




    def check_overlap(self):
        pass



    def check_action(self, action_list):
        action = []
        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].type == 'agent':
                action.append(action_list[0])
                _ = action_list.pop(0)
            else:
                action.append(None)

        return action

    def step(self, actions_list):

        actions_list = self.check_action(actions_list)

        self.stepPhysics(actions_list)
        self.speed_limit()
        self.step_cnt += 1
        self.cross_detect()



        step_reward = self.get_reward()
        obs_next = self.get_obs()              #need to add agent or ball check in get_obs

        done = self.is_terminal()
        self.done = done
        self.change_inner_state()

        if self.minimap_mode:
            self._build_minimap()

        output_obs_next = self._build_from_raw_obs(obs_next)

        return output_obs_next, step_reward, done, ''

    def _build_from_raw_obs(self, obs):
        if self.minimap_mode:
            image = pygame.surfarray.array3d(self.viewer.background).swapaxes(0,1)
            return [{"agent_obs": obs[0], "minimap":image, "id":"team_0"},
                    {"agent_obs": obs[1], "minimap": image, "id":"team_1"}]
        else:
            return [{"agent_obs":obs[0], "id":"team_0"}, {"agent_obs": obs[1], "id":"team_1"}]

    def _build_minimap(self):
        #need to render first
        if not self.display_mode:
            self.viewer.set_mode()
            self.display_mode = True

        self.viewer.draw_background()
        for w in self.map['objects']:
            self.viewer.draw_map(w)

        self.viewer.draw_ball(self.agent_pos, self.agent_list)

        if self.draw_obs:
            self.viewer.draw_obs(self.obs_boundary, self.agent_list)




    def cross_detect(self, **kwargs):
        """
        check whether the agent has reach the cross(final) line
        :return:
        """
        for agent_idx in range(self.agent_num):

            agent = self.agent_list[agent_idx]

            if agent.type == 'ball':
                for object_idx in range(len(self.map['objects'])):
                    object = self.map['objects'][object_idx]

                    if not object.can_pass():
                        continue
                    else:
                        if object.color == 'red' and object.check_cross(self.agent_pos[agent_idx], agent.r):
                            agent.color = 'red'
                            agent.finished = True  # when agent has crossed the finished line
                            agent.alive = False


    def get_reward(self):

        ball_end_pos = None

        for agent_idx in range(self.agent_num):
            agent = self.agent_list[agent_idx]

            if agent.type == 'ball' and agent.finished:
                ball_end_pos = self.agent_pos[agent_idx]

        if ball_end_pos is not None and ball_end_pos[0] < 400:
            if self.agent_pos[0][0] < 400:
                return [0.,100.]
            else:
                return [100., 0.]
        elif ball_end_pos is not None and ball_end_pos[0] > 400:
            if self.agent_pos[0][0] < 400:
                return [100. ,0.]
            else:
                return [0., 100.]

        else:
            return [0. ,0.]



    def is_terminal(self):

        if self.step_cnt >= self.max_step:
            return True

        for agent_idx in range(self.agent_num):
            agent = self.agent_list[agent_idx]
            if agent.type == 'ball' and agent.finished:
                return True

        return False

    def check_win(self):
        if self.done:
            self.ball_end_pos = None
            for agent_idx in range(self.agent_num):
                agent = self.agent_list[agent_idx]
                if agent.type == 'ball' and agent.finished:
                    self.ball_end_pos = self.agent_pos[agent_idx]

        if self.ball_end_pos is None:
            return '-1'
        else:
            if self.ball_end_pos[0] < 400:
                if self.agent_pos[0][0] < 400:
                    return '1'
                else:
                    return '0'
            elif self.ball_end_pos[0] > 400:
                if self.agent_pos[0][0] < 400:
                    return '0'
                else:
                    return '1'





    def render(self, info=None):

        if self.minimap_mode:
            pass
        else:

            if not self.display_mode:
                self.viewer.set_mode()
                self.display_mode=True

            self.viewer.draw_background()
            for w in self.map['objects']:
                self.viewer.draw_map(w)

            self.viewer.draw_ball(self.agent_pos, self.agent_list)

            if self.draw_obs:
                self.viewer.draw_obs(self.obs_boundary, self.agent_list)

        if self.draw_obs:
            if len(self.obs_list) > 0:
                self.viewer.draw_view(self.obs_list, self.agent_list, leftmost_x=500, upmost_y=10, gap = 100)

        if self.show_traj:
            self.get_trajectory()
            self.viewer.draw_trajectory(self.agent_record, self.agent_list)

        self.viewer.draw_direction(self.agent_pos, self.agent_accel)
        #self.viewer.draw_map()

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
