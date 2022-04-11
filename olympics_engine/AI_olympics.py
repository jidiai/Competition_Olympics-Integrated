from scenario import Running, table_hockey, football, wrestling
import sys
from pathlib import Path
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
from olympics_engine.generator import create_scenario

import random


class AI_Olympics:
    def __init__(self, random_selection, minimap):

        self.random_selection = random_selection
        self.minimap_mode = minimap

        self.running_game = Running(create_scenario("running"), minimap=self.minimap_mode)
        self.tablehockey_game = table_hockey(create_scenario("table-hockey"), minimap=self.minimap_mode)
        self.football_game = football(create_scenario('football'), minimap=self.minimap_mode)
        self.wrestling_game = wrestling(create_scenario('wrestling'), minimap=self.minimap_mode)

        self.game_pool = [{"name": 'running', 'game': self.running_game},
                          {"name": 'table-hockey', "game": self.tablehockey_game},
                           {"name": 'football', "game": self.football_game},
                          {"name": 'wrestling', "game": self.wrestling_game}]
        self.view_setting = self.running_game.view_setting

    def reset(self):

        self.done = False
        if self.random_selection:
            selected_game_idx = random.randint(0, len(self.game_pool)-1)
        else:
            selected_game_idx = 0
            self.current_game_idx = 0

        print(f'Playing {self.game_pool[selected_game_idx]["name"]}')
        self.current_game = self.game_pool[selected_game_idx]['game']
        self.game_score = [0,0]

        init_obs = self.current_game.reset()
        return init_obs

    def step(self, action_list):

        obs, reward, done, _ = self.current_game.step(action_list)

        if done:
            winner = self.current_game.check_win()
            if winner != '-1':
                self.game_score[int(winner)] += 1

            if self.random_selection:
                    self.done = True
            else:
                if self.current_game_idx == len(self.game_pool)-1:
                    self.done = True
                else:
                    self.current_game_idx += 1
                    self.current_game = self.game_pool[self.current_game_idx]['game']
                    print(f'Playing {self.game_pool[self.current_game_idx]["name"]}')
                    obs = self.current_game.reset()


        if self.done:
            print('game score = ', self.game_score)
            if self.game_score[0] > self.game_score[1]:
                self.final_reward = [100, 0]
                print('Results: team purple win!')
            elif self.game_score[1] > self.game_score[0]:
                self.final_reward = [0, 100]
                print('Results: team green win!')
            else:
                self.final_reward = [0,0]
                print('Results: Draw!')

            return obs, self.final_reward, self.done, ''
        else:
            return obs, reward, self.done, ''

    def is_terminal(self):
        return self.done

    def render(self):
        self.current_game.render()



