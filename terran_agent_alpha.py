from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

import random
import math
import os
import numpy as np
import pandas as pd

DATA_FILE = 'Terran_Agent_data'
RESULT_FILE = 'Winrate'

Q_actions = [
    'no_op',
    
    'Build_SupplyDepot', 
    'Build_Refinery',    
    'Build_Barracks',    
    'Build_Reactor',     
    'Build_TechLab',     
    'Build_EngineeringBay',      
    'Build_Factory',     
    'Build_Armory',      

    'Train_SCV',         
    'Train_Marine',      
    'Train_Marauder',    
    'Train_Hellion',
    'Train_Cyclone',

    'Research_CombatShield',
    'Research_TerranInfantryWeapons',
    'Research_TerranInfantryArmor',
    'Research_TerranVehicleWeapons',

    'Move_Refinery',
]



for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            Q_actions.append('Attack' + '-' + str(mm_x - 16) + '-' + str(mm_y - 16))

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.disallowed_actions = {}

    def choose_action(self, observation, excluded_actions=[]):
        self.check_state_exist(observation)
        
        self.disallowed_actions[observation] = excluded_actions
        
        state_action = self.q_table.ix[observation, :]
        
        for excluded_action in excluded_actions:
            del state_action[excluded_action]

        if np.random.uniform() < self.epsilon:
            
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax()
        else:
            action = np.random.choice(state_action.index)
            
        return action

    def learn(self, s, a, r, s_):
        if s == s_:
            return
        
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        q_predict = self.q_table.ix[s, a] # 현재 state
        
        s_rewards = self.q_table.ix[s_, :]
        
        if s_ in self.disallowed_actions:
            for excluded_action in self.disallowed_actions[s_]:
                del s_rewards[excluded_action]
        
        if s_ != 'terminal':
            q_target = r + self.gamma * s_rewards.max() # 다음 state
        else:
            q_target = r
            
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class TerranAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(Q_actions))))

        self.previous_action = None
        self.previoud_state = None

        self.move_number = 0    # 각 스텝 (0 ~ 2)

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression = 'gzip')
        if os.path.isfile(RESULT_FILE + '.txt'):
            f = open('Winrate.txt', 'r')
            self.rate = []
            for line in f:
                self.rate.append(int(line))
            f.close()


    def transformLocation(self, x, y):
        if not self.base_top_left:
            return (64 - x, 64 - y)
        
        return (x, y)

    def splitAction(self, action_id):
        agent_action = Q_actions[action_id]
            
        x = 0
        y = 0
        if '-' in agent_action:
            agent_action, x, y = agent_action.split('-')

        return (agent_action, x, y)

    
    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
            obs.observation.single_select[0].unit_type == unit_type):
            return True
        if (len(obs.observation.multi_select) > 0 and
            obs.observation.multi_select[0].unit_type == unit_type):
            return True
        return False

    def get_units_by_type(self, obs, unit_type):
#        print(str(obs.observation))
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions
    

    def step(self, obs):
        super(TerranAgent, self).step(obs)

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()

            if player_y.any() and player_y.mean() <= 31 :
                self.base_top_left = 1
            else :
                self.base_top_left = 0

            self.command_center_rallied = False

            vespenes = self.get_units_by_type(obs, units.Neutral.VespeneGeyser)

            self.vespene_1_x = vespenes[0].x
            self.vespene_1_y = vespenes[0].y
            self.vespene_2_x = vespenes[1].x
            self.vespene_2_y = vespenes[1].y

            self.combatshield_research = False
            self.infantryweapons_research = False
            self.infantryarmor_research = False
            self.vehicleweapons_research = False

            self.reactor_count = 0
            self.techlab_count = 0
            self.refinery_worker_count = 0

            self.barrack_location = {}
            self.rand = 0

            self.time_counter = -1

       
        if obs.last():
            reward = obs.reward

            self.rate.append(reward)
            f = open('Winrate.txt', 'w')
            for num in self.rate:
                f.write(str(num) + '\n')
            f.close()

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')

            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

            self.previous_action = None
            self.previous_state = None

            self.command_center_rallied = False
            
            self.move_number = 0
            self.vespene_1_x = 0
            self.vespene_1_y = 0
            self.vespene_2_x = 0
            self.vespene_2_y = 0

            self.combatshield_research = False
            self.infantryweapons_research = False
            self.infantryarmor_research = False
            self.vehicleweapons_research = False

            self.reactor_count = 0
            self.techlab_count = 0
            self.refinery_worker_count = 0

            self.barrack_location = {}

            self.time_counter = -1

            return actions.FUNCTIONS.no_op()


        
        supply_depot_count = len(self.get_units_by_type(obs, units.Terran.SupplyDepot))
        refinery_count = len(self.get_units_by_type(obs, units.Terran.Refinery))
        barracks_count = len(self.get_units_by_type(obs, units.Terran.Barracks))
        engineering_bay_count = len(self.get_units_by_type(obs, units.Terran.EngineeringBay))
        factory_count = len(self.get_units_by_type(obs, units.Terran.Factory))
        armory_count = len(self.get_units_by_type(obs, units.Terran.Armory))

        scv_count = obs.observation.player.food_workers
        free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
        army_supply = obs.observation.player.food_army
        player_minerals = obs.observation.player.minerals
        player_vespene = obs.observation.player.vespene


        
        if self.move_number == 0:
            self.move_number += 1
            self.time_counter += 1

            
            current_state = np.zeros(14)
            current_state[0] = self.time_counter // 10 
            current_state[1] = player_minerals // 100 
            current_state[2] = player_vespene // 50 
            current_state[3] = barracks_count 
            current_state[4] = factory_count 
            current_state[5] = army_supply // 10

            
            enemy_squares = np.zeros(4)
            enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative ==
                                features.PlayerRelative.ENEMY).nonzero()
            for i in range(0, len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 32))
                x = int(math.ceil((enemy_x[i] + 1) / 32))

                enemy_squares[((y - 1) * 2) + (x - 1)] = 1
            if not self.base_top_left:
                enemy_squares = enemy_squares[::-1]
            for i in range(0, 4):
                current_state[i + 6] = enemy_squares[i]

            
            friendly_squares = np.zeros(4)
            friendly_y, friendly_x = (obs.observation.feature_minimap.player_relative ==
                                      features.PlayerRelative.SELF).nonzero()
            for i in range(0, len(friendly_y)):
                y = int(math.ceil((friendly_y[i] + 1) / 32))
                x = int(math.ceil((friendly_x[i] + 1) / 32))

                friendly_squares[((y - 1) * 2) + (x - 1)] = 1
            if not self.base_top_left:
                friendly_squares = friendly_squares[::-1]
            for i in range(0, 4):
                current_state[i + 10] = friendly_squares[i]
            

            
            if self.previous_action is not None:
                self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))


            
            excluded_actions = []
            if supply_depot_count >= 10 or free_supply > 10:
                excluded_actions.append(1)  
            if supply_depot_count == 0 or refinery_count >= 2:
                excluded_actions.append(2)  
            if supply_depot_count == 0 or barracks_count >= 3:
                excluded_actions.append(3)  
            if barracks_count ==0 or self.reactor_count >= 2:
                excluded_actions.append(4)  
            if barracks_count ==0 or self.techlab_count >= 2:
                excluded_actions.append(5)  
            if barracks_count ==0 or engineering_bay_count >= 1:
                excluded_actions.append(6)  
            if barracks_count == 0 or factory_count >= 2:
                excluded_actions.append(7)  
            if factory_count == 0:
                excluded_actions.append(8)  
            if scv_count >= 22:
                excluded_actions.append(9)  
            if barracks_count ==0 or free_supply == 0:
                excluded_actions.append(10)  
                excluded_actions.append(11) 
            if factory_count == 0 or free_supply < 2:
                excluded_actions.append(12) 
                excluded_actions.append(13) 
            if factory_count > 0 and free_supply >= 2 and player_vespene < 100:
                excluded_actions.append(13) 

            if self.techlab_count == 0 or player_vespene < 100 or self.combatshield_research is True:
                excluded_actions.append(14) 
            if engineering_bay_count == 0 or player_vespene < 100 or self.infantryweapons_research is True:
                excluded_actions.append(15) 
            if engineering_bay_count == 0 or player_vespene < 100 or self.infantryarmor_research is True:
                excluded_actions.append(16) 
            if armory_count == 0 or player_vespene < 100 or self.vehicleweapons_research is True:
                excluded_actions.append(17) 
            if refinery_count == 0 or self.refinery_worker_count == 8:
                excluded_actions.append(18) 
            if army_supply == 0:
                excluded_actions.append(19)
                excluded_actions.append(20)
                excluded_actions.append(21)
                excluded_actions.append(22) 


            rl_action = self.qlearn.choose_action(str(current_state), excluded_actions)

            self.previous_state = current_state
            self.previous_action = rl_action

            agent_action, x, y = self.splitAction(self.previous_action)
            
            print(agent_action)
            if agent_action == 'Build_SupplyDepot' or agent_action == 'Build_Barracks' or agent_action == 'Build_EngineeringBay' or agent_action == 'Build_Refinery' or agent_action == 'Build_Factory' or agent_action == 'Build_Armory' or agent_action == 'Move_Refinery':
                scvs = self.get_units_by_type(obs, units.Terran.SCV)
                if len(scvs) > 0:
                    if scvs[0].x < 1 or scvs[0].y < 1 or scvs[0].x > 82 or scvs[0].y > 82 :
                        return actions.FUNCTIONS.select_point("select", (scvs[1].x, scvs[1].y))
                    return actions.FUNCTIONS.select_point("select", (scvs[0].x, scvs[0].y))

            elif agent_action == 'Train_Marine' or agent_action == 'Train_Marauder' :
                barracks = self.get_units_by_type(obs, units.Terran.Barracks)
                if len(barracks) > 0:
                    return actions.FUNCTIONS.select_point("select_all_type", (barracks[0].x, barracks[0].y))

            elif agent_action == 'Train_Hellion' or agent_action == 'Train_Cyclone':
                factorys = self.get_units_by_type(obs, units.Terran.Factory)
                if len(factorys) > 0:
                    return actions.FUNCTIONS.select_point("select_all_type", (factorys[0].x, factorys[0].y))

            elif agent_action == 'Build_Reactor' or agent_action == 'Build_TechLab':
                self.rand = random.choice(list(self.barrack_location.keys()))
                return actions.FUNCTIONS.select_point("select", self.barrack_location[self.rand])
                
            elif agent_action == 'Train_SCV':
                commandcenters = self.get_units_by_type(obs, units.Terran.CommandCenter)
                if len(commandcenters) > 0:
                    return actions.FUNCTIONS.select_point("select_all_type", (commandcenters[0].x, commandcenters[0].y))

            elif agent_action == 'Research_CombatShield' :
                techlabs = self.get_units_by_type(obs, units.Terran.BarracksTechLab)
                if len(techlabs) > 0:
                    if techlabs[0].x > 82 or techlabs[0].y > 82:
                        return actions.FUNCTIONS.no_op()
                    return actions.FUNCTIONS.select_point("select_all_type", (techlabs[0].x, techlabs[0].y))

            elif agent_action == 'Research_TerranInfantryWeapons' or agent_action == 'Research_TerranInfantryArmor':
                bays = self.get_units_by_type(obs, units.Terran.EngineeringBay)
                if len(bays) > 0:
                    return actions.FUNCTIONS.select_point("select_all_type", (bays[0].x, bays[0].y))

            elif agent_action == 'Research_TerranVehicleWeapons' :
                armorys = self.get_units_by_type(obs, units.Terran.Armory)
                if len(armorys) > 0:
                    return actions.FUNCTIONS.select_point("select_all_type", (armorys[0].x, armorys[0].y))

            elif agent_action == 'Attack':
                if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                    return actions.FUNCTIONS.select_army("select")


        elif self.move_number == 1:
            self.move_number += 1

            supply_depot_count = len(self.get_units_by_type(obs, units.Terran.SupplyDepot))
            refinery_count = len(self.get_units_by_type(obs, units.Terran.Refinery))
            barracks_count = len(self.get_units_by_type(obs, units.Terran.Barracks))
            engineering_bay_count = len(self.get_units_by_type(obs, units.Terran.EngineeringBay))
            factory_count = len(self.get_units_by_type(obs, units.Terran.Factory))
            armory_count = len(self.get_units_by_type(obs, units.Terran.Armory))

            agent_action, x, y = self.splitAction(self.previous_action)

            if agent_action == 'Build_SupplyDepot':
                if self.unit_type_is_selected(obs, units.Terran.SCV):
                    if self.can_do(obs, actions.FUNCTIONS.Build_SupplyDepot_screen.id):
                        if self.base_top_left:
                            x = random.randint(1, 41)
                        else:
                            x = random.randint(42, 82)
                        y = random.randint(1, 82)
                        return actions.FUNCTIONS.Build_SupplyDepot_screen("now", (x,y))
            elif agent_action == 'Build_Barracks':
                if self.unit_type_is_selected(obs, units.Terran.SCV):
                    if self.can_do(obs, actions.FUNCTIONS.Build_Barracks_screen.id):
                        if self.base_top_left:
                            x = random.randint(42, 82)
                        else:
                            x = random.randint(1, 41)
                        y = random.randint(1, 82)
                        self.barrack_location[barracks_count] = (x,y)
                        return actions.FUNCTIONS.Build_Barracks_screen("now", (x,y))
            elif agent_action == 'Build_EngineeringBay':
                if self.unit_type_is_selected(obs, units.Terran.SCV):
                    if self.can_do(obs, actions.FUNCTIONS.Build_EngineeringBay_screen.id):
                        if self.base_top_left:
                            x = random.randint(1, 41)
                        else:
                            x = random.randint(42, 82)
                        y = random.randint(1, 82)
                        return actions.FUNCTIONS.Build_EngineeringBay_screen("now", (x,y))
            elif agent_action == 'Build_Factory':
                if self.unit_type_is_selected(obs, units.Terran.SCV):
                    if self.can_do(obs, actions.FUNCTIONS.Build_Factory_screen.id):
                        if self.base_top_left:
                            x = random.randint(42, 82)
                        else:
                            x = random.randint(1, 41)
                        y = random.randint(1, 82)
                        return actions.FUNCTIONS.Build_Factory_screen("now", (x,y))
            elif agent_action == 'Build_Armory':
                if self.unit_type_is_selected(obs, units.Terran.SCV):
                    if self.can_do(obs, actions.FUNCTIONS.Build_Armory_screen.id):
                        if self.base_top_left:
                            x = random.randint(1, 41)
                        else:
                            x = random.randint(42, 82)
                        y = random.randint(1, 82)
                        return actions.FUNCTIONS.Build_Armory_screen("now", (x,y))
            elif agent_action == 'Build_Refinery':
                if self.unit_type_is_selected(obs, units.Terran.SCV):
                    if self.can_do(obs, actions.FUNCTIONS.Build_Refinery_screen.id):
                        if refinery_count == 0:
                            self.refinery_worker_count += 1
                            return actions.FUNCTIONS.Build_Refinery_screen("now", (self.vespene_1_x,self.vespene_1_y))
                        if refinery_count == 1:
                            self.refinery_worker_count += 1
                            return actions.FUNCTIONS.Build_Refinery_screen("now", (self.vespene_2_x,self.vespene_2_y))
            elif agent_action == 'Move_Refinery':
                if self.unit_type_is_selected(obs, units.Terran.SCV):
                    if (refinery_count == 1 and self.refinery_worker_count <= 3) or (refinery_count == 2 and self.refinery_worker_count <= 4):
                        self.refinery_worker_count += 1
                        return actions.FUNCTIONS.Harvest_Gather_screen("queued", (self.vespene_1_x,self.vespene_1_y))
                    if refinery_count == 2 and self.refinery_worker_count <= 6:
                        self.refinery_worker_count += 1
                        return actions.FUNCTIONS.Harvest_Gather_screen("queued", (self.vespene_2_x,self.vespene_2_y))
            elif agent_action == 'Build_Reactor':
                if self.unit_type_is_selected(obs, units.Terran.Barracks):
                    if self.can_do(obs, actions.FUNCTIONS.Build_Reactor_screen.id):
                        self.reactor_count += 1
                        return actions.FUNCTIONS.Build_Reactor_screen("now", self.barrack_location[self.rand])
            elif agent_action == 'Build_TechLab':
                if self.unit_type_is_selected(obs, units.Terran.Barracks):
                    if self.can_do(obs, actions.FUNCTIONS.Build_TechLab_screen.id):
                        self.techlab_count += 1
                        return actions.FUNCTIONS.Build_TechLab_screen("now", self.barrack_location[self.rand])
            elif agent_action == 'Train_SCV':
                if self.unit_type_is_selected(obs, units.Terran.CommandCenter):
                    if self.can_do(obs, actions.FUNCTIONS.Train_SCV_quick.id):
                        return actions.FUNCTIONS.Train_SCV_quick("queued")
            elif agent_action == 'Train_Marine':
                if self.unit_type_is_selected(obs, units.Terran.Barracks):
                    if self.can_do(obs, actions.FUNCTIONS.Train_Marine_quick.id):
                        if self.reactor_count == 2 and self.techlab_count == 2 :
                            return actions.FUNCTIONS.Train_Marine_quick("queued")
                        else:
                            return actions.FUNCTIONS.Train_Marine_quick("now")
            elif agent_action == 'Train_Marauder':
                if self.unit_type_is_selected(obs, units.Terran.Barracks):
                    if self.can_do(obs, actions.FUNCTIONS.Train_Marauder_quick.id):
                        if self.reactor_count == 2 and self.techlab_count == 2 :
                            return actions.FUNCTIONS.Train_Marauder_quick("queued")
                        else:
                            return actions.FUNCTIONS.Train_Marauder_quick("now")
            elif agent_action == 'Train_Hellion':
                if self.unit_type_is_selected(obs, units.Terran.Factory):
                    if self.can_do(obs, actions.FUNCTIONS.Train_Hellion_quick.id):
                        return actions.FUNCTIONS.Train_Hellion_quick("queued")
            elif agent_action == 'Train_Cyclone':
                if self.unit_type_is_selected(obs, units.Terran.Factory):
                    if self.can_do(obs, actions.FUNCTIONS.Train_Cyclone_quick.id):
                        return actions.FUNCTIONS.Train_Cyclone_quick("queued")
            elif agent_action == 'Research_CombatShield':
                if self.unit_type_is_selected(obs, units.Terran.BarracksTechLab):
                    if self.can_do(obs, actions.FUNCTIONS.Research_CombatShield_quick.id):
                        self.combatshield_research = True
                        return actions.FUNCTIONS.Research_CombatShield_quick("queued")
            elif agent_action == 'Research_TerranInfantryWeapons':
                if self.unit_type_is_selected(obs, units.Terran.EngineeringBay):
                    if self.can_do(obs, actions.FUNCTIONS.Research_TerranInfantryWeapons_quick.id):
                        self.infantryweapons_research = True
                        return actions.FUNCTIONS.Research_TerranInfantryWeapons_quick("queued")
            elif agent_action == 'Research_TerranInfantryArmor':
                if self.unit_type_is_selected(obs, units.Terran.EngineeringBay):
                    if self.can_do(obs, actions.FUNCTIONS.Research_TerranInfantryArmor_quick.id):
                        self.infantryarmor_research = True
                        return actions.FUNCTIONS.Research_TerranInfantryArmor_quick("queued")
            elif agent_action == 'Research_TerranVehicleWeapons':
                if self.unit_type_is_selected(obs, units.Terran.Armory):
                    if self.can_do(obs, actions.FUNCTIONS.Research_TerranVehicleWeapons_quick.id):
                        self.vehicleweapons_research = True
                        return actions.FUNCTIONS.Research_TerranVehicleWeapons_quick("queued")
            elif agent_action == 'Attack':
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    location = self.transformLocation(int(x) + (random.randint(-1, 1) * 8), int(y) + (random.randint(-1, 1) * 8))
                    return actions.FUNCTIONS.Attack_minimap("now", location)

        elif self.move_number == 2:
            self.move_number = 0

            supply_depot_count = len(self.get_units_by_type(obs, units.Terran.SupplyDepot))
            refinery_count = len(self.get_units_by_type(obs, units.Terran.Refinery))
            barracks_count = len(self.get_units_by_type(obs, units.Terran.Barracks))
            engineering_bay_count = len(self.get_units_by_type(obs, units.Terran.EngineeringBay))
            factory_count = len(self.get_units_by_type(obs, units.Terran.Factory))
            armory_count = len(self.get_units_by_type(obs, units.Terran.Armory))

            agent_action, x, y = self.splitAction(self.previous_action)

            if agent_action == 'Build_SupplyDepot' or agent_action == 'Build_Barracks' or agent_action == 'Build_EngineeringBay' or agent_action == 'Build_Factory' or agent_action == 'Build_Armory':
                if self.can_do(obs, actions.FUNCTIONS.Harvest_Gather_screen.id):
                    minerals = self.get_units_by_type(obs, units.Neutral.MineralField)
                    if minerals:
                        mineral = random.choice(minerals)

                        return actions.FUNCTIONS.Harvest_Gather_screen("queued", (mineral.x, mineral.y))

            elif agent_action == 'Train_SCV':
                if self.unit_type_is_selected(obs, units.Terran.CommandCenter):
                    if self.command_center_rallied == False:
                        minerals = self.get_units_by_type(obs, units.Neutral.MineralField)
                        if minerals:
                            mineral = random.choice(minerals)
                            self.command_center_rallied = True;

                            return actions.FUNCTIONS.Rally_Workers_screen("now", (mineral.x, mineral.y))

                
        return actions.FUNCTIONS.no_op()       


def main(unused_argv):
    agent = TerranAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                map_name = "Simple64",
                players = [sc2_env.Agent(sc2_env.Race.terran),
                            sc2_env.Bot(sc2_env.Race.protoss,
                                        sc2_env.Difficulty.easy)],
                            # very_easy, easy, medium, medium_hard
                            # hard, harder, very_hard
                agent_interface_format = features.AgentInterfaceFormat(
                    feature_dimensions = features.Dimensions(screen=84, minimap=64),
                    use_feature_units = True), # enable feature units
                    
                step_mul = 16,
                    # 16 - 150 APM
                    #  8 - 300 APM
                game_steps_per_episode = 20000,
                    
                visualize = True) as env:

                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()

                # loop - feeding step details into the agent and receiving actions
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)
