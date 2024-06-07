# basic imports
import numpy as np
import pickle
#import pyqtgraph as qg
# CoBel-RL framework
from cobel.interfaces.discrete import InterfaceDiscrete
from cobel.misc.gridworld_tools import make_gridworld
from ppo_fwm import PPO

def load_gridworld(file_path):
    with open(file_path, 'rb') as f:
        file = pickle.load(f)
    return file

def construct_gridworld(gridworld, world_size, observations):
    gridworld['starting_states'] = np.arange(0, world_size**2, 1, dtype=int)

    # a dictionary that contains all employed modules 
    modules = {}
    modules['rl_interface'] = InterfaceDiscrete(modules, gridworld['sas'], observations, gridworld['rewards'], gridworld['terminals'],
                                                gridworld['starting_states'], gridworld['coordinates'], gridworld['goals'],
                                                random_terminals=True, with_GUI=False)
    return modules

def single_run():
    # This method performs a single experimental run, i.e. one experiment.
    
    np.random.seed()
    
    # Put this in into a function in PPO later
    p = {'in_size':[], 'hidden':[], 's_size':[], 'n_actions':[], 'batch_size':[], 'num_episodes':[], 'num_time_steps':[]}
    p['in_size']        = 30 # number_states (25) + last reward (1) + last action (4)
    p['hidden']         = 128
    p['k_size']         = 32
    p['v_size']         = 32
    p['n_actions']      = 4
    p['batch_size']     = 10 
    p['num_episodes']   = 2500
    p['num_time_steps'] = 250

    # initialize the world (gridworld with changing goals)
    num_worlds = 0
    training_worlds = []
    world_size = 5
    observations = np.eye(world_size**2)
    # create as many training worlds as the number of batches
    while num_worlds < p['batch_size']:
        # normal grid world
        terminals = list(range(0,25))
        gridworld1 = make_gridworld(world_size, world_size, terminals=terminals)
        modules = construct_gridworld(gridworld1, world_size, observations)
        training_worlds += [modules['rl_interface']]
        num_worlds += 1

        # grid world 2
        '''gridworld2 = load_gridworld('more_gridworlds/world2.pkl')
        modules = construct_gridworld(gridworld2, world_size, observations)
        training_worlds += [modules['rl_interface']]
        num_worlds += 1'''

        # grid world 3
        '''gridworld3 = load_gridworld('more_gridworlds/world3.pkl')
        modules = construct_gridworld(gridworld3, world_size, observations)
        training_worlds += [modules['rl_interface']]
        num_worlds += 1'''

    
    # initialize RL agent
    # choose agent='fwm' or 'lstm' for an agent with or without memory augmentation
    rl_agent = PPO(p, training_worlds, agent='fwm')
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # train the agent
    training_log_avg = rl_agent.train(checkpoint_path='training/training')
    
    rl_agent.save_training_log(training_log_avg, 'training/training')

    # test the trained agent

    '''rl_agent.load('training/fwm_final/training'+str+'.pth')
    test_log, test_log_avg, analysis_log = rl_agent.test(p)

    rl_agent.save_test_log(test_log, test_log_avg, analysis_log, 'testing/test')'''

if __name__ == '__main__':
        single_run()