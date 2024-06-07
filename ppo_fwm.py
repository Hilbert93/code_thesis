import torch
import torch.nn as nn
from torch.distributions import Categorical
from datetime import datetime
import numpy as np
import pickle
from simple_FW import FWMRNN



################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class AgentFWM(nn.Module):

    def __init__(self, p):
        super(AgentFWM, self).__init__()
        self.l1      = FWMRNN(isize=p['in_size'], hsize=p['hidden'], k_size=p['k_size'], v_size=p['v_size'], withFWM=True)
        self.h_c     = nn.Linear(2*p['hidden'], p['n_actions']).to(device) # layer to calculate action probabilities
        self.h_v     = nn.Linear(2*p['hidden'], 1).to(device)              # layer to calculate state value
        self.softmax = nn.Softmax(dim=-1)                                  # layer to scale actions between 0 and 1
        self.p       = p
        
    def reset(self):
        device           = next(self.parameters()).device
        self.lstm_hidden = (torch.zeros(1, self.p['batch_size'], self.p['hidden']).to(device),
                            torch.zeros(1, self.p['batch_size'], self.p['hidden']).to(device))
        self.fwm_hidden  = torch.zeros(self.p['batch_size'], self.p['v_size'], self.p['k_size']).to(device)
        self.hidden      = (self.lstm_hidden, self.fwm_hidden)

    def unchain(self):
        if self.hidden:
            lstm_hidden, fwm_hidden = self.hidden
            lstm_hidden = (lstm_hidden[0].detach(), lstm_hidden[1].detach())
            fwm_hidden  = fwm_hidden.detach()
            self.hidden = (lstm_hidden, fwm_hidden)
    
    def clear_trace(self):
        self.l1.clear_trace()

    def __call__(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        out, self.hidden, m_trace = self.l1(x, self.hidden) # output of FWM and hidden states of LSTM and FWM
        out = out.reshape(-1, out.size(-1))
        act = self.h_c(out)
        val = self.h_v(out)
        act = self.softmax(act)
        return act, val, m_trace # m_trace is just for analysis
    
class AgentLSTM(nn.Module):

    def __init__(self, p):
        super(AgentLSTM, self).__init__()
        self.l1 = FWMRNN(p['in_size'], p['hidden'], k_size=p['k_size'], v_size=p['v_size'], withFWM=False)
        self.h_c     = nn.Linear(p['hidden'], p['n_actions']).to(device) # layer to calculate action probabilities
        self.h_v     = nn.Linear(p['hidden'], 1).to(device)              # layer to calculate state value
        self.softmax = nn.Softmax(dim=-1)                                # layer to scale action probabilities between 0 and 1
        self.p       = p
        
    def reset(self):
        device = next(self.parameters()).device
        self.hidden = (torch.zeros(1, self.p['batch_size'], self.p['hidden']).to(device),
                       torch.zeros(1, self.p['batch_size'], self.p['hidden']).to(device))

    def unchain(self):
        if self.hidden:
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

    def clear_trace(self):
        self.l1.clear_trace()

    def __call__(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        out, self.hidden, m_trace = self.l1(x, self.hidden) # output and hidden/cell states of LSTM
        out = out.reshape(-1, out.size(-1))
        act = self.h_c(out)
        val = self.h_v(out)
        act = self.softmax(act)
        return act, val, m_trace # m_trace is just for analysis

class PPO:
    def __init__(self, p, training_worlds, agent='fwm', lr_actor=1e-3, gamma=0.9, K_epochs=2, eps_clip=0.2):
            
        self.state_dim       = training_worlds[0].observation_space.shape[0]
        self.action_dim      = training_worlds[0].action_space.n
        self.training_worlds = training_worlds
        self.lr_actor        = lr_actor
        self.gamma           = gamma
        self.eps_clip        = eps_clip
        self.K_epochs        = K_epochs
        self.agent           = agent
        self.buffer          = RolloutBuffer()
                       
        self.p = p
        if self.agent == 'fwm':
            self.policy        = AgentFWM(self.p)
            self.policy_old    = AgentFWM(self.p)
        elif self.agent == 'lstm':
            self.policy        = AgentLSTM(self.p)
            self.policy_old    = AgentLSTM(self.p)

        self.optimizer         = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.MseLoss           = nn.MSELoss(reduction='none')
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def select_action(self, state):
        # choose an action for a given state
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action_probs, state_val, m_trace = self.policy_old(state) # m_trace is just for analysis
            dist           = Categorical(action_probs)
            action         = dist.sample().detach() # action is chosen from distribution
            action_logprob = dist.log_prob(action).detach()
            state_val      = state_val.detach()
            
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action, state_val, m_trace

    def update(self):
        # Monte Carlo estimate of returns
        rewards_to_go = []
        discounted_reward = np.zeros((1,self.p['batch_size']))
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            for batch_j in range(self.p['batch_size']):
                if is_terminal[0, batch_j]:
                    discounted_reward[0][batch_j] = 0
            
            discounted_reward = reward + (self.gamma * discounted_reward)    
            rewards_to_go.insert(0, discounted_reward)
            
        # convert at first to array and then to tensor out of performance reasons
        rewards_to_go = np.array(rewards_to_go)
        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32).to(device)
        rewards_to_go = torch.squeeze(rewards_to_go).detach().to(device)
        # normalizing the rewards (1e-7 to avoid dividing by 0)
        rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-7) # [num_time_steps, batch_size]

        # convert list to tensor. Otherwise PyTorch can't calculate the new parameters for the NN's
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0), 1).detach().to(device) # [num_time_steps, batch_size, input_size]
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device) # [num_time_steps, batch_size]
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device) # [num_time_steps, batch_size]
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device) # [num_time_steps, batch_size]

        # calculate advantages
        advantages = rewards_to_go.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # reset hidden states
            self.policy.reset()
            # Evaluating old actions and values
            action_probs, state_values, m_trace = self.policy(old_states)
            action_probs = torch.reshape(action_probs,(self.p['num_time_steps'], self.p['batch_size'], self.action_dim))
            dist         = Categorical(action_probs)
            logprobs     = dist.log_prob(old_actions) # [num_time_steps, batch_size]
            dist_entropy = dist.entropy()

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.reshape(state_values, (self.p['num_time_steps'],self.p['batch_size']))
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach()) # [num_time_steps, batch_size]

            # Finding Surrogate Loss  
            surr1 = ratios * advantages # [num_episode, batch_size]
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages # [num_time_steps, batch_size]

            # final loss of clipped objective PPO
            # inverted signs to do gradient descent instead of gradient ascent afterwards
            loss = -torch.min(surr1, surr2) + 0.05 * self.MseLoss(state_values, rewards_to_go) - 0.05 * dist_entropy # [num_time_steps, batch_size]
            loss = torch.mean(loss, 1) # [num_time_steps]

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def train(self, checkpoint_path=''):
        episode                = 1
        cumulative_reward      = np.zeros(self.p['batch_size'])

        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        # these logs are saved to plot the training afterwards
        training_log_average = {'episode': [], 'current_average_reward': [], 'current_reward': [], 'goal': []}
        
        # training loop
        while episode <= self.p['num_episodes']:

            time_step = 0
            current_ep_reward = np.zeros(self.p['batch_size'])

            # reset all environments of the training worlds
            state_array = np.zeros([1, self.p['batch_size'], self.state_dim])
            goal_state = np.zeros([1, self.p['batch_size']])
            for batch_j in range(self.p['batch_size']):
                state_array[:,batch_j,:], goal_state[:,batch_j] = self.training_worlds[batch_j].reset()

            # reset / initialize hidden states 
            self.policy.reset()
            self.policy_old.reset()
            self.policy.clear_trace()
            self.policy_old.clear_trace()
            
            # initialize the previous action, reward, terminal flag
            done = np.zeros([1, self.p['batch_size']], dtype=bool)
            reward = np.zeros([1, self.p['batch_size']])
            action = None

            while time_step < self.p['num_time_steps']:
                # convert states to tensor and add previous reward and action
                state_tensor = torch.FloatTensor(state_array)
                
                # add reward from previous time step
                reward_input = torch.FloatTensor(np.expand_dims(reward, axis=-1))
                state_tensor = torch.cat((state_tensor, reward_input), 2)

                # add the previous action one hot encoded as input
                if action is not None:
                    action_input = torch.FloatTensor(np.eye(self.action_dim)[action.cpu().numpy()])
                    state_tensor = torch.cat((state_tensor, action_input), 2)
                else:
                    action_input = torch.zeros([1, self.p['batch_size'], self.action_dim])
                    state_tensor = torch.cat((state_tensor, action_input), 2) # [1, batch_size, 30]

                # select action with policy (state_val and m_trace are for analysis purposes)
                action, state_val, m_trace = self.select_action(state_tensor)
                action = torch.unsqueeze(action, 0) # [1, batch_size, 1] 

                for batch_j in range(self.p['batch_size']):
                    # we need to use arrays here. Otherwise the world.step() function does not work
                    state_array[:,batch_j,:], reward[:,batch_j], done[:,batch_j], _ = self.training_worlds[batch_j].step(action[:,batch_j])

                time_step +=1
                
                # saving reward and is_terminals
                # copies has to be used otherwise the latest array is stored for every array in the list
                self.buffer.rewards.append(reward.copy())
                self.buffer.is_terminals.append(done.copy())

                # calculate accumulative reward
                current_ep_reward += np.squeeze(reward)

                # if the goal state is reached, reset the starting state randomly
                for batch_j in range(self.p['batch_size']):
                    if done[:,batch_j]==True:
                        state_array[:,batch_j,:] = self.training_worlds[batch_j].random_state()
            
            # calculate the different rewards
            cumulative_reward += current_ep_reward
            current_average_reward = np.sum(current_ep_reward)/self.p['batch_size']
            # save the average log file
            training_log_average['episode'].append(episode)
            training_log_average['current_reward'].append(np.around(current_ep_reward, 2).tolist()) # reward of each world
            training_log_average['current_average_reward'].append(np.around(current_average_reward, 2).tolist()) # reward averaged over all worlds
            training_log_average['goal'].append(goal_state)
            
            # print the reward of each world
            print("Episode : {} \t\t Reward : {}".format(episode, current_ep_reward))

            episode += 1
            # update the network
            self.update()
                   
            # save model weights after 1000 episodes
            if checkpoint_path != '' and episode == 1000:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path + '1000.pth')
                self.save(checkpoint_path + '1000.pth')
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

        total_reward = np.sum(cumulative_reward)
        # save final model weights
        print("--------------------------------------------------------------------------------------------")
        print("saving model at : " + checkpoint_path + '.pth')
        self.save(checkpoint_path + '.pth')
        print("model saved")
        print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
        print("--------------------------------------------------------------------------------------------")
        # print total training time
        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT): ", start_time)
        print("Finished training at (GMT): ", end_time)
        print("Total training time: ", end_time - start_time)
        print("A reward of", cumulative_reward, "per world was reached during", self.p['num_episodes'], "episodes")
        print("That's an average reward of", total_reward/self.p['batch_size'], "per world")
        print("============================================================================================")
        # right now I don't need to return the logs, but later I should be able to save them as csv
        return training_log_average

    def save_training_log(self, training_log, file_path):

        with open (file_path + '.pickle', 'wb') as file:
            pickle.dump(training_log, file)

    def test(self, p):
        print("Testing PPO agent in gridworld environment")
        testing_episodes  = p['num_episodes']
        episode_time_steps = p['num_time_steps']
        
        episode = 1
        cumulative_reward = 0
        
        # keep track for analysis
        test_log = {'reward': [], 'state': [], 'action': [], 'goal': [], 'state_val': []}
        test_log_average = {'episode': [], 'total_average_reward': [], 'current_reward':[]}

        if self.agent=='fwm':
            analysis_log = {'key':[], 'value':[], 'value_old':[], 'w_beta':[], 'query':[], 'out':[], 'r_beta':[], 'F':[],
                            'rnn_out':[], 'rnn_cell':[]}
        elif self.agent=='lstm':
            analysis_log = {'out':[], 'rnn_out':[], 'rnn_cell':[]}

        # test loop
        while episode <= testing_episodes:
            
            test_log_episode = {'episode_time_step': [], 'reward': [], 'state': [], 'action': [], 'goal': [], 'state_val': []}
            time_step = 0
            current_ep_reward = np.zeros(self.p['batch_size'])

            # reset all environments of training_worlds
            state_array = np.zeros([1, self.p['batch_size'], self.state_dim])
            goal_state = np.zeros([1, self.p['batch_size']])
            for batch_j in range(self.p['batch_size']):
                state_array[:,batch_j,:], goal_state[:,batch_j] = self.training_worlds[batch_j].reset()
                test_log_state = np.argmax(state_array, axis=2)
                test_log_episode['state'].append(np.squeeze(test_log_state).tolist())

            # reset / initialize hidden states 
            self.policy.reset()
            self.policy_old.reset()
            self.policy.clear_trace()
            self.policy_old.clear_trace()
            
            # initialize the previous action, reward, terminal flag
            done = np.zeros([1, self.p['batch_size']], dtype=bool)
            reward = np.zeros([1, self.p['batch_size']])
            action = None

            while time_step < episode_time_steps:
                # convert states to tensor and add previous reward and action
                state_tensor = torch.FloatTensor(state_array)
                
                # add reward from previous time step
                reward_input = torch.FloatTensor(np.expand_dims(reward, axis=-1))
                state_tensor = torch.cat((state_tensor, reward_input), 2)

                # add the previous action one hot encoded as input
                if action is not None:
                    action_input = torch.FloatTensor(np.eye(self.action_dim)[action.cpu().numpy()])
                    state_tensor = torch.cat((state_tensor, action_input), 2)
                else:
                    action_input = torch.zeros([1, self.p['batch_size'], self.action_dim])
                    state_tensor = torch.cat((state_tensor, action_input), 2)

                # select action with policy
                action, state_val, m_trace = self.select_action(state_tensor)

                action = torch.unsqueeze(action, 0)

                for batch_j in range(self.p['batch_size']):
                    state_array[:,batch_j,:], reward[:,batch_j], done[:,batch_j], _ = self.training_worlds[batch_j].step(action[:,batch_j])

                test_log_state = np.argmax(state_array, axis=2)
                time_step +=1
                
                # saving reward and is_terminals
                # copies has to be used otherwise the latest array is stored for every array in the list
                self.buffer.rewards.append(reward.copy())
                self.buffer.is_terminals.append(done.copy())

                # calculate accumulative reward
                current_ep_reward += np.squeeze(reward)

                # append log_file
                test_log_episode['episode_time_step'].append(time_step)
                test_log_episode['reward'].append(np.squeeze(reward).tolist())
                test_log_episode['state'].append(np.squeeze(test_log_state).tolist())
                test_log_episode['action'].append(np.squeeze(action).tolist())
                test_log_episode['goal'].append(np.squeeze(goal_state).tolist())
                test_log_episode['state_val'].append(np.squeeze(state_val).tolist())

                # if the goal state is reached, reset the starting state randomly
                for batch_j in range(self.p['batch_size']):
                    if done[:,batch_j]==True:
                        state_array[:,batch_j,:] = self.training_worlds[batch_j].random_state()
                        test_log_state = np.argmax(state_array, axis=2)
                        test_log_episode['state'].append(np.squeeze(test_log_state).tolist())

            # book keeping
            test_log['reward'].append(test_log_episode['reward'])
            test_log['state'].append(test_log_episode['state'])
            test_log['goal'].append(test_log_episode['goal'])
            test_log['action'].append(test_log_episode['action'])
            test_log['state_val'].append(test_log_episode['state_val'])
            
            if self.agent=='fwm':
                analysis_log['rnn_out'].append(np.squeeze(m_trace['rnn_out'].copy()))
                analysis_log['rnn_cell'].append(np.squeeze(m_trace['rnn_cell'].copy()))
                analysis_log['F'].append(np.squeeze(m_trace['F'].copy()))
                analysis_log['key'].append(np.squeeze(m_trace['key'].copy()))
                analysis_log['value'].append(np.squeeze(m_trace['value'].copy()))
                analysis_log['value_old'].append(np.squeeze(m_trace['value_old'].copy()))
                analysis_log['w_beta'].append(np.squeeze(m_trace['w_beta'].copy()))
                analysis_log['query'].append(np.squeeze(m_trace['query'].copy()))
                analysis_log['r_beta'].append(np.squeeze(m_trace['r_beta'].copy()))
                analysis_log['out'].append(np.squeeze(m_trace['out'].copy()))

            elif self.agent=='lstm':
                analysis_log['rnn_out'].append(np.squeeze(m_trace['rnn_out'].copy()))
                analysis_log['rnn_cell'].append(np.squeeze(m_trace['rnn_cell'].copy()))
                analysis_log['out'].append(np.squeeze(m_trace['out'].copy()))

            # calculate the different rewards
            cumulative_reward += current_ep_reward
            world_average_reward = cumulative_reward/episode
            total_average_reward = np.sum(world_average_reward)/self.p['batch_size']

            # save the average log file
            test_log_average['episode'].append(episode)
            test_log_average['total_average_reward'].append(round(total_average_reward, 2))
            test_log_average['current_reward'].append(np.around(current_ep_reward, 2).tolist())

            print("Episode : {} \t\t Reward : {}".format(episode, current_ep_reward))

            episode += 1

        return test_log, test_log_average, analysis_log 

    def save_test_log(self,test_log, test_log_average, analysis_log, file_path):

        with open (file_path + '.pickle', 'wb') as file:
            pickle.dump(test_log, file)

        with open (file_path + '_avg.pickle', 'wb') as file:
            pickle.dump(test_log_average, file)

        with open (file_path + '_analysis.pickle', 'wb') as file:
            pickle.dump(analysis_log, file)