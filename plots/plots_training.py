import matplotlib.pyplot as plt
import numpy as np
import pickle


# training_log_average = {'episode': [], 'total_average_reward': [], 'current_average_reward': [], 'current_reward': [], 'world_average_reward': []}
# current_average_reward is the reward of one episode, averaged over the 10 environments

# load all agents
def read_in(file_path):
    with open(file_path, 'rb') as f:
        file = pickle.load(f)
        file = file['current_average_reward']
    return file

def std(list):
    array = np.array(list)
    std = np.std(array, axis=0)
    return std

def training_data(file_path, file_name):

    num_trainings = 10
    current_rewards = []
    for i in range(1,11):
        current_rewards.append(read_in(file_path+str(i)+file_name))

    average50_agents = [] # this will store the average reward over the last 50 episodes for each of the 10 agents
    for j in range(len(current_rewards)):
        average50 = []
        for i in range(len(current_rewards[0])):
            if i < 50:
                average = sum([current_rewards[j][i] for i in range(0,i+1)])/(i+1)
                average50.append(average)
            else:
                average = sum([current_rewards[j][i] for i in range(i-50, i)])/50
                average50.append(average)
        average50_agents.append(average50)

    average50_all = [sum(i)/num_trainings for i in zip(*average50_agents)]
    
    return average50_agents, average50_all


training_episodes = np.arange(2500)
average50_agents_lstm, average50_all_lstm = training_data('training/lstm_final/training','.pickle')
average50_agents_fwm, average50_all_fwm = training_data('training/fwm_final/training','.pickle')
# calculate standard deviations
std_fwm = std(average50_agents_fwm)
std_lstm = std(average50_agents_lstm)

plt.rcParams.update({'font.size': 24})
fig = plt.figure(figsize=(12, 9))
#fig.suptitle('Training of LSTM and FWM agent')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.xlim([0,2500])
plt.plot(training_episodes, average50_all_lstm, color='red')
plt.errorbar(training_episodes, average50_all_lstm, std_lstm, color='tab:red', alpha =0.1)
plt.plot(training_episodes, average50_all_fwm, color='blue')
plt.errorbar(training_episodes, average50_all_fwm, std_fwm, color='tab:blue', alpha =0.1)
plt.legend(['average reward LSTM', 'average reward FWM'], loc=4)
plt.savefig("plots/training.png",bbox_inches='tight')
plt.show()
