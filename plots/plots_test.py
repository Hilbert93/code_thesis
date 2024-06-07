import matplotlib.pyplot as plt
import numpy as np
import pickle

# test_log = {'reward': [], 'state': [], 'action': [], 'goal': [], 'state_val': []}
# test_log_average = {'episode': [], 'total_average_reward': [], 'current_reward':[]}
def read_in(file_path):
    with open(file_path, 'rb') as f:
        file = pickle.load(f)
        file = file['reward']
        file_array = np.squeeze(np.array(file))
    return file_array
def read_in_avg(file_path):
    with open(file_path, 'rb') as f:
        file = pickle.load(f)
        file = file['total_average_reward'][-1]
    return file

# find goal locations with less than 5 goals
'''no5goals = [] # 53 in the fourth row, 8 second row
for i in range(1,11):
    with open('testing/fwm/test'+str(i)+'.pickle', 'rb') as f:
        file = pickle.load(f)
    with open('testing/fwm/test'+str(i)+'_avg.pickle', 'rb') as f:
        file_avg = pickle.load(f)  
    array = np.array(file_avg['current_reward'])
    #print(np.shape(array))
    index = np.where(array<5)
    #print(index)
    print(index[0])
    for j in index[0]:
        print(file['goal'][j][0])
        no5goals.append(file['goal'][j][0])
print(no5goals)'''

# show the average reward of each agent
average_reward = []
for i in range(1,11):
    average_reward.append(read_in_avg('testing/fwm_final/test'+str(i)+'_avg.pickle'))
print(average_reward)

# find out how many time steps the agents needed to find the goal for the first 5 trials
def test_steps(file_path, file_name):

    test_steps = []
    for i in range(1,11):
        test_steps.append(read_in(file_path+str(i)+file_name))
        
    all_goal_steps = []
    no_5_goals = 0
    num_episodes = 200
    num_time_steps = 250

    # get the amount of step for each trial
    # it seems to be inefficient, but it works
    for k in range(len(test_steps)):
        reshaped = np.reshape(test_steps[k], (1, num_episodes*num_time_steps))
        rewards = np.where(reshaped == 1)[1]
        ep_steps = []
        # get the amount of step for each finding of the goal
        for i in range(num_episodes):
            current_ep_steps = []
            for j in range(len(rewards)):
                if i*250 < rewards[j] < (i+1)*250:
                    num_steps = rewards[j] - i*250
                    current_ep_steps.append(num_steps)
            if len(current_ep_steps) > 4: 
                ep_steps.append(current_ep_steps)
                i+=1

        goal_steps_5 = []
        for i in range(len(ep_steps)):
            goal_steps = []
            for j in range(5):
                if j == 0:
                    temporary = ep_steps[i][0]
                    goal_steps.append(temporary)
                else:
                    temporary = ep_steps[i][j] - ep_steps[i][j-1]
                    goal_steps.append(temporary)
            goal_steps_5.append(goal_steps)
        all_goal_steps.append(sum(np.array(goal_steps_5))/(len(ep_steps)))
        no_5_goals += (num_episodes - len(ep_steps))

    average_goal_steps = sum(all_goal_steps[i] for i in range(5))/5

    return average_goal_steps, no_5_goals


average_goal_steps_lstm, no_5_goals_lstm = test_steps('testing/lstm_final/test','.pickle')
average_goal_steps_fwm, no_5_goals_fwm = test_steps('testing/fwm_final/test','.pickle')

# print the exact average number of time steps
print('LSTM:')
print(average_goal_steps_lstm.T)
print('no 5 goals in '+str(no_5_goals_lstm)+' of 2000 episodes')
print('FWM:')
print(average_goal_steps_fwm.T)
print('no 5 goals in '+str(no_5_goals_fwm)+' of 2000 episodes')

plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(8, 6.5))
#fig.suptitle('Time steps needed to find the goal for LSTM and FWM agent')
plt.xlabel('Trial')
plt.ylabel('Time steps')
x = np.arange(1,6)
x_label = np.arange(1,6)
plt.xticks(x, x)
plt.bar(x-0.2, average_goal_steps_lstm, color='tab:red', width=0.4)
plt.bar(x+0.2, average_goal_steps_fwm, color='tab:blue', width=0.4)
plt.legend(['LSTM', 'FWM'], loc=1)
plt.savefig("plots/test.png",bbox_inches='tight')
plt.show()