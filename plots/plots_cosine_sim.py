import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# analysis_log = {'key':[], 'value':[], 'w_beta':[], 'query':[], 'out':[], 'r_beta':[], 'F':[],
#                            'rnn_out':[], 'rnn_cell':[]}
# test_log = {'episode': [], 'episode_time_step': [], 'reward': [], 'trajectory': [], 'goal': []}
# test_log_average = {'episode': [], 'total_average_reward': [], 'current_average_reward': [], 'current_reward':[], 'world_average_reward':[]}

def read_in(file_path):
    with open(file_path, 'rb') as f:
        file = pickle.load(f)
    return file
file_path = 'testing/fwm_final/'
file_name = 'test10'
analysis_log = read_in(file_path + file_name+'_analysis.pickle')
test_log = read_in(file_path + file_name+'.pickle')
test_log_avg = read_in(file_path + file_name+'_avg.pickle')

# making a list of all first episodes of each goal location
variable = np.array(test_log['goal'])
goal_location = np.squeeze(variable[:,0])
goal_episode = []
goal_reward = []
for i in range(0,25):
    goal_episodes = np.where(goal_location == i)
    goal_episode.append(goal_episodes[0][0])
    goal_reward.append(test_log_avg['current_reward'][goal_episodes[0][0]])
print(goal_reward)
j = 0
all_key_vectors = np.zeros([25, 32])

# build the average vector for each goal location
for episode in (goal_episode):
    reward_array = np.array(test_log['reward'][episode])
    reward_time_steps1 = np.argwhere(reward_array)
    reward_time_steps = reward_time_steps1[(249 > reward_time_steps1)] # (44,)

    key_array = np.squeeze(analysis_log['key'][episode]) # 250x32
    #key_array = np.squeeze(analysis_log['value'][episode])
    #key_array = np.squeeze(analysis_log['query'][episode])
    #key_array = np.squeeze(analysis_log['out'][episode])

    sum_key_vectors = np.zeros([1, np.shape(key_array)[1]])

    for i in (reward_time_steps):
        sum_key_vectors = sum_key_vectors + key_array[i+1,:]
    average_key_vector = sum_key_vectors.copy() / (len(reward_time_steps))

    all_key_vectors[j,:] = average_key_vector.copy()
    j+=1

cos_sim_k = cosine_similarity(all_key_vectors, all_key_vectors)

plt.rcParams.update({'font.size': 16})
fig1, ax = plt.subplots(1,1,figsize=(7.5,6.5))
im1 = ax.imshow(cos_sim_k)
ax.set_xlabel('Goal location')
ax.set_ylabel('Goal location')
fig1.colorbar(im1, ax=ax, fraction=0.043, location='right')
plt.savefig("plots/cos_sim.png",bbox_inches='tight')
plt.show()