import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# analysis_log = {'key':[], 'value':[], 'w_beta':[], 'query':[], 'out':[], 'r_beta':[], 'F':[],
#                            'rnn_out':[], 'rnn_cell':[]}
# test_log = {'episode': [], 'episode_time_step': [], 'reward': [], 'state': [], 'action': [], 'goal': [], 'state_val': []}

def read_in(file_path):
    with open(file_path, 'rb') as f:
        file = pickle.load(f)
    return file
file_path = 'testing/fwm_final/'
file_name = 'test10'
analysis_log = read_in(file_path + file_name+'_analysis.pickle')
test_log = read_in(file_path + file_name+'.pickle')
test_log_avg = read_in(file_path + file_name+'_avg.pickle')

#goal in the middle of the maze
'''variable = np.array(test_log['goal'])
goal_location = np.squeeze(variable[:,0])
goal_episodes = np.where(goal_location == 12)
goal_reward = test_log_avg['current_reward'][goal_episodes[0][0]]
print(goal_reward)'''

#episode = goal_episodes[0][0]
episode = 10
actions = test_log['action'][episode]
out_vector = analysis_log['out'][episode]
out_vector_st = StandardScaler().fit_transform(out_vector)

# do PCA on output vector
pca = PCA(n_components=2)
comp = pca.fit_transform(out_vector_st)
principalDf = pd.DataFrame(data = comp
             , columns = ['pc1', 'pc2'])
principalDf['action'] = actions
targets = [0, 1, 2, 3]        # available actions
colors = ['r', 'g', 'b', 'y'] # left, up, right, down

plt.rcParams.update({'font.size': 16})
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
for target, color in zip(targets,colors):
    indicesToKeep = principalDf['action'] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'pc1']
               , principalDf.loc[indicesToKeep, 'pc2']
               , c = color)
plt.savefig("plots/pca.png",bbox_inches='tight')
plt.show()