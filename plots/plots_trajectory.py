import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

# test_log = {'reward': [], 'state': [], 'action': [], 'goal': [], 'state_val': []}
# test_log_average = {'episode': [], 'total_average_reward': [], 'current_reward':[]}
with open('testing/fwm_final/test6.pickle', 'rb') as f:
    test_log = pickle.load(f)
with open('testing/fwm_final/test6_avg.pickle', 'rb') as f:
    test_log_avg = pickle.load(f)

# map state to coordinates
def vector(s):
    x = s%5
    y = -(s//5)
    return x, y

episode = 2
goal_location = (test_log['goal'][episode][0])

reward_array = np.array(test_log['reward'])
print(test_log_avg['current_reward']) # helps to search for different episodes
print(test_log_avg['current_reward'][episode])
goal_time_steps = np.squeeze(np.argwhere(reward_array[episode]))

trial = 0 # start counting with trial 0
if trial == 0:
    path_array = np.array(test_log['state'][episode][0:(goal_time_steps[0]+trial+2)])
else:    
    path_array = np.array(test_log['state'][episode][goal_time_steps[trial-1]+(trial+1):(goal_time_steps[trial]+trial+2)])
print(path_array)
path = vector(path_array)
goal = vector(goal_location)

x, y = np.mgrid[0:5, 0:5]

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'lines.linewidth': 3.0})
plt.rcParams.update({'lines.markersize': 10.0})
fig = plt.figure(figsize=(7,6.5))
#fig.suptitle('Second trajectory of test 1 episode 2')
plt.scatter(x,-y, color='k')
# world 1
plt.plot(x,-y, color='k')
plt.plot(y,-x, color='k')

# world 2
'''x2, y2 = np.mgrid[0:5, 1:4:2]
plt.plot(x,-y, color='k')
plt.plot(y2,-x2, color='k')'''

# world 3
'''x3 = np.array([[0,0],[1,1],[1,1],[2,2],[3,3],[3,3],[4,4],[1,3],[0,1],[3,4],[0,4],[0,1],[3,4],[1,3]])
y3 = np.array([[0,4],[0,1],[3,4],[0,4],[0,1],[3,4],[0,4],[0,0],[1,1],[1,1],[2,2],[3,3],[3,3],[4,4]])
for i in range(len(x3)):
    plt.plot(x3[i],-y3[i], color='k')'''

plt.scatter(path[0][0], path[1][0], color='blue')
plt.text(path[0][0]-0.1, path[1][0]-0.1, 'start', verticalalignment='top', horizontalalignment='right')
plt.scatter(goal[0], goal[1], color='red')
plt.text(goal[0]-0.1, goal[1]-0.1, 'goal', verticalalignment='top', horizontalalignment='right')

colors = plt.cm.jet(np.linspace(0, 1, len(path[0])-1)) # different color for each step

for i in range(len(path[0])-1):
    plt.plot((path[0][i], path[0][i+1]), (path[1][i], path[1][i+1]), color=colors[i])
# customize the colormap
cmap = plt.cm.jet
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
bounds = np.linspace(0, len(path[0])-1, len(path[0]))
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# creating ScalarMappable
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, fraction=0.042)
plt.axis('off')
plt.savefig("plots/goal.png",bbox_inches='tight')
plt.show()