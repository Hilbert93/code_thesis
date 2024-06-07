import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # just to avoid an error

# analysis_log = {'key':[], 'value':[], 'w_beta':[], 'query':[], 'out':[], 'r_beta':[], 'F':[],
#                            'rnn_out':[], 'rnn_cell':[]}
# test_log = {'reward': [], 'state': [], 'action': [], 'goal': [], 'state_val': []}
# test_log_average = {'episode': [], 'total_average_reward': [], 'current_reward':[]}

def read_in(file_path):
    with open(file_path, 'rb') as f:
        file = pickle.load(f)
    return file
file_path = 'testing/fwm_final/'
file_name = 'test10'
analysis_log = read_in(file_path + file_name+'_analysis.pickle')
test_log = read_in(file_path + file_name+'.pickle')
test_log_avg = read_in(file_path + file_name+'_avg.pickle')
episode = 10 # episode for analysis

# choose an episode to analyse
time_steps = np.arange(0,250)
#print(test_log_avg['current_reward'][episode])
reward_array = np.array(test_log['reward'][episode])
reward_time_steps = np.argwhere(reward_array)
goal_100 = np.argwhere(np.squeeze(reward_time_steps) > 99)
reward_time_steps_100 = reward_time_steps[0:goal_100[0][0]]
goal_location = test_log['goal'][episode]
#print(goal_location)

# Choose plots to show
lstm = False
vectors = False
strength = False
state_val = False
plt.rcParams.update({'font.size': 16})

# hidden and cell states of lstm
rnn_out_array = np.squeeze(analysis_log['rnn_out'][episode])
rnn_cell_array = np.squeeze(analysis_log['rnn_cell'][episode])

if lstm == True:
    fig1, ax = plt.subplots(1,2,figsize=(12,9))
    im1 = ax[0].imshow(rnn_out_array[0:100,:].T)
    for goals in reward_time_steps_100:
        ax[0].axvline(x=goals, ls=':', color='red', linewidth=1)
    ax[0].set_xlabel('Time steps')
    ax[0].set_ylabel('Unit number')

    fig1.colorbar(im1, ax=ax[0], fraction=0.056, location='top')
    im2 = ax[1].imshow(rnn_cell_array[0:100,:].T)
    for goals in reward_time_steps_100:
        ax[1].axvline(x=goals, ls=':', color='red', linewidth=1)
    ax[1].set_xlabel('Unit number')
    fig1.colorbar(im2, ax=ax[1], fraction=0.056, location='top')

    plt.savefig("plots/lstm.png",bbox_inches='tight')
    plt.show()

# writing and reading vectors
key_array = np.squeeze(analysis_log['key'][episode])
value_array = np.squeeze(analysis_log['value'][episode])
query_array = np.squeeze(analysis_log['query'][episode])
out_array = np.squeeze(analysis_log['out'][episode])
value_old_array = np.squeeze(analysis_log['value_old'][episode])
information_array = value_array - value_old_array

if vectors == True:
    fig2, ax = plt.subplots(2,2,figsize=(12,9))
    im1 = ax[0,0].imshow(value_array[0:100,:].T)
    for goals in reward_time_steps_100:
        ax[0,0].axvline(x=goals, ls=':', color='red', linewidth=1)
    ax[0,0].set_title('writing value vector v')
    ax[0,0].set_xlabel('Time steps')
    ax[0,0].set_ylabel('Vector coordinate')
    fig2.colorbar(im1, ax=ax[0,0], fraction=0.016, location='right')

    im2 = ax[0,1].imshow(key_array[0:100,:].T)
    for goals in reward_time_steps_100:
        ax[0,1].axvline(x=goals, ls=':', color='red', linewidth=1)
    ax[0,1].set_title('writing key vector k')
    ax[0,1].set_xlabel('Time steps')
    fig2.colorbar(im2, ax=ax[0,1], fraction=0.016, location='right')

    im3 = ax[1,0].imshow(out_array[0:100,:].T)
    for goals in reward_time_steps_100:
        ax[1,0].axvline(x=goals, ls=':', color='red', linewidth=1)
    ax[1,0].set_title('reading value vector out')
    ax[1,0].set_xlabel('Time steps')
    ax[1,0].set_ylabel('Vector coordinate')
    fig2.colorbar(im3, ax=ax[1,0], fraction=0.016, location='right')

    im4 = ax[1,1].imshow(query_array[0:100,:].T)
    for goals in reward_time_steps_100:
        ax[1,1].axvline(x=goals, ls=':', color='red', linewidth=1)
    ax[1,1].set_title('reading key vector q')
    ax[1,1].set_xlabel('Time steps')
    fig2.colorbar(im4, ax=ax[1,1], fraction=0.016, location='right')

    plt.savefig("plots/vectors.png",bbox_inches='tight')
    plt.show()

# writing and reading strength
w_beta_array = np.squeeze(analysis_log['w_beta'][episode])
r_beta_array = np.squeeze(analysis_log['r_beta'][episode])

if strength == True:
    fig3, ax = plt.subplots(1,2,figsize=(12,9))
    im1 = ax[0].plot(np.arange(0,100,1), w_beta_array[0:100])
    for goals in reward_time_steps_100:
        ax[0].axvline(x=goals, ls=':', color='red', linewidth=1)
    ax[0].set_xlabel('Time steps')
    ax[0].set_ylabel('Writing strength')
    ax[0].set_xlim([0,100])

    im2 = ax[1].plot(np.arange(0,100,1), r_beta_array[0:100])
    for goals in reward_time_steps_100:
        ax[1].axvline(x=goals, ls=':', color='red', linewidth=1)
    ax[1].set_xlabel('Time steps')
    ax[1].set_ylabel('Reading strength')
    ax[1].set_xlim([0,100])

    plt.savefig("plots/strength.png",bbox_inches='tight')
    plt.show()
# state value and reading strength
state_values = np.squeeze(test_log['state_val'][episode])
reading_strength = np.squeeze(analysis_log['r_beta'][episode])

if state_val ==True:
    fig4, ax = plt.subplots(1,1,figsize=(7.5,6.5))
    im1 = ax.scatter(reading_strength, state_values)
    ax.set_xlabel('Reading strength')
    ax.set_ylabel('State value')
    plt.savefig("plots/value_reading.png",bbox_inches='tight')
    plt.show()