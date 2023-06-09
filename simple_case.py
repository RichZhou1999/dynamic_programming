import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

voltage = 240
battery_volume = 60 * 1000 / voltage #amp
resistance = 0
power_boundary = 9.6 * 1000
power_boundary_decrease_point = 0.8

action_size = 2
actions_prob = 1/action_size
actions = [i for i in range(action_size)]
charge_interval = 10

# state_size_delta_soc = 101
# state_size_delta_time = int((24*60/charge_interval))
# state_size_time = int((24*60/charge_interval) * 2)

state_size_delta_soc = 11
state_size_delta_time = 48
state_size_time = 96



# current_list = np.linspace(0,40,action_size)
# print(current_list)

price_max_value = 1
x = np.linspace(0, int(state_size_delta_time)-1, int(state_size_delta_time))
price_curve = price_max_value/((state_size_delta_time/2)**2) * (x-(state_size_delta_time/2))**2
price_curve = np.concatenate((price_curve, price_curve),axis =0)

action_space = [i for i in range(action_size)]


def is_terminal(state):
    if ((state[0] == 0) or(state[1] == 0) or (state[2] == state_size_time-1)):
        return True
    return False


def step(state, action):
    if is_terminal(state):
        # if state[0] == 0:
        #     return state, 0, True
        return state, -state[0]*100, True
    new_state = [0, 0, 0]

    if action == 1:
        new_state[0] = state[0] - 0.1
    else:
        new_state[0] = state[0]
    new_state[1] = state[1] - 1
    new_state[2] = state[2] + 1

    #     new_state[0] = np.round(state[0]*(1+increase_rate_tumor/100),1)
    #     new_state[1] = np.round(state[1]*(1+increase_rate_bad_feeling/100),1)
    #     new_state[0] = min(10, new_state[0])
    #     new_state[1] = min(10, new_state[1])
    #     new_state[0] = max(1, new_state[0])
    #     new_state[1] = max(1, new_state[1])
    #     new_state[2] = max(drugB_usage, state[2])
    new_state[1] = max(new_state[1], 0)
    new_state[2] = min(new_state[2], state_size_time - 1)

    if new_state[1] < 0:
        raise Exception("delta time out of bound")
    if new_state[2] >= state_size_time:
        raise Exception("current time out of bound")

    reward = - action * price_curve[state[2]]
    new_state[0] = np.round(new_state[0], 2)
    done = False
    if ((new_state[0] <= 0) or (new_state[1] <= 0) or (new_state[2] >= state_size_time)):
        done = True
        reward += -state[0]*100
    return new_state, reward, done

def get_index(state):
    index = int(np.round(state/0.1))
    return index


def compute_state_value(max_iter=9, discount=1, policy=actions_prob * np.ones(
    (state_size_delta_soc, state_size_delta_time, state_size_time, action_size))):
    new_state_values = np.zeros((state_size_delta_soc, state_size_delta_time, state_size_time))
    iteration = 0

    # while iteration <= max_iter:
    # for _ in tqdm(range(max_iter)):
    for _ in range(max_iter):
        state_values = new_state_values.copy()
        old_state_values = state_values.copy()

        for i in np.linspace(0, 1, state_size_delta_soc):
            for j in range(int(state_size_delta_time)):
                for m in range(int(state_size_time) - j):
                    i = np.round(i, 1)
                    # print(i)
                    j = np.round(j, 1)
                    index_i = get_index(i)
                    # print(index_i)
                    value = 0
                    for k, a in enumerate(actions):
                        (next_i, next_j, next_m), reward, done = step([i, j, m], a)
                        next_index_i = get_index(next_i)
                        value += policy[index_i, j, m, k] * (
                                    reward + discount * state_values[next_index_i, next_j, next_m])
                    new_state_values[index_i, j, m] = value

        # iteration += 1

    return new_state_values, iteration



def greedy_Policy(values,discount = 1):
    new_state_values = values
    policy = np.zeros((state_size_delta_soc, state_size_delta_time, state_size_time, action_size))

    state_values = new_state_values.copy()

    for i in np.linspace(0, 1, state_size_delta_soc):
        for j in range((int(state_size_delta_time))):
            for m in range(int(state_size_time) - j):
                i = np.round(i, 1)
                index_i = get_index(i)
                value = np.min(values);
                for k,a in enumerate(actions):
                    (next_i, next_j, next_m), reward, done = step([i, j, m], a)
                    next_index_i = get_index(next_i)
                    valtemp = reward + discount*state_values[next_index_i, next_j, next_m]
                    if valtemp > value:
                        value = valtemp
                        actionind = k


                policy[index_i,j,m,actionind] = 1

    return policy

# get_index(0.3)
# print(int(np.round(0.3/0.1)))
# print(get_index(0.3))
# values, sync_iteration = compute_state_value(max_iter=30)
# print(values)

num_iteration = 10
policy = actions_prob * np.ones((state_size_delta_soc, state_size_delta_time, state_size_time, action_size))
for i in tqdm(range(num_iteration)):
    values, sync_iteration = compute_state_value(max_iter=10, policy = policy)
    policy = greedy_Policy(values)

np.save(f'greedy_policy_iter{num_iteration}', policy)

# values, sync_iteration = compute_state_value(max_iter=5, policy = greedy_policy)
# greedy_policy = greedy_Policy(values)
# values, sync_iteration = compute_state_value(max_iter=5, policy = greedy_policy)
# greedy_policy = greedy_Policy(values)


# values, sync_iteration = compute_state_value(max_iter=5, policy = greedy_policy)
# greedy_policy = greedy_Policy(values)
# values, sync_iteration = compute_state_value(max_iter=5, policy = greedy_policy)
# greedy_policy = greedy_Policy(values)
# values, sync_iteration = compute_state_value(max_iter=5, policy = greedy_policy)
# greedy_policy = greedy_Policy(values)

# np.save(f'greedy_policy_iter{sync_iteration}', greedy_policy)

# plt.matshow(greedy_policy[:, :, 95, 1])
# plt.show()
