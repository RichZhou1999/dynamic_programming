import numpy as np

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
state_size_delta_time = 24
state_size_time = 48



# current_list = np.linspace(0,40,action_size)
# print(current_list)

price_max_value = 1
x = np.linspace(0, int(state_size_time)-1, int(state_size_time))
price_curve = price_max_value/((state_size_time/2)**2) * (x-(state_size_time/2))**2

initial_state = [0.5, 287, 0]
action_space = [i for i in range(action_size)]


def step(state, action):
    new_state = [0, 0, 0]
    delta_soc = np.round(current_list[action] * charge_interval / battery_volume / 60, 1)
    new_state[0] = state[0] - delta_soc
    new_state[1] = state[1] - 1
    new_state[2] = state[2] + 1

    #     new_state[0] = np.round(state[0]*(1+increase_rate_tumor/100),1)
    #     new_state[1] = np.round(state[1]*(1+increase_rate_bad_feeling/100),1)
    #     new_state[0] = min(10, new_state[0])
    #     new_state[1] = min(10, new_state[1])
    #     new_state[0] = max(1, new_state[0])
    #     new_state[1] = max(1, new_state[1])
    #     new_state[2] = max(drugB_usage, state[2])
    if new_state[1] < 0:
        raise Exception("delta time out of bound")
    if new_state[2] >= state_size_time:
        raise Exception("current time out of bound")


    reward = -delta_soc * price_curve[state[2]]
    done = False
    if ((new_state[0] <= 0) or (new_state[1] <= 0) or (new_state[2] >= state_size_time)):
        if new_state[0] > 0:
            reward += -100
        done = True
    return new_state, reward, done

def get_index(state):
    index = int(state/0.1)
    return index


def compute_state_value(max_iter=9, discount=0.95, policy=actions_prob * np.ones(
    (state_size_delta_soc, state_size_delta_time, state_size_time, action_size))):
    new_state_values = np.zeros((state_size_delta_soc, state_size_delta_time, state_size_time))
    iteration = 0

    while iteration <= max_iter:
        state_values = new_state_values.copy()
        old_state_values = state_values.copy()

        for i in np.linspace(0, 1, state_size_delta_soc):
            print(i)
            for j in range(1, int(state_size_delta_time)):
                for m in range(int(state_size_time)-1):
                    i = np.round(i, 1)
                    j = np.round(j, 1)
                    index_i = get_index(i)
                    value = 0
                    for k, a in enumerate(actions):
                        (next_i, next_j, next_m), reward, done = step([i, j, m], a)
                        next_index_i = get_index(next_i)
                        value += policy[index_i, j, m, k] * (
                                    reward + discount * state_values[next_index_i, next_j, next_m])
                    new_state_values[index_i, j, m] = value

        iteration += 1

    return new_state_values, iteration

values, sync_iteration = compute_state_value(max_iter=3)
print(values)