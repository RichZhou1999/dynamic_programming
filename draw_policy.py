import numpy as np
from charge_env import Charge_env
policy = np.load('simple_case_policy.npy')


state = [0.8, 24,0]
done = False
charge_env = Charge_env()
action_history = [0]*charge_env.state_size_time
while not done:
    state_0_index = charge_env.get_index(state[0])
    action_prob = policy[state_0_index,state[1],state[2],:]
    action = np.argmax(action_prob)
    if action !=0:
        action_history[state[2]] = action
    new_state, reward, done = charge_env.step(state, action)
    state = new_state
print(action_history)
