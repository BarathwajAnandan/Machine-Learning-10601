from environment import MountainCar
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm


class QLearn:
    def __init__(self, env, mode, epsilon, rate, gamma, max_iterations, episodes):
        self._env = env
        self._mode = mode
        self._epsilon = epsilon
        self._lr = rate
        self._gamma = gamma
        self._max_iterations = max_iterations
        self._episodes = episodes

        self._weights = np.zeros((self._env.state_space, self._env.action_space))
        self._bias = 0
        self._state = np.zeros((self._env.state_space,1))

        self._rewards = []

    def read_state(self, state_dict):

        state = None
        if (self._mode == "tile"):
            state = np.zeros((self._env.state_space, 1))
            state[list(state_dict.keys())] = 1

        elif (self._mode == "raw"):
            state = np.array(list(state_dict.values())).reshape(2, 1)

        return state

    def train(self):
        # pbar = tqdm(total=self._episodes)
        for episode in tqdm(range(self._episodes)):
            self._env.render()
            iter = 0
            done = False
            ep_reward = 0
            self._state = self.read_state(self._env.reset())

            while (not done and iter < self._max_iterations):
                if(np.random.random() < self._epsilon):
                    action = np.random.randint((self._env.action_space))
                else:
                    action = np.argmax(self._state.T@self._weights + self._bias)

                new_state_dict, reward, done = self._env.step(action)
                new_state = self.read_state(new_state_dict)

                q = self._state.T@self._weights[:, action] + self._bias
                discounted_future_reward = np.max(new_state.T@self._weights + self._bias)
                self._weights[:,action] -= self._lr*(q - (reward + self._gamma*discounted_future_reward))*self._state[:,0]

                self._bias -= self._lr * (q - (reward + self._gamma * discounted_future_reward))

                self._state = new_state
                ep_reward += reward
                iter += 1

            self._rewards.append(ep_reward)
            # self._env.render()
        self._env.close()

    def output_data(self, weights_file, rewards_file):
        weights = np.concatenate((self._bias, self._weights.flatten()))

        np.savetxt(weights_file, weights, delimiter='\n')
        np.savetxt(rewards_file, np.array(self._rewards), delimiter='\n')

    def moving_average(self, a, n):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def plot_rewards(self):

        seq = np.arange(1,self._episodes+1)
        mean_rewards = self.moving_average(self._rewards, 25)
        seq2 = np.arange(1,self._episodes+1 - 24)
        # ipdb.set_trace()
        plt.plot(seq, self._rewards, 'b', label="returns per episode")
        plt.plot(seq2, mean_rewards, 'r', label = "rolling mean returns")

        plt.xlabel("Number of Episodes")
        plt.ylabel("Returns")
        plt.title('Raw state representation')
        plt.grid(True)
        plt.legend()
        plt.show()


def main(args):
    mode = "raw" # "tile"  # sys.argv[1]
    weight_out = "./weight_m.out"  # sys.argv[2]
    returns_out = "./returns_m.out" # sys.argv[3]
    episodes = 200 # sys.argv[4]
    max_iterations = 200 # sys.argv[5]
    epsilon = 0.05 # sys.argv[6]
    gamma = 0.999 # sys.argv[7]
    learning_rate = 0.001 # sys.argv[8]

    # mode = str(sys.argv[1])
    # weight_out = sys.argv[2]
    # returns_out = sys.argv[3]
    # episodes = int(sys.argv[4])
    # max_iterations = int(sys.argv[5])
    # epsilon = float(sys.argv[6])
    # gamma = float(sys.argv[7])
    # learning_rate = float(sys.argv[8])

    env = MountainCar(mode)
    qlearn = QLearn(env, mode, epsilon, learning_rate, gamma, max_iterations, episodes)
    qlearn.train()
    qlearn.output_data(weight_out, returns_out)
    qlearn.plot_rewards()

if __name__ == "__main__":
    main(sys.argv)