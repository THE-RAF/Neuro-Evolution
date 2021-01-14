from NEAT_agent import NeatTrainer
import gym
import pickle


with open('winner_net', 'rb') as network_file:
	network = pickle.load(network_file)

env = gym.make('CartPole-v0')

trainer = NeatTrainer(env=env, discrete_action_space=True, eval_n_episodes=3, episode_max_steps=1000, render_training=False)
trainer.eval_network(network, render=True)

env.close()