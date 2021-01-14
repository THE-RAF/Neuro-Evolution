import neat
import numpy as np
import os
import pickle

class NeatTrainer:
	def __init__(self, env, discrete_action_space=True, eval_n_episodes=1, episode_max_steps=500, render_training=False):
		self.env = env

		self.discrete_action_space = discrete_action_space

		self.eval_n_episodes = eval_n_episodes
		self.episode_max_steps = episode_max_steps

		self.max_fitnesses = []
		self.average_fitnesses = []

		self.render_training = render_training

	def eval_network(self, network, render):
		scores = []
		for episode in range(self.eval_n_episodes):
			state = self.env.reset()

			score = 0
			for step in range(self.episode_max_steps):
				if render:
					self.env.render()

				if self.discrete_action_space:
					action = np.argmax(network.activate(state))
					
				else:
					action = np.array(network.activate(state))

				state, reward, done, info = self.env.step(action)
				score += reward

				if done:
					break

			scores.append(score)

		fitness = np.mean(scores)

		return fitness

	def eval_genomes(self, genomes, config):
		fitnesses = []

		for genome_id, genome in genomes:
			genome.fitness = 0
			net = neat.nn.FeedForwardNetwork.create(genome, config)
			genome.fitness += self.eval_network(net, self.render_training)

			fitnesses.append(genome.fitness)

		self.max_fitnesses.append(max(fitnesses))
		self.average_fitnesses.append(np.mean(fitnesses))

	def train(self, n_generations=10):
		local_dir = os.path.dirname(__file__)
		config_file = os.path.join(local_dir, 'config.txt')

		config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
									neat.DefaultSpeciesSet, neat.DefaultStagnation,
									config_file)

		population = neat.Population(config)

		population.add_reporter(neat.StdOutReporter(True))
		stats = neat.StatisticsReporter()
		population.add_reporter(stats)
		
		winner = population.run(self.eval_genomes, n_generations)
		winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

		return winner_net

	def save_network(self, network, filename):
		with open(filename, 'wb') as pickle_out:
			pickle.dump(network, pickle_out)
