from NEAT_agent import NeatTrainer

import gym

import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn')


env = gym.make('CartPole-v0')

trainer = NeatTrainer(env=env, discrete_action_space=True, eval_n_episodes=3, episode_max_steps=1000, render_training=False)

final_net = trainer.train(n_generations=50)

trainer.eval_network(final_net, render=True)

env.close()

plt.plot(trainer.max_fitnesses)
plt.plot(trainer.average_fitnesses)
plt.show()

trainer.save_network(final_net, filename='winner_net')
