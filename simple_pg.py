import tensorflow as tf
import numpy as np
import gym

import utils

def train(env_name='CartPole-v0'):

	env = gym.make(env_name)

	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.n

	obs_ph = tf.placeholder(
		shape=(None, obs_dim),
		dtype=tf.float32,
		name='obs_ph',
	)

	logits = utils.mlp(
		inputs=obs_ph,
		hidden_units=[32, act_dim],
		activation=tf.tanh,
		output_activation=None,
	)

	action = tf.squeeze(tf.multinomial(logits, 1), axis=1)

	act_ph = tf.placeholder(
		shape=(None,),
		dtype=tf.int32,
		name='act_ph',
	)
	weights_ph = tf.placeholder(
		shape=(None,),
		dtype=tf.float32,
		name='weights_ph',
	)
	action_mask = tf.one_hot(act_ph, depth=act_dim)
	logp = tf.reduce_sum(action_mask * tf.nn.log_softmax(logits), axis=1)
	loss = - tf.reduce_mean(logp * weights_ph)
	train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	batch_size = 5000

	for i in np.arange(50):

		batch_obs = []
		batch_act = []
		batch_weights = []
		batch_ret = []
		batch_len = []

		obs = env.reset()
		ep_rewards = []
		done = False

		while True:

			batch_obs.append(obs.copy())
			a = sess.run(action, feed_dict={obs_ph: obs.reshape(1, -1)})[0]

			obs, r, done, _ = env.step(a)

			ep_rewards.append(r)
			batch_act.append(a)

			if done:
				ep_ret = sum(ep_rewards)
				ep_len = len(ep_rewards)
				batch_ret.append(ep_ret)
				batch_len.append(ep_len)

				batch_weights += [ep_ret] * ep_len

				obs = env.reset()
				ep_rewards = []
				done = False

				if len(batch_obs) > batch_size:
					break

		batch_loss, _ = sess.run([loss, train_op], feed_dict={
			obs_ph: batch_obs,
			act_ph: batch_act,
			weights_ph: batch_weights,
		})
		print('Epoch {}\tReturn: {}'.format(i, np.mean(batch_ret)))

train()