"""
Custom Implementation

Vanilla Policy Gradient
- Currently only tested on CartPole-v0
"""
import tensorflow as tf
import numpy as np
import gym
import utils

def vpg(env_fn):

	env = env_fn()
	act_dim = env.action_space.n
	obs_dim = env.observation_space.shape[0]

	obs_ph = tf.placeholder(
		shape=(None, obs_dim),
		dtype=tf.float32,
		name='obs_ph',
	)
	ret_ph = tf.placeholder(
		shape=(None,),
		dtype=tf.float32,
		name='ret_ph',
	)
	act_ph = tf.placeholder(
		shape=(None,),
		dtype=tf.int32,
		name='act_ph',
	)
	adv_ph = tf.placeholder(
		shape=(None,),
		dtype=tf.float32,
		name='adv_ph',
	)
	logp_old_ph = tf.placeholder(
		shape=(None,),
		dtype=tf.float32,
		name='logp_old_ph',
	)
		
	# Create Value Network
	val_est = utils.mlp(
		inputs=obs_ph,
		hidden_units=[32, 32, 1],
		activation=tf.tanh,
		output_activation=None,
	)
	val_fn_loss = tf.reduce_sum((val_est - ret_ph) ** 2)
	val_train_op = tf.train.AdamOptimizer(1e-3).minimize(val_fn_loss)

	# Create Policy Network
	logits = utils.mlp(
		inputs=obs_ph,
		hidden_units=[32, 32, act_dim],
		activation=tf.tanh,
		output_activation=None,
	)
	pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)
	action_mask = tf.one_hot(act_ph, depth=act_dim)
	logp = tf.reduce_sum(action_mask * tf.nn.log_softmax(logits), axis=1)
	pol_fn_loss = - tf.reduce_mean(logp * adv_ph)
	pol_train_op = tf.train.AdamOptimizer(1e-2).minimize(pol_fn_loss)

	# Init session
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	buf = utils.VPGBuffer(env.observation_space.shape, env.action_space.shape, 5000)
	all_ph = [obs_ph, act_ph, adv_ph, ret_ph, logp_old_ph]

	for i in range(50):

		obs = env.reset()
		done = False

		ep_rewards = []
		rewards = []

		while True:
			a, v = sess.run([pi, val_est], feed_dict={obs_ph: obs.reshape(1, -1)})
			new_obs, r, done, _ = env.step(a[0])
			buf.store(obs, a, r, v, 0)
			rewards.append(r)

			obs = new_obs

			if done or buf.ptr == buf.max_size:
				if buf.ptr == buf.max_size:
					last_val = sess.run(val_est, feed_dict={obs_ph: obs.reshape(1, -1)})
					buf.finish_path(last_val)
					break
				if done:
					obs = env.reset()
					done = False
					last_val = 0
					buf.finish_path(last_val)
					ep_rewards.append(sum(rewards))
					rewards = []

		feed_dict = dict(zip(all_ph, buf.get()))
		sess.run(pol_train_op, feed_dict)
		for _ in range(80):
			sess.run(val_train_op, feed_dict)

		print('Epoch {}\tReturn: {}'.format(i, np.mean(ep_rewards)))


vpg(lambda: gym.make('CartPole-v0'))
			
			


