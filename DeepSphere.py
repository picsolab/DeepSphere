"""
DeepSphere
@ by Xian Teng & Muheng Yan
@ Contact: xian.teng@pitt.edu, yanmuheng@pitt.edu
"""

import tensorflow as tf



class DeepSphere():

	def weights(self, shape):
		return tf.get_variable(name="weight",shape=shape,dtype=tf.float32,initializer=tf.random_normal_initializer())

	def __init__(self, config):
		"""
		config.time_steps
		config.num_dim
		config.num_hidden
		config.keep_prob
		config.lamda
		config.gamma
		config.tolerance
		"""
		self.input = tf.placeholder(tf.float32,shape=[None,config.time_steps,config.num_dim])
		self.batch_size = tf.shape(self.input)[0]

		""" initialize centroid and radius neurons """
		self.centroid = tf.get_variable("centroid", [config.num_hidden], tf.float32, tf.random_normal_initializer())
		self.radius = tf.get_variable("radius", initializer=tf.constant(0.1))
		with tf.variable_scope("att_weights"):
			self.weights = self.weights(shape=[config.time_steps, 1])

		with tf.variable_scope("encoder"):
			inputs = tf.nn.dropout(self.input, config.keep_prob)
			inputs = tf.unstack(inputs,config.time_steps,1)
			""" run encode """
			self.enc_cell = tf.nn.rnn_cell.LSTMCell(config.num_hidden)
			enc_ouputs,enc_state = tf.nn.static_rnn(self.enc_cell,inputs,dtype=tf.float32)
			enc_ouputs = tf.stack(enc_ouputs,axis=0) # [time_steps,batch_size,num_hidden]
			""" weighted sum """
			enc_outputs = tf.transpose(enc_ouputs, perm=[1, 2, 0]) # permutate to [batch_size x num_hidden x time_steps]
			op = lambda x: tf.matmul(x,self.weights)
			z = tf.map_fn(op,enc_outputs)
			z = tf.squeeze(z)

		with tf.variable_scope("hypersphere_learning"):
			self.distance = tf.map_fn(tf.norm,
				(z - tf.reshape(tf.tile(self.centroid,[self.batch_size]),[self.batch_size, config.num_hidden])))
			distanceE2 = tf.square(self.distance)
			residue = tf.nn.relu(distanceE2 - tf.square(self.radius))
			penalty = tf.nn.relu(tf.square(self.radius) - distanceE2)
			penalty = tf.nn.softmax(penalty)

			""" case-level label """
			self.label = (self.radius * (1.0 + config.tolerance) >= self.distance)

		with tf.variable_scope("decoder"):
			# inputs are zeros
			dec_inputs = [tf.zeros([self.batch_size,config.num_dim], dtype=tf.float32) for _ in range(config.time_steps)]
			self.dec_cell = tf.nn.rnn_cell.LSTMCell(config.num_hidden)	
			""" run decoder """
			dec_output,dec_state = tf.nn.static_rnn(self.dec_cell,dec_inputs,initial_state=enc_state,dtype=tf.float32)			
			dec_output = tf.transpose(tf.stack(dec_output[::-1]),perm=[1,0,2]) # [batch_size,time_steps,num_hidden]
			dec_output = tf.layers.dense(inputs=dec_output,units=config.num_dim,activation=None)


		with tf.variable_scope("loss"):
			""" recostruction error """
			self.rec_diff = self.input - dec_output # [batch_size, time_step, num_hidden]
			rec_error = tf.reduce_mean(tf.reduce_mean(tf.pow(self.rec_diff, 2), axis=1), axis=1) # [batch_size x 1]
			penalized_rec_error = tf.reduce_mean(tf.multiply(rec_error, penalty))
			
			""" hypersphere learnong loss """
			hyper_loss = tf.square(self.radius) + config.gamma * tf.reduce_sum(residue) + tf.reduce_mean(distanceE2)
			self.loss = hyper_loss + config.lamda * penalized_rec_error
