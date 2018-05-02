from DeepSphere import DeepSphere
import tensorflow as tf
import _pickle
import numpy as np
from sklearn.metrics import accuracy_score
import time
from datetime import datetime
import os


def rmse(predictions, targets):
	return np.sqrt(np.mean((predictions - targets) ** 2))

def diff_convert(time_steps,num_nodes,diff_dict):
	num_samples = len(diff_dict)
	diff = np.zeros((num_samples,time_steps,num_nodes,num_nodes))
	for m in range(num_samples):
		if diff_dict[m] == {}:
			continue # no anomalous pixels
		else:
			coord = diff_dict[m]["coord"]
			value = diff_dict[m]["value"]
			for s in range(len(coord[0])):
				i,j,k = coord[0][s],coord[1][s],coord[2][s]
				diff[m,i,j,k] = value[s]
	return diff

class Config():
	def __init__(self, time_steps, num_dim):
		self.time_steps = time_steps
		self.num_dim = num_dim
		self.num_hidden = 30
		self.keep_prob = 0.98
		self.lamda = 2.0
		self.gamma = 0.05
		self.tolerance = 1e-1
		self.num_epochs = 300

if __name__=="__main__":
	
	dirname = "save"
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	""" load data """
	train = _pickle.load(open("data/train.pkl","rb"))
	test = _pickle.load(open("data/test.pkl","rb"))

	train_data,train_label = train['data'],train['label']
	test_data,test_label,test_diff = test['data'],test["label"],test["diff"]

	_,time_steps,num_nodes,_ = train_data.shape
	num_dim = num_nodes * num_nodes

	train_data = train_data.reshape((train_data.shape[0],time_steps,num_dim))
	test_data = test_data.reshape((test_data.shape[0],time_steps,num_dim))
	test_diff = diff_convert(time_steps,num_nodes,test["diff"])
	test_diff = test_diff.reshape((test_diff.shape[0],time_steps,num_dim))


	""" configuration """
	config = Config(time_steps,num_dim)
	tf.set_random_seed(1000)
	np.random.seed(1000)


	""" initialize neural network """
	model = DeepSphere(config)
	train_op = tf.train.AdamOptimizer(1e-3).minimize(model.loss)    
	
	""" Session """
	with tf.Session() as sess:
		summary_writer = tf.summary.FileWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)
		merged = tf.summary.merge_all()
		saver = tf.train.Saver(tf.global_variables(),max_to_keep=1)

		sess.run(tf.global_variables_initializer())

		start = time.time()

		""" loop epoch """
		for e in range(config.num_epochs):
			
			""" train """
			_,train_loss,train_label_pred,r,d,penalty = sess.run(
				[train_op, model.loss, model.label, model.radius, model.distance, model.penalty],feed_dict={model.input:train_data})
			train_acc = accuracy_score(train_label,train_label_pred)
			end = time.time()
			if e % 10 == 0:
				print("epoch {}/{},loss = {},train_acc = {},time = {}".format(
					e,config.num_epochs,train_loss,train_acc,end-start))
				print(np.c_[np.array(r-d),np.array(penalty)]) # print r-d and penalty side by side
			start = time.time()

		""" test """
		test_loss, test_label_pred, test_diff_pred = sess.run([model.loss, model.label, model.rec_diff], 
			feed_dict={model.input:test_data})

		test_acc = accuracy_score(test_label,test_label_pred)
		rmse_error = rmse(test_diff,test_diff_pred)

		print("test_acc = {}, rmse = {}".format(test_acc,rmse_error))
		_pickle.dump(test_diff_pred,open("data/test_diff_pred.pkl","wb"))
