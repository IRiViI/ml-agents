
import tensorflow as tf

class CustomLayerSpecs():

	def __init__(self):
		pass



class CustomConvLayerSpecs(CustomLayerSpecs):

	def __init__(self, 
		filters, 
		kernel_shape=(3,3), strides=(1,1), 
		activation='elu', 
		kernel_initializer='glorot_uniform',
		bias_initializer='zeros',
		use_bias=True, maxPool=False):

		CustomLayerSpecs.__init__(self)

		self.filters = filters

		self.kernel_shape = kernel_shape
		self.strides = strides

		# if activation == 'elu':
		# 	activation = tf.nn.elu
		# elif activation == 'relu':
		# 	activation = tf.nn.relu
		# else:
		# 	raise TypeError("activation type '{}' is not recognized", activation)
		self.activation = activation

		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer

		self.use_bias = use_bias

		self.maxPool = maxPool

class CustomDenseLayerSpecs(CustomLayerSpecs):

	def __init__(self, 
		nodes, 
		activation='default', 
		kernel_initializer='default',
		bias_initializer='zeros',
		use_bias=True, maxPool=False):

		CustomLayerSpecs.__init__(self)

		self.nodes = nodes

		self.activation = activation

		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer

		self.use_bias = use_bias


                    