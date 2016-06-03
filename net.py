import tensorflow as tf

class FFNN(object):
    def __init__(self, h=4, lr=0.01, use_bias=True, seed=0, VAL=0.1):
        # NOTE - using the bias seems to be essential based on a few runs without it
        tf.set_random_seed(seed)
        self.params = None  # TODO write network in tensorflow
        self.lr = lr
        self.h_size = 4
        # env params - writing this for cartpole
        self.obs_size = 4
        self.actions_size = 2

        # network
        self.state = tf.placeholder(tf.float32, [None, self.obs_size])      # None means dimension can be any length
        ## -- so we can input batches of obs (size 4)
        #self.Wxh = tf.Variable(tf.zeros([self.obs_size, self.h_size]))       # x is the input state
        #self.bxh = tf.Variable(tf.zeros([self.h_size]))
        self.Wxh = tf.Variable(tf.random_uniform(shape=[self.obs_size, self.h_size], minval=-VAL, maxval=VAL), name='Wxh')  # x is the input state
        self.bxh = tf.Variable(tf.random_uniform(shape=[self.h_size], minval=-VAL, maxval=VAL), name='bxh')
        if use_bias:
            self.h = tf.nn.tanh(tf.matmul(self.state, self.Wxh) + self.bxh)
        else:
            self.h = tf.nn.tanh( tf.matmul(self.state, self.Wxh) )
        self.Whq = tf.Variable(tf.random_uniform(shape=[self.h_size, self.actions_size],minval=-VAL, maxval=VAL), name='Whq')
        self.bhq = tf.Variable(tf.random_uniform(shape=[self.actions_size],minval=-VAL, maxval=VAL), name='bhq')  # try with no hidden layer to start

        # Tried without bias on Q - since intuitively there shouldnt be a pref for either action - symmetric problem
        # --> seemed to fail miserably
        if use_bias:
            self.Q = tf.matmul(self.h, self.Whq) + self.bhq
        else:
            self.Q = tf.matmul(self.h, self.Whq)

        self.vars = [self.Wxh, self.bxh, self.Whq, self.bhq]

        # TODO CONSIDER: Clipping? Variable initialisation?
        # targets
        self.Qtar = tf.placeholder(tf.float32, [None, self.actions_size])
        # loss
        self.dif = self.Qtar - self.Q
        self.mse = tf.reduce_mean(tf.square(self.dif))        # TODO - check reduction_indices , reduction_indices=[1]
        #self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(loss=self.mse,var_list=self.vars)
        self.optimiser = tf.train.GradientDescentOptimizer(self.lr)
        self.train = self.optimiser.minimize(self.mse, var_list=self.vars)


        #self.grads_and_vars = self.optimiser.compute_gradients(loss=self.mse, var_list=self.vars)
        #self.grad_placeholder = [(tf.placeholder("float", shape=grad[1].get_shape()), grad[1]) for grad in self.grads_and_vars]
        #self.apply_placeholder_op = self.optimiser.apply_gradients(self.grad_placeholder)
        #self.apply_grads = self.optimiser.apply_gradients(self.grads_and_vars)
        # alternative method to slicing is to just pass in the targets with only the Q value for the taken action
        # being different - this way the --> DOING THIS NOW


    """def get_slice(self, obs, action_index):
        return self.sess.run(self.Q_slice, feed_dict={self.state:obs, self.action_index:action_index})
    """

    def create(self):
        '''
        Created this function because deepcopy had a problem (in creating target network) but actually copy works
        -- check that target net and net are not the same after an update of net --
        :return:
        '''
        # create all variables:
        init = tf.initialize_all_variables()
        # create session:
        self.sess = tf.Session()
        self.sess.run(init)

    def get_Q(self, obs):
        # set self.state
        return self.sess.run(self.Q, feed_dict={self.state: obs})

    def sgd(self, minibatch_xs, minibatch_Qs):
        #train_output = self.sess.run(self.train_step, feed_dict={self.state: minibatch_xs, self.Qtar: minibatch_Qs})
        #wxh_pre = self.sess.run(self.Wxh)
        #bhq_pre = self.sess.run(self.bhq)
        #grad_vals = self.sess.run(self.grads_and_vars, feed_dict={self.state: minibatch_xs, self.Qtar: minibatch_Qs})
        #self.sess.run(self.apply_placeholder_op, feed_dict={self.grad_placeholder:grad_vals})


        # debug check - qs = self.sess.run(self.Q, feed_dict={self.state: minibatch_xs})
        self.sess.run(self.train, feed_dict={self.state: minibatch_xs, self.Qtar: minibatch_Qs})
        #wxh_post = self.sess.run(self.Wxh)
        #bhq_post = self.sess.run(self.bhq)
        #return train_output
        #change = wxh_post-wxh_pre
        #logger.debug('Wxh params changed by %s' % change)
        #change = bhq_post - bhq_pre
        #logger.debug('bhq params changed by %s' % change)
        return

    def get_params(self):
        params = {}
        for p in self.vars:
            val = self.sess.run(p)
            params[p.name] = val
        return params

    def print_params(self):
        params = self.get_params()
        for name,val in params.iteritems():
            print '{}: \n{}\n'.format(name, val)
        return
