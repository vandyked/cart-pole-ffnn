from net import FFNN
from utils import do_rollout
import numpy as np
import copy
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class FFNNAgent(object):
    def __init__(self, action_space, hyperparams):
        self.action_space = action_space
        np.random.seed(hyperparams['seed'])
        self.hyperparams = hyperparams

        # options learning and hyperparams
        self.gamma = hyperparams['gamma']
        self.batch_size = hyperparams['batch_size']
        self.epsilon = hyperparams['epsilon']
        self.epsilon_min = hyperparams['epsilon_min']
        self.epsilon_decay_rate = hyperparams['epsilon_decay_rate']
        self.target_net_hold_epsiodes = hyperparams['target_net_hold_epsiodes']
        self.learning_rate = hyperparams['learning_rate']
        self.learning_rate_decay = hyperparams['learning_rate_decay']
        self.n_updates_per_episode = hyperparams['n_updates_per_episode']
        self.mem_pool_maxsize = hyperparams['mem_pool_maxsize']  # s,a,r,s',done tuples
        self.n_iter = hyperparams['n_iter']
        self.n_steps = hyperparams['n_steps']
        self.hidden_layer_size = hyperparams['hidden_layer_size']

        # mem pool
        self.mem_pool = []

        # network (ie Q func approximator)
        self.net = FFNN(h=self.hidden_layer_size, lr=self.learning_rate, seed=hyperparams['seed'], VAL=hyperparams['init_net_val'])
        self.net.create()
        self.theta = self.net.params


    def recreate_target_net(self):
        # target net - held fixed for some number of update steps
        self.target_net = copy.copy(self.net)

    def act(self, ob, reward, done):
        '''
        actions in cartpole are just indexes - either 0 or 1
        :param ob:
        :param reward:
        :param done:
        :return: action index (0 or 1)
        '''
        if np.random.uniform() < self.epsilon:
            # behave randomly
            return self.action_space.sample()
        else:
            # greedily
            Qs = self.net.get_Q(np.matrix(ob))
            logger.debug('Q values: %s' % Qs)
            largest_action_index = Qs[0].argsort()[-1]  # Qs[0] -- since can only act on 1 belief state at a time
            return largest_action_index

    def check_mem_pool_size(self):
        if len(self.mem_pool) > self.mem_pool_maxsize:
            self.mem_pool = self.mem_pool[-self.mem_pool_maxsize:]

    def optimise(self, sars_tuples, iter_no):
        '''

        :param sars_tuples:     (s,a,r,s', done)  done tells whether s' is terminal
        :param iter_no:
        :return:
        '''

        # TODO work with (s,a,r,s') sequence to optimise self.net
        self.mem_pool += sars_tuples
        self.check_mem_pool_size()
        # TODO - figure out the learning rule (need a target network I think...)
        # EPSILON:
        if iter_no % self.target_net_hold_epsiodes == 0:
            self.recreate_target_net()  # nb: creates it on iter_no 0. changes it thereafter
            if self.epsilon > self.epsilon_min:      # ALWAYS KEEP EXPLORIN!
                self.epsilon *= self.epsilon_decay_rate
            logger.debug(self.epsilon)
        # LEARNING RATE:
        if iter_no % 5:
            if self.learning_rate > 0.001:
                self.learning_rate *= self.learning_rate_decay   # could do something like test for a little - like a validation set

        # TODO - learning step
        # 1. get batch - randomly pick batch_size samples from the mem_pool
        if self.batch_size < len(self.mem_pool):
            if len(self.mem_pool) < 50:
                n_updates = 1       # dont overtrain on first (inevitably poor) rollouts
            else:
                n_updates = self.n_updates_per_episode
            for _ in xrange(n_updates):
                batch_i = np.random.choice(range(len(self.mem_pool)), size=self.batch_size, replace=False, p=None)
                # 2. get targets - r + gamma * max Q # TODO fix this bad/slow way of accessing data

                #Qtargets = [self.mem_pool[i][2] + self.gamma * np.max(self.target_net.get_Q(np.matrix(self.mem_pool[i][3])))
                ## if self.mem_pool[i][4] is False else self.mem_pool[i][2] for i in batch_i]
                Qtargets = []
                states = []
                for i in batch_i:
                    s,a,r,sprime,done = self.mem_pool[i]  # unpack
                    tar_i = self.net.get_Q(np.matrix(s))[0]            # NOTE prediction from self.net
                    if done:
                        tar_i[a] = r
                    else:
                        tar_i[a] = r + self.gamma * np.max(self.target_net.get_Q(np.matrix(sprime)))    # NOTE prediction from target net
                    Qtargets.append(tar_i)
                    states.append(s)

                # 3. set tf datavariables and call the learning step of model
                train_out = self.net.sgd(minibatch_xs=np.asmatrix(states), minibatch_Qs=np.asmatrix(Qtargets))
                logger.debug('sgd on mse output: %s' % train_out)
        return

    def episode_and_optimise(self, env):
        '''todo
        '''
        for iter_no in range(self.n_iter):
            total_r, t, sars_tuples = do_rollout(self, env, self.n_steps, render=False)
            # Learning step
            self.optimise(sars_tuples, iter_no)
            yield {'total_r' : total_r}

    def print_params(self):
        self.net.print_params()