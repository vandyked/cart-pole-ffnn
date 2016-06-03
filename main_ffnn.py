import gym
import logging
import json, sys, cPickle, os
from os import path
import argparse
from ffnn_agent import FFNNAgent
from utils import do_rollout, RandomAgent, LRAgent
import numpy as np
try:
    import matplotlib.pyplot as plt
    # stupid catch statement - since matplotlib doesn't work in virtualenv
    # must be run from cmd line via: frameworkpython main_ffnn.py
    doplot = True
except:
    doplot = False

def main_LRAgent():
    # Init and constants:
    env = gym.make('CartPole-v0')
    agent = LRAgent(env.action_space)
    # agent = RandomAgent(env.action_space)

    episode_rewards = []
    for _ in xrange(100):  # "Solving" means averaging over 195 on this
        total_rew, _, _ = do_rollout(agent, env, 200, render=True)
        episode_rewards.append(total_rew)
        print('Episode total reward: %s' % total_rew)
    print('mean TEST reward: %s ' % np.mean(episode_rewards))


def main_FFNNAgent():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true')
    args = parser.parse_args()

    # Init and constants:
    env = gym.make('CartPole-v0')
    outdir = '/tmp/ffnn-agent-results'

    # Hyperparams:  learning options, network structure, number iterations & steps,
    hyperparams = {}
    # ----------- NOT worth playing with:
    hyperparams['gamma'] = 1.0
    hyperparams['hidden_layer_size'] = 4
    hyperparams['n_steps'] = 200
    hyperparams['seed'] = 0  # 0
    # ----------- worth playing with:  (current best settings in comments)
    hyperparams['init_net_val'] = 0.05  # 0.05
    hyperparams['batch_size'] = 50  # 50
    hyperparams['epsilon'] = 1.0  # 1 - starting value
    hyperparams['epsilon_min'] = 0.1  # 0.1  - always keep exploring?
    hyperparams['epsilon_decay_rate'] = 0.98  # 95    # 0.98         ~.995 over 500 its leaves it at 0.08
    # -- observation is that exploration/exploitation trade off is very important
    hyperparams['target_net_hold_epsiodes'] = 1  # 1
    hyperparams['learning_rate'] = 0.15  # 0.15
    hyperparams['learning_rate_decay'] = 0.9  # 0.9
    hyperparams['n_updates_per_episode'] = 1  # 1 - means pick X random minibatches, doing GradDescent on each
    hyperparams['mem_pool_maxsize'] = 100  # 100 - number of (s,a,r,s',done) tuples -- ~big seems bad
    hyperparams['n_iter'] = 1000 / 1  # 1000
    # ------------------------------------  BEST SETTINGS GIVE: test mean: 200 +- 0

    # FFNN agent:
    agent = FFNNAgent(env.action_space, hyperparams)

    # start recording:
    env.monitor.start(outdir, force=True)

    # Prepare snapshotting
    # ----------------------------------------
    def writefile(fname, s):
        with open(path.join(outdir, fname), 'w') as fh: fh.write(s)

    # print all starting params:
    agent.print_params()

    # Train the agent, and snapshot each stage
    train_episode_rewards = []
    for (i, iterdata) in enumerate(agent.episode_and_optimise(env)):
        logger.debug('Iteration %2i. Episode mean reward: %7.3f' % (i, iterdata['total_r']))
        train_episode_rewards.append(iterdata['total_r'])
        # agent = RandomAgent(env.action_space)

        if args.display: do_rollout(agent, env, hyperparams['n_steps'], render=True)
        # writefile('FFNNAgent-%.4i.pkl'%i, cPickle.dumps(agent, -1))        # nested class error still in pickle'ing
    logger.info('mean TRAIN reward: %s ' % np.mean(train_episode_rewards))

    # Print all learnt params
    agent.print_params()

    # testing (ie be greedy)
    agent.epsilon = 0  # explore 0% of the time
    test_episode_rewards = []
    for _ in xrange(100):  # "Solving" means averaging over 195 on this
        total_rew, _, _ = do_rollout(agent, env, hyperparams['n_steps'], render=True)
        test_episode_rewards.append(total_rew)
        logger.debug('Episode total reward: %s' % total_rew)
    logger.info('mean TEST reward: %s ' % np.mean(test_episode_rewards))

    # plot learning curve
    if doplot:
        plt.plot(train_episode_rewards)
        plt.xlabel('Episode number')
        plt.suptitle(str(hyperparams))
        plt.title('Training -')
        plt.ylabel('Episode total rewards (time standing)')
        plt.show()

    # Write out the env at the end so we store the parameters of this
    # environment.
    info = {}
    info['params'] = hyperparams
    info['argv'] = sys.argv
    info['env_id'] = env.spec.id
    writefile('info.json', json.dumps(info))

    # Stop recording
    env.monitor.close()
    logger.info("Successfully ran FFNN agent")

    # logger.info("Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir, algorithm_id='ffnn')


if __name__ == '__main__':
    main_FFNNAgent()
    #main_LRAgent()