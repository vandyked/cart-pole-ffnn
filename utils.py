
class RandomAgent(object):
    '''Just to test environment API setup is correct
    '''
    def __init__(self, action_space):
        self.action_space = action_space
    def act(self, ob, reward, done):
        return self.action_space.sample()

class LRAgent(object):
    '''Left-Right Agent
    '''
    def __init__(self, action_space):
        self.action_space = action_space
        self.last = 0
    def act(self, ob, reward, done):
        self.last = (self.last + 1)%2
        return self.last



def do_rollout(agent, env, num_steps, render=False):
    '''
     generic function (modified from gym examples) that will work for most agent:
    :param agent:
    :param env:
    :param num_steps:
    :param render:
    :return:
    '''
    total_rew = 0
    # Initial observations from environment
    ob = env.reset()
    reward = 0
    done = False
    sars_tuples = []

    for t in range(num_steps):
        current_tuple = []
        a = agent.act(ob, reward, done)
        current_tuple += [ob, a]
        (ob, reward, done, _info) = env.step(a)
        current_tuple += [reward, ob, done]
        sars_tuples.append(tuple(current_tuple))
        total_rew += reward
        if render and t%3==0: env.render()
        if done: break
    return total_rew, t+1, sars_tuples


