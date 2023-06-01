import numpy as np
from itertools import count
from datetime import datetime

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import gym
import random
import math
import scipy.optimize
from collections import namedtuple

writer = SummaryWriter("./tb_record_simhash_continuous")
Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward', 'key'))


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


class SimHash(object):
    def __init__(self, k, D, beta):
        self.k = k
        self.D = D
        self.beta = beta
        self.A = np.random.normal(0, 1, (k, D))
        self.hash_table = {}
        self.new_hash_table = {}

    def get_keys(self, states):
        key = (np.asarray(np.sign(self.A @ states), dtype=int) + 1) // 2  # to binary code array
        key = int(''.join(key.astype(str).tolist()), base=2)  # to int (binary)
        if key in self.hash_table:
            self.hash_table[key] += 1
        else:
            self.hash_table[key] = 1
        return key

    def get_bonus(self, key):
        cnt = np.array(self.hash_table[int(key)])
        return self.beta * np.reciprocal(np.sqrt(cnt))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 32)
        self.affine2 = nn.Linear(32, 32)

        self.action_mean = nn.Linear(32, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class Critic(nn.Module):
    def __init__(self, num_inputs):
        super(Critic, self).__init__()
        # self.affine1 = nn.Linear(num_inputs, 32)
        # self.affine2 = nn.Linear(32, 32)
        self.value_head = nn.Linear(num_inputs, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        # x = torch.tanh(self.affine1(x))
        # x = torch.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


class TRPO:
    def __init__(self, env, k=32, log_interval=1, eval_interval=50,
                 gamma=0.99, tau=1, l2_reg=1e-2, max_kl=1e-2, damping=1e-1, batch_size=5000):
        self.env = env
        self.num_inputs = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.l2_reg = l2_reg
        self.max_kl = max_kl
        self.damping = damping
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.eval_interval = eval_interval

        self.actor = Actor(self.num_inputs, self.num_actions)
        self.critic = Critic(self.num_inputs)
        self.simhash = SimHash(k, self.num_inputs, 0.01)

    def select_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        action_mean, _, action_std = self.actor(Variable(state))
        action = torch.normal(action_mean, action_std)
        return action

    @staticmethod
    def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = Avp(p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    @staticmethod
    def linesearch(model,
                   f,
                   x,
                   fullstep,
                   expected_improve_rate,
                   max_backtracks=10,
                   accept_ratio=.1):
        fval = f(True).data
        # print("fval before", fval.item())
        for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep
            set_flat_params_to(model, xnew)
            newfval = f(True).data
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            # print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                # print("fval after", newfval.item())
                return True, xnew
        return False, x

    @staticmethod
    def trpo_step(model, get_loss, get_kl, max_kl, damping):
        loss = get_loss()
        grads = torch.autograd.grad(loss, model.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

        def Fvp(v):
            kl = get_kl()
            kl = kl.mean()

            grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * Variable(v)).sum()
            grads = torch.autograd.grad(kl_v, model.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

            return flat_grad_grad_kl + v * damping

        stepdir = TRPO.conjugate_gradients(Fvp, -loss_grad, 10)

        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

        lm = torch.sqrt(shs / max_kl)
        fullstep = stepdir / lm[0]

        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        # print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))

        prev_params = get_flat_params_from(model)
        success, new_params = TRPO.linesearch(model, get_loss, prev_params, fullstep,
                                         neggdotstepdir / lm[0])
        set_flat_params_to(model, new_params)

        return loss

    def update_params(self, batch):
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)
        actions = torch.Tensor(np.concatenate(batch.action, 0))
        states = torch.Tensor(batch.state)
        keys = batch.key
        values = self.critic(Variable(states))

        returns = torch.Tensor(actions.size(0), 1)
        deltas = torch.Tensor(actions.size(0), 1)
        advantages = torch.Tensor(actions.size(0), 1)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.simhash.get_bonus(keys[i]) + self.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values.data[i]
            advantages[i] = deltas[i] + self.gamma * self.tau * prev_advantage * masks[i]

            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]

        returns = (returns - returns.mean()) / returns.std()
        targets = Variable(returns)

        # Original code uses the same LBFGS to optimize the value loss
        def get_value_loss(flat_params):
            set_flat_params_to(self.critic, torch.Tensor(flat_params))
            for param in self.critic.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            values_ = self.critic(Variable(states))

            value_loss = (values_ - targets).pow(2).mean()

            # weight decay
            for param in self.critic.parameters():
                value_loss += param.pow(2).sum() * self.l2_reg
            value_loss.backward()
            return value_loss.data.double().numpy(), get_flat_grad_from(self.critic).data.double().numpy()

        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                                get_flat_params_from(self.critic).double().numpy(),
                                                                maxiter=25)
        set_flat_params_to(self.critic, torch.Tensor(flat_params))

        # advantages = (advantages - advantages.mean()) / advantages.std()

        action_means, action_log_stds, action_stds = self.actor(Variable(states))
        fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

        def get_loss(volatile=False):
            if volatile:
                with torch.no_grad():
                    action_means, action_log_stds, action_stds = self.actor(Variable(states))
            else:
                action_means, action_log_stds, action_stds = self.actor(Variable(states))

            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()

        def get_kl():
            mean1, log_std1, std1 = self.actor(Variable(states))

            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)

        TRPO.trpo_step(self.actor, get_loss, get_kl, self.max_kl, self.damping)

    def train(self, num_epoch=500):
        for i_episode in range(1, num_epoch+1):
            memory = Memory()
            self.actor.train()
            self.critic.train()

            num_steps = 0
            reward_batch = 0
            num_episodes = 0
            while num_steps < self.batch_size:
                state = env.reset()

                reward_sum = 0
                for t in range(10000):  # Don't infinite loop while learning
                    action = self.select_action(state)
                    action = action.data[0].numpy()
                    next_state, reward, done, _ = env.step(action)
                    reward = 1 if (done and t < 499) else 0
                    reward_sum += reward

                    mask = 1
                    if done:
                        mask = 0

                    key = self.simhash.get_keys(state)
                    memory.push(state, np.array([action]), mask, next_state, reward, key)

                    if done:
                        break

                    state = next_state
                num_steps += (t - 1)
                num_episodes += 1
                reward_batch += reward_sum

            reward_batch /= num_episodes
            batch = memory.sample()
            self.update_params(batch)

            tag = '{}/epoch-average-reward'.format(self.env.unwrapped.spec.id)
            writer.add_scalar(tag, reward_batch, i_episode)
            if i_episode % self.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward {:.2f}'.format(
                    i_episode, reward_sum, reward_batch))
            if i_episode % self.eval_interval == 0:
                self.eval(1)

    def eval(self, num_episode):
        self.actor.eval()
        self.critic.eval()
        for i in range(num_episode):
            state = self.env.reset()
            while True:
                self.env.render()
                action = self.select_action(state)
                action = action.data[0].numpy()
                next_state, _, done, _ = self.env.step(action)
                state = next_state
                if done:
                    break


if __name__ == '__main__':
    gym.envs.register(
        id='MyMountainCar',
        entry_point='gym.envs.classic_control:Continuous_MountainCarEnv',
        max_episode_steps=500
    )
    env = gym.make("MyMountainCar")
    seed = 33
    env.seed(seed)
    torch.manual_seed(seed)
    agent = TRPO(env)
    agent.train()
