import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from skimage.color import rgb2gray
from skimage.transform import resize
from collections import namedtuple
from datetime import datetime
import numpy as np
import pathlib
import gym

algo = 'atari_baseline'
writer = SummaryWriter(f"./tb_record_{algo}")
Transition = namedtuple('Transition', ('state', 'action', 'reward'))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def preprocess_image(img):
    img = rgb2gray(img)
    img = resize(img, (52, 52))
    img = img * 2 - 1
    return img


class Actor(nn.Module):
    def __init__(self, num_outputs):
        super(Actor, self).__init__()

        self.layer1 = nn.Conv2d(4, 16, 8, stride=4, padding=3)
        self.layer2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.layer3 = nn.Linear(6 * 6 * 32, 256)
        self.layer4 = nn.Linear(256, num_outputs)

        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()

    def forward(self, inputs):
        inputs = inputs.to(device)

        x = self.layer1(inputs)
        x = self.batch_norm1(F.relu(x))
        x = self.layer2(x)
        x = self.batch_norm2(F.relu(x))
        x = self.flatten(x)

        x = self.layer3(x)
        x = F.relu(x)
        x = F.softmax(self.layer4(x), dim=1)
        return x.to('cpu')


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.layer1 = nn.Conv2d(4, 16, 8, stride=4, padding=3)
        self.layer2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.layer3 = nn.Linear(6 * 6 * 32, 256)
        self.layer4 = nn.Linear(256, 1)

        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()

    def forward(self, inputs):
        inputs = inputs.to(device)

        x = self.layer1(inputs)
        x = self.batch_norm1(F.relu(x))
        x = self.layer2(x)
        x = self.batch_norm2(F.relu(x))
        x = self.flatten(x)

        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        return x.to('cpu')


class TRPO(object):
    def __init__(self, env, env_eval, gamma=0.995, lr_c=1e-3, save_folder=None):
        self.env = env
        self.env_eval = env_eval
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.n

        self.actor = Actor(num_outputs=self.num_actions)
        self.critic = Critic()
        self.actor.to(device)
        self.critic.to(device)

        self.critic_loss_func = nn.MSELoss()
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_c)
        self.gamma = gamma
        self.memory = []

        if save_folder is None:
            name = self.env.unwrapped.spec.id.split('/')[-1].split('-')[0]
            self.save_folder = f'pretrain/{algo}/' + name
        else:
            self.save_folder = save_folder
        pathlib.Path(self.save_folder).mkdir(parents=True, exist_ok=True)
        self.actor_path = self.save_folder + '/actor'
        self.critic_path = self.save_folder + '/critic'

    def select_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state).float().unsqueeze(0)
            dist = Categorical(self.actor(state))
            return dist.sample().item()

    def update_agent(self, update_step, delta=0.01, backtrack_ratio=0.8, max_backtracks=15, damping=1e-1):
        states = torch.cat([tr.state for tr in self.memory], dim=0).float()
        actions = torch.cat([tr.action for tr in self.memory], dim=0).flatten()

        returns = []
        for tr in self.memory:
            R = 0
            tr_returns = []
            rewards = tr.reward
            for reward in rewards[::-1]:
                R = reward + self.gamma * R
                tr_returns.append(R)
            tr_returns = torch.as_tensor(tr_returns[::-1]).unsqueeze(1)
            returns.append(tr_returns)
        returns = torch.cat(returns, dim=0).float()
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        idx = np.arange(returns.numpy().shape[0])
        np.random.shuffle(idx)
        idx = idx[:update_step]
        states = torch.from_numpy(states.numpy()[idx])
        actions = torch.from_numpy(actions.numpy()[idx])
        returns = torch.from_numpy(returns.numpy()[idx])

        baselines = self.critic(states)
        self.critic_optimizer.zero_grad()
        value_loss = self.critic_loss_func(baselines, returns)
        value_loss.backward()
        self.critic_optimizer.step()
        with torch.no_grad():
            baselines = self.critic(states)

        dist = self.actor(states)
        prob = dist[range(dist.shape[0]), actions].clamp(min=1e-38, max=1.)
        const_dist = dist.detach().clone()
        const_prob = prob.detach().clone()

        parameters = list(self.actor.parameters())
        advantages = (returns - baselines).detach().flatten()
        # advantages = advantages / (advantages.std() + 1e-8)

        L = ((prob / const_prob) * advantages).mean()
        dL = torch.autograd.grad(L, parameters, retain_graph=True)
        loss_grad = torch.cat([grad.flatten() for grad in dL])

        def Fvp(v):
            kl = self.get_kl(const_dist, dist).mean()
            grads = torch.autograd.grad(kl, parameters, create_graph=True, retain_graph=True)
            flat_grad_kl = torch.cat([grad.flatten() for grad in grads])
            v_v = v.detach().clone().to(device)
            kl_v = (flat_grad_kl * v_v).sum()
            grads = torch.autograd.grad(kl_v, parameters, retain_graph=True)
            flat_grad_grad_kl = torch.cat([grad.flatten() for grad in grads]).data
            return flat_grad_grad_kl + v * damping

        stepdir = self.conjugate_gradient(Fvp, loss_grad, 10)
        shs = stepdir @ Fvp(stepdir)
        max_length = torch.sqrt(2 * delta / shs) if shs != 0.0 else 0
        max_step = (max_length * stepdir).to('cpu')

        free_mem = L.flatten().sum()
        torch.autograd.grad(free_mem, parameters, retain_graph=False)

        def criterion(step):
            self.update_actor(step)
            with torch.no_grad():
                dist_new = self.actor(states)
                prob_new = dist_new[range(dist_new.shape[0]), actions]
                L_new = ((prob_new / const_prob) * advantages).mean()
                KL_new = self.get_kl(const_dist, dist_new).mean()
                if L_new - L > 0 and KL_new <= delta:
                    return True
            self.update_actor(-step)
            return False

        i = 0
        while not criterion((backtrack_ratio ** i) * max_step) and i < max_backtracks:
            i += 1

    def update_actor(self, grad_flattened):
        n = 0
        for params in self.actor.parameters():
            num_element = params.numel()
            g = grad_flattened[n:n + num_element].view(params.shape).to(device)
            params.data += g
            n += num_element

    def conjugate_gradient(self, Avp, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size()).to(device)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            if rdotr < residual_tol:
                break
            _Avp = Avp(p)
            php = torch.dot(p, _Avp)
            if php == 0.0:
                break
            alpha = rdotr / php
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def get_kl(self, p, q):
        p_log = p.clamp(min=1e-38, max=1.).log()
        q_log = q.clamp(min=1e-38, max=1.).log()
        return (p * (p_log - q_log)).sum(-1)

    def save_model(self):
        torch.save(self.actor.state_dict(), self.actor_path)
        torch.save(self.critic.state_dict(), self.critic_path)

    def train(self, num_epoch=500, update_step=10000, show_freq=None):
        i_episode = 0
        best_reward = None
        for i in range(num_epoch):
            if show_freq is not None and i % show_freq == 0:
                self.eval(num_episode=1)
            self.actor.train()
            self.critic.train()
            print('Epoch {}/{}'.format(i + 1, num_epoch))
            start_time = float(datetime.now().timestamp())
            epoch_rewards = []

            epoch_t = 0
            while True:  # episodes loop
                prev_states = list()
                state = self.env.reset()
                episode_reward = 0
                sample = []

                preprocess_state = preprocess_image(state)
                prev_states.append(preprocess_state)
                prev_states.append(preprocess_state)
                prev_states.append(preprocess_state)
                prev_states.append(preprocess_state)

                t = 0
                while True:
                    input_stack = np.array(prev_states[-4:])
                    action = self.select_action(input_stack)
                    next_state, reward, done, _ = self.env.step(action)
                    sample.append((input_stack, action, reward))
                    prev_states.append(preprocess_image(next_state))
                    episode_reward += reward
                    epoch_t += 1
                    t += 1
                    if done:
                        break

                states, actions, rewards = zip(*sample)
                states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
                actions = torch.as_tensor(actions).unsqueeze(1)
                rewards = np.array(rewards)

                self.memory.append(Transition(states, actions, rewards))
                epoch_rewards.append(episode_reward)
                sample.clear()

                i_episode += 1
                complete_ratio = min(epoch_t, update_step) * 19 // update_step
                str1, str2 = '=' * complete_ratio, '-' * (19 - complete_ratio)
                print('\r{}/{} [{}>{}] '.format(min(epoch_t, update_step), update_step, str1, str2), end='')
                if epoch_t >= update_step:
                    break

            epoch_avg_reward = sum(epoch_rewards) / len(epoch_rewards)
            if i/num_epoch > 0.9 and (best_reward is None or epoch_avg_reward > best_reward):
                best_reward = epoch_avg_reward
                self.save_model()

            self.update_agent(update_step=update_step)
            end_time = float(datetime.now().timestamp())
            running_time = end_time - start_time
            print('\r{}/{} [====================] '.format(update_step, update_step), end='')
            print('- {:.2f}s {:.2f}ms/step '.format(running_time, running_time * 1000 / epoch_t, 2), end='')
            print('- num_episode: {} - avg_reward: {:.2f}'.format(len(epoch_rewards), epoch_avg_reward))
            print('Peak cuda memory used: {:.2f}MB'.format(int(torch.cuda.max_memory_allocated()) / 1048576), end='\n\n')
            tags = ['{}/epoch-average-reward'.format(self.env.unwrapped.spec.id)]
            for tag, value in zip(tags, [epoch_avg_reward]):
                writer.add_scalar(tag, value, i+1)

            torch.cuda.reset_max_memory_allocated(device=device)
            epoch_rewards.clear()
            self.memory.clear()

    def eval(self, num_episode):
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            for i in range(num_episode):
                prev_states = list()
                state = self.env_eval.reset()
                preprocess_state = preprocess_image(state)
                prev_states.append(preprocess_state)
                prev_states.append(preprocess_state)
                prev_states.append(preprocess_state)
                prev_states.append(preprocess_state)

                t = 0
                while True:
                    input_stack = np.array(prev_states[-4:])
                    action = self.select_action(input_stack)
                    next_state, reward, done, _ = self.env_eval.step(action)
                    prev_states.append(preprocess_image(next_state))
                    t += 1
                    if done or t > 10000:
                        break


if __name__ == '__main__':
    random_seed = 48763
    env_name = 'ALE/Freeway-v5'
    train_env = gym.make(env_name)
    test_env = gym.make(env_name)

    train_env.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    agent = TRPO(train_env, test_env, gamma=0.995, lr_c=3e-4)
    agent.train(num_epoch=1500, update_step=10000, show_freq=None)
