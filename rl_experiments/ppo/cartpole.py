import torch.utils.tensorboard as tensorboard
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import random
import gym
import sys


def to_t(x):
    return torch.tensor(x, dtype=torch.float)


class CriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Critic:
    def __init__(self, learning_rate, trace_decay_rate, logger):
        self.learning_rate = learning_rate
        self.trace_decay_rate = trace_decay_rate
        self.critic_net = CriticNet()
        self.logger = logger

    def episode_start(self):
        self.critic_net.zero_grad()

    def step(self, previous_state, state, reward, done):
        previous_state_value = self.critic_net(to_t(previous_state))
        state_value = 0 if done else self.critic_net(to_t(state))
        td_error = reward + state_value - previous_state_value

        for p in self.critic_net.parameters():
            if p.grad is not None:
                p.grad *= self.trace_decay_rate

        previous_state_value.backward()

        self.logger(td_error, self.critic_net)

        for p in self.critic_net.parameters():
            p.grad.data = torch.clamp(p.grad.data, -10, 10)
            p.data += self.learning_rate * td_error * p.grad.data

        return previous_state_value, state_value


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


class Policy:
    def __init__(self, env, logger):
        self.env = env
        self.policy_net = PolicyNet()
        self.logger = logger

    def sample_action(self, state):
        probabilities = self.policy_net(to_t(state))
        action = random.choices(
            [0, 1],
            weights=probabilities
        )[0]
        return action

    def probabilities(self, state, action):
        probabilities = self.policy_net(state)

        if state.dim() == 1:
            return probabilities[action]
        elif state.dim() == 2:
            return torch.gather(probabilities, 1, action.reshape(-1, 1).long()).flatten()
        else:
            raise Exception('bruh')

    def evaluate(self, episodes=150):
        total_reward = 0
        meet_goal = 0

        for episode in range(episodes):
            episode_reward = 0
            state = self.env.reset()
            done = False

            while not done:
                action = self.sample_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                episode_reward += reward

            if episode_reward >= 200:
                meet_goal += 1

        if meet_goal == episodes:
            print('solved!')
            torch.save(self.policy_net.state_dict(), 'solution_policy')
            sys.exit()

        self.logger(meet_goal / episodes)

        return total_reward / episodes


class Actor:
    def __init__(self, env, critic, gae_gamma, gae_lambda, trajectory_length):
        self.env = env
        self.critic = critic
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.trajectory_length = trajectory_length
        self.state = None
        self.done = True

    def step(self, policy):
        trajectory_states = []  # torch.zeros(self.trajectory_length, 4)
        trajectory_actions = []  # torch.zeros(self.trajectory_length)
        td_errors = []  # torch.zeros(self.trajectory_length)
        previous_state = self.state

        for t in range(self.trajectory_length):
            if self.done:
                self.state = self.env.reset()
                previous_state = self.state

            # trajectory_states[t] = to_t(self.state)
            trajectory_states.append(to_t(self.state))
            action = policy.sample_action(self.state)
            # trajectory_actions[t] = action
            trajectory_actions.append(action)

            self.state, reward, self.done, _ = self.env.step(action)
            previous_state_value, state_value = self.critic.step(previous_state, self.state, reward, self.done)
            # td_errors[t] = reward + self.gae_gamma * state_value - previous_state_value
            td_errors.append(reward + self.gae_gamma * state_value - previous_state_value)
            previous_state = self.state

            if torch.isnan(td_errors[t]).any():
                pass

        # advantages = torch.zeros(self.trajectory_length)
        advantages = torch.zeros(len(td_errors))
        advantages = torch.zeros(self.trajectory_length)
        previous_advantage = 0

        for t in range(self.trajectory_length - 1, -1, -1):
            advantage = self.gae_gamma * self.gae_lambda * previous_advantage + td_errors[t]
            advantages[t] = advantage
            previous_advantage = advantage

        trajectory_states = torch.stack(trajectory_states)
        trajectory_actions = to_t(trajectory_actions)
        return trajectory_states, trajectory_actions, advantages.detach()


class PPO:
    def __init__(self, actor, policy, learning_rate, ratio_clipping_amount, epochs, logger):
        self.actor = actor
        self.policy = policy
        self.policy_optim = optim.SGD(self.policy.policy_net.parameters(), lr=learning_rate)
        self.ratio_clipping_amount = ratio_clipping_amount
        self.epochs = epochs
        self.logger = logger

    def step(self):
        trajectory_states, trajectory_actions, advantages = self.actor.step(self.policy)
        old_probabilities = self.policy.probabilities(trajectory_states, trajectory_actions).detach()

        for epoch in range(self.epochs):
            probabilities = self.policy.probabilities(trajectory_states, trajectory_actions)
            ratio = probabilities / old_probabilities
            clipped_ratio = torch.clamp(ratio, 1 - self.ratio_clipping_amount, 1 + self.ratio_clipping_amount)
            objective = torch.mean(torch.min(clipped_ratio * advantages, ratio * advantages))
            self.logger(objective, ratio, clipped_ratio, advantages)
            loss = -objective
            self.policy_optim.zero_grad()
            loss.backward()
            self.policy_optim.step()


def main():
    env = gym.make('CartPole-v1')
    writer = tensorboard.SummaryWriter()

    critic_step = 0

    def critic_logger(td_error, critic_net):
        nonlocal critic_step
        writer.add_scalar('critic/td_error', td_error, critic_step)
        writer.add_scalar('critic/td_error_abs', abs(td_error), critic_step)
        grad_sizes = torch.tensor([
            torch.sqrt(torch.sum(p.grad.data.pow(2))) for p in critic_net.parameters()
        ])
        writer.add_scalar('critic/mean_grad_size', torch.mean(grad_sizes), critic_step)
        layers = [critic.critic_net.fc1, critic.critic_net.fc2, critic.critic_net.fc3]
        weight_sizes = torch.tensor([
            torch.sqrt(torch.sum(layer.weight.pow(2))) for layer in layers
        ])
        writer.add_scalar('critic/mean_weight_size', torch.mean(weight_sizes), critic_step)
        bias_sizes = torch.tensor([
            torch.sqrt(torch.sum(layer.bias.pow(2))) for layer in layers
        ])
        writer.add_scalar('critic/mean_bias_size', torch.mean(bias_sizes), critic_step)
        critic_step += 1

    ppo_step = 0

    def ppo_logger(objective, ratio, clipped_ratio, advantages):
        nonlocal ppo_step
        writer.add_scalar('ppo/objective', objective, ppo_step)
        writer.add_scalar('ppo/mean_ratio', torch.mean(ratio), ppo_step)
        writer.add_scalar('ppo/mean_clipped_ratio', torch.mean(clipped_ratio), ppo_step)
        writer.add_scalar('ppo/mean_advantage', torch.mean(advantages), ppo_step)
        ppo_step += 1

    critic_learning_rate = 5e-4
    trace_decay_rate = 0.7
    critic = Critic(critic_learning_rate, trace_decay_rate, critic_logger)

    policy_step = 0

    def policy_logger(fraction_meet_goal):
        nonlocal policy_step
        writer.add_scalar('policy/percent_meet_goal', 100 * fraction_meet_goal, policy_step)
        policy_step += 1

    gae_gamma = 0.98
    gae_lambda = 0.96
    trajectory_length = 50
    actor = Actor(env, critic, gae_gamma, gae_lambda, trajectory_length)
    policy = Policy(env, policy_logger)
    policy_learning_rate = 4e-4
    ratio_clipping_amount = 0.3
    epochs = 30
    ppo = PPO(actor, policy, policy_learning_rate, ratio_clipping_amount, epochs, ppo_logger)

    for i in range(10000):
        print(i)

        if i % 10 == 0:
            writer.add_scalar('reward', policy.evaluate(), i // 10)

        ppo.step()

    writer.close()


if __name__ == '__main__':
    main()
