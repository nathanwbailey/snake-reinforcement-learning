import torch
import model
import CacheRecall
import numpy as np
import random
from itertools import count
import pygame as pg
import matplotlib.pyplot as plt
import torch.optim as optim
import math
import cv2

class Agent():


    def __init__(self, BATCH_SIZE, MEMORY_SIZE, GAMMA, example_image, output_dim, action_dim, action_dict, EPS_START, EPS_END, EPS_DECAY_VALUE, lr, TAU) -> None:
        self.BATCH_SIZE = BATCH_SIZE
        self.MEMORY_SIZE = MEMORY_SIZE
        self.GAMMA = GAMMA
        self.input_dim = self.preprocess_image(example_image).size()[0]*4
        self.output_dim = output_dim
        self.action_dim = action_dim
        self.action_dict = action_dict
        self.EPS_START=EPS_START
        self.EPS_END=EPS_END
        self.EPS_DECAY_VALUE=EPS_DECAY_VALUE
        self.eps=EPS_START
        self.TAU=TAU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.episode_durations = []
        self.episode_score = []
        self.cache_recall = CacheRecall.CacheRecall(memory_size=MEMORY_SIZE)
        self.policy_net = model.SnakeNetv2(input_dim=4, output_dim=self.output_dim).to(self.device)
        self.target_net = model.SnakeNetv2(input_dim=4, output_dim=self.output_dim).to(self.device)
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.steps_done = 0


    def preprocess_image(self, img):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(img,1)
        img = cv2.resize(img, (80, 80))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = torch.FloatTensor(img)
        return img


    def plot_durations(self):
        plt.figure(1)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        #Plot the durations
        plt.plot(durations_t.numpy())
        # Take 100 episode averages of the durations and plot them too, to show a running average on the graph
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.savefig('trainingv2.png')

    def plot_score(self):
        plt.figure(1)
        plt.clf()
        score_t = torch.tensor(self.episode_score, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(score_t.numpy())
        if len(score_t) >= 100:
            means = score_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.savefig('training_scorev2.png')

    @torch.no_grad()
    def take_action(self, state):
        self.eps = self.eps*self.EPS_DECAY_VALUE
        self.eps = max(self.eps, self.EPS_END)
        if self.eps < np.random.rand():
            state = state.unsqueeze(0)
            action_idx = torch.argmax(self.policy_net(state), dim=1).item()
        else:
            action_idx = random.randint(0, self.action_dim-1)
        self.steps_done+=1
        return action_idx


    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
            self.target_net.load_state_dict(target_net_state_dict)

    def optimize_model(self):
        if len(self.cache_recall) < self.BATCH_SIZE:
            return
        #Grab the batch
        batch = self.cache_recall.recall(self.BATCH_SIZE)
        batch = [*zip(*batch)]
        state = torch.stack(batch[0])
        next_state = torch.stack(batch[1])
        #Grab the next states that are not final states
        non_final_next_states = torch.stack([s for s in batch[1] if s is not None])
        action = torch.stack(batch[2])
        reward = torch.cat(batch[3])
        state_action_values = self.policy_net(state).gather(1, action)
        with torch.no_grad():
            next_state_action_values = self.target_net(next_state).max(1)[0]
        expected_state_action_values = (next_state_action_values * self.GAMMA) + reward
        loss_fn = torch.nn.SmoothL1Loss()
        loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def frameskip(self, env, action, num_skip):
        total_reward = 0
        for _ in range(num_skip):
            reward = env.act(self.action_dict[action])
            total_reward += reward
            done = env.game_over()
            if done:
                break
        return total_reward, done


    def train(self, episodes, env):
        self.steps_done = 0
        self.global_steps_done = 0
        for episode in range(episodes):
            env.reset_game()
            state = env.getScreenRGB()
            state = self.preprocess_image(state).to(self.device)
            state = torch.stack((state, state, state, state), dim=0)
            for c in count():
                action = self.take_action(state)
                reward = env.act(self.action_dict[action])
                reward = torch.tensor([reward], device=self.device)
                action = torch.tensor([action], device=self.device)
                next_state = env.getScreenRGB()
                next_state = self.preprocess_image(next_state).to(self.device)
                next_state = torch.cat((next_state.unsqueeze(0), state[:3, :, :]), dim=0).to(self.device)
                done = env.game_over()
                #if done:
                    #next_state = None
                if len(self.cache_recall) >= self.MEMORY_SIZE:
                    self.cache_recall.memory.popleft()
                self.cache_recall.cache((state, next_state, action, reward, done))
                state=next_state
                self.optimize_model()
                self.update_target_network()

                pg.display.update()
                if done:
                    #Update the number of durations for the episode
                    self.episode_durations.append(c+1)
                    self.episode_score.append(env.score())
                    #Plot them and save the networks
                    self.plot_durations()
                    self.plot_score()
                    print("EPS: {}".format(self.eps))
                    print("Durations: {}".format(c+1))
                    print("Score: {}".format(env.score()))
                    torch.save(self.target_net.state_dict(), 'target_netv2.pt')
                    torch.save(self.policy_net.state_dict(), 'policy_netv2.pt')
                    #Start a new episode
                    break