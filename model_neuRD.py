import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import os

class MyCNN(nn.Module):
    def __init__(self, input_channels=1, output_size=7):
        super(MyCNN, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(64 * 6 * 7, 128)  # The input size depends on your input dimensions
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        # Forward pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the output before passing through fully connected layers
        x = x.view(x.size()[0], -1)

        # Forward pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Apply softmax to the output
        # x = F.softmax(x, dim=-1)

        # Apply sigmoid to the output
        # x = F.sigmoid(x)
        return x

    def save(self, file_name='model.pth'):
    	model_folder_path = './models_experiment24'
    	if not os.path.exists(model_folder_path):
    		os.makedirs(model_folder_path)

    	file_name = os.path.join(model_folder_path, file_name)
    	torch.save(self.state_dict(), file_name)


class neuRDTrainer:
	def __init__(self, actor, critic, Q_fixed, lr, gamma):
		self.lr = lr
		self.gamma = gamma
		self.actor = actor
		self.critic = critic
		self.Q_fixed = Q_fixed
		self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.lr)
		self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.lr)
		# self.model_optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)
		self.criterion = nn.MSELoss()

	def train_actor(self, state, action, reward, next_state, done):
		state = torch.tensor(np.array(state), dtype=torch.float)
		action = torch.tensor(action, dtype=torch.long)

		current_logits = self.actor(state) 		#the current logits, y(s)

		with torch.no_grad():
			current_policy = F.softmax(current_logits, dim=-1) 	#the current policy, pi(s')

			### Expierimental
			current_values = self.critic(state) 	#the current action values, Q(s)			
			# current_values = self.Q_fixed(state)		
			###

			actor_targets = current_logits.clone()
			for idx in range(len(done)):
				advantage = current_values[idx][action[idx].item()] - torch.dot(current_policy[idx], current_values[idx])
				actor_targets[idx][action[idx].item()] = actor_targets[idx][action[idx].item()] + advantage
				# advantage = current_values[idx] - torch.dot(current_policy[idx], current_values[idx])
				# actor_targets[idx] = actor_targets[idx] + advantage

		actor_loss = self.criterion(actor_targets, current_logits)	
		self.actor_optimizer.zero_grad()			
		actor_loss.backward()		
		self.actor_optimizer.step()

	def train_critic(self, state, action, reward, next_state, done):
		state = torch.tensor(np.array(state), dtype=torch.float)
		next_state = torch.tensor(np.array(next_state), dtype=torch.float)
		action = torch.tensor(action, dtype=torch.long)
		reward = torch.tensor(reward, dtype=torch.float)

		current_values = self.critic(state) 	#the current action values, Q(s)

		with torch.no_grad():
			next_logits = self.actor(next_state)
			next_policy = F.softmax(next_logits, dim=-1)

			### Experimental
			# next_values = self.critic(next_state)
			next_values = self.Q_fixed(next_state)
			###

			critic_targets = current_values.clone()
			for idx in range(len(done)):
				Q_target = reward[idx]
				if not done[idx]:
					Q_target = reward[idx] + self.gamma * torch.dot(next_policy[idx], next_values[idx])
					# Q_target = reward[idx] + self.gamma * torch.max(next_values[idx])
				critic_targets[idx][action[idx].item()] = Q_target

		critic_loss = self.criterion(critic_targets, current_values)		
		self.critic_optimizer.zero_grad()	
		critic_loss.backward()
		self.critic_optimizer.step()