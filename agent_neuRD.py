import torch
import random
import numpy as np
import time
import copy
from collections import deque
from game_ConnectFour import ConnectFourGameAI
from model_neuRD import MyCNN, neuRDTrainer
from mancala_functions import play_game_mancala

MAX_MEMORY = 1000
BATCH_SIZE = 1000
LR = 0.0005
PTR = 10 #policy_train_rate = how many time slower we train the policy versus the Q function
REWARD_ETA = 0.2

class Agent_neuRD:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 1 # discount rate
        self.policy_train_rate = PTR
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.actor = MyCNN(input_channels=2)
        self.critic = MyCNN(input_channels=2)
        self.Q_fixed = MyCNN(input_channels=2)
        self.trainer = neuRDTrainer(self.actor, self.critic, self.Q_fixed, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        state = game.get_observation() #this is the state, as observed by the agent
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_memory(self, gen):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        if (gen % self.policy_train_rate) == 0: #train the policy/actor. Otherwise, train the value/critic
            self.trainer.train_actor(states, actions, rewards, next_states, dones)
        else:
            self.trainer.train_critic(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, game):
        # legal_moves = torch.tensor(game.get_legal_moves()) #1=legal, 0=illegal
        state0 = torch.tensor(state, dtype=torch.float)
        #unsqueeze to give a batch size of 1
        state0 = torch.unsqueeze(state0, 0)
        prediction = torch.nn.functional.softmax(self.actor(state0), dim=-1)
        # filtered_prediction = prediction * legal_moves        
        try:
            choice = torch.multinomial(prediction, 1)
        except:
            game.illegal_move_made=True
            print("invalid multinomial distribution (sum of probabilities <= 0)")
        # categorical = torch.distributions.Categorical(filtered_prediction)
        # choice = categorical.sample()
        if game.illegal_move_made==True:
            print("legal moves = ", legal_moves)
            print("state0 = ", state0)
            print("prediction = ", prediction)
            print("filtered_prediction = ", filtered_prediction)
            # print("probabilities = ", categorical.probs)
            print("choice = ", choice)
        return choice.item()


def train():
    #this is the loop where we can play games and train.
    my_agent = Agent_neuRD()
    regularization_policy = Agent_neuRD()
    # my_agent.model.load_state_dict(torch.load("models_experiment3/mancala_model_18200.pth"))

    game = ConnectFourGameAI()
    num_generations = 125000
    num_rounds = 10
    model_number = 0
    num_illegal_moves = 0
    generations_to_replace_Q_fixed = 100
    generations_to_replace_reg_policy = 5000
    start_time = time.time()
    for gen in range(num_generations+1):
        print("generation: ", gen)
        for round in range(num_rounds):
            result = play_game_mancala(my_agent, my_agent, game, True, regularization_policy, regularization_policy, REWARD_ETA)
            if result == 4:
                num_illegal_moves += 1
        print("number of illegal moves this generation =", num_illegal_moves)
        num_illegal_moves = 0
        my_agent.train_memory(gen)
        if (gen % 500) == 0:
            print("\tSAVING MODEL")
            my_agent.actor.save("c4_model_"+str(model_number)+".pth")
            # my_agent.actor.save("c4_actor_"+str(model_number)+".pth")
            # my_agent.critic.save("c4_critic_"+str(model_number)+".pth")
            model_number+=1
        if (gen % generations_to_replace_Q_fixed) == 0:
            print("\tQ FIXED")
            my_agent.Q_fixed = copy.deepcopy(my_agent.critic)
        if (gen % generations_to_replace_reg_policy) == 0:
            regularization_policy = copy.deepcopy(my_agent)

    print(time.time()-start_time)

if __name__ == '__main__':
    train()