import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
np.set_printoptions(precision=4)

def play_game_mancala(player1, player2, game, training_mode, player1_regularizer, player2_regularizer, reward_eta):
	#input player1 and player2 are the Agent objects with the neural networks
	
	#initialize new game
	game.reset()
	player1_previous_state = None
	player2_previous_state = None
	turn = 0
	rt = np.zeros((43)) #Reward Transformation. this will be a list where on every turn the 
	# player who moves gets a reward (eta*log(pi/pi_reg)), and the other player will get the negative of that.
	# rewards from player 1 moving will be on odd indexes, rewards from player 2 will be on even indexes.
	# Therefore, the 0'th index has nothing. Connect 4 has a max of 42 moves, so the list is 43 long.

	while True:
		#Get move
		turn += 1
		if game.turn==1:
			state = player1.get_state(game)		
			move = player1.get_action(state, game)
			if training_mode==True:
				pi_a = torch.nn.functional.softmax(player1.actor(torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)), dim=-1)[:,move].item()
				pi_reg_a = torch.nn.functional.softmax(player1_regularizer.actor(torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)), dim=-1)[:,move].item()
				rt[turn] = reward_eta*np.log(pi_a/pi_reg_a) #regularized reward addition. This will be subtracted for the player that moved and added for the player that didn't move
			player_moved = 1
		elif game.turn==2:
			state = player2.get_state(game)	
			move = player2.get_action(state, game)
			if training_mode==True:
				pi_a = torch.nn.functional.softmax(player2.actor(torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)), dim=-1)[:,move].item()
				pi_reg_a = torch.nn.functional.softmax(player2_regularizer.actor(torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)), dim=-1)[:,move].item()
				rt[turn] = reward_eta*np.log(pi_a/pi_reg_a) #regularized reward addition. This will be subtracted for the player that moved and added for the player that didn't move
			player_moved = 2

		# We need to store a replay experience that is (state, action, reward, next_state, done)
		# however, state & next_state need to be both of the same player.
		# so we have to we have to temporarily save several "observations" (i.e. states) and rotate
		# what we save based on who made the moves.
		# reward is +1 for win, -1 for loss, 0 for draw.
		# done is a boolean just to indicate if it was a terminal state or not. For now, both players
		#   get a terminal state (each of their last moves). When the game is done, there is no "next_state"

		#play move
		game.play_step(move)
		if game.illegal_move_made == True:
			if training_mode==True:
				player1.remember(state, move, -1, state, True) #the second state won't actually be used.
				#the -20 is to punish illegal moves.
			if player_moved==1:
				result = 2
			elif player_moved==2:
				result = 1
			# result = 4
			break #should end the game here, when an illegal move was made.
			# raise Exception("Selected move in connectFour is illegal.")


		# remember(state, action, reward, next_state, done) 

		#we always have to remember the last pair of states
		if training_mode==True:
			reward_transformation = -rt[turn-2] + rt[turn-1]
			if player_moved==1 and turn>1:
				player1.remember(player1_previous_state, player1_previous_action, (0 + reward_transformation), state, False)
			elif player_moved==2 and turn>2:
				player1.remember(player2_previous_state, player2_previous_action, (0 + reward_transformation), state, False)

		#then update our temporary memory
		if player_moved==1:
			player1_previous_state = state
			player1_previous_action = move
		elif player_moved==2:
			player2_previous_state = state
			player2_previous_action = move

		#check if game has ended. Break if so.
		result = game.check_game_end()
		if not result==3: #game finished
			if training_mode==True:
				done=True #boolean
				if player_moved==1:
					p1_last_rt = -rt[turn]
					p2_last_rt = -rt[turn-1] + rt[turn]
				elif player_moved==2:
					p1_last_rt = -rt[turn-1] + rt[turn]
					p2_last_rt = -rt[turn]
				# only remember on one player since they are the same (for now)
				# The 4th input (state) is just a dummy placeholder. since the game ended, it won't be used.
				if result==0: #draw
					player1.remember(player1_previous_state, player1_previous_action, 0 + p1_last_rt, state, done)
					player1.remember(player2_previous_state, player2_previous_action, 0 + p2_last_rt, state, done)
				elif result==1: #player 1 won
					player1.remember(player1_previous_state, player1_previous_action, 1 + p1_last_rt, state, done)
					player1.remember(player2_previous_state, player2_previous_action, -1 + p2_last_rt, state, done)
				elif result==2: #player 2 won
					player1.remember(player1_previous_state, player1_previous_action, -1 + p1_last_rt, state, done)
					player1.remember(player2_previous_state, player2_previous_action, 1 + p2_last_rt, state, done)
			
			break
	return result








