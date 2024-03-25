import numpy as np
import scipy
import time

class ConnectFourGameAI:
	def __init__(self):
		self.state = np.zeros((6,7)) #starting state of the game. All empty
		#State will be 1's for player 1 and -1's for player 2.
		self.my_filter1 = np.array([[1, 1, 1, 1]])
		self.my_filter2 = np.array([[1], [1], [1], [1]])
		self.my_filter3 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
		self.my_filter4 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
		self.turn=1
		self.illegal_move_made=False

	def play_step(self, move):
	#move is 0-6 correlating with columns 1-7.

	#check if move is legal.
		if np.sum(self.state[:,move]==0) > 0:
			move_row = np.max(np.where(self.state[:,move]==0))
			if self.turn == 1:
				self.state[move_row, move] = 1
			elif self.turn ==2:
				self.state[move_row, move] = -1
		else:
			# print("State:")
			# print(self.state)
			# print("move = ", move)
			# print("turn = ", self.turn)
			self.illegal_move_made=True
			# raise Exception("Selected move in connectFour is illegal.")

		if self.turn==1:
			self.turn=2
		elif self.turn==2:
			self.turn=1

	def check_game_end(self):
		#Return Values: 0=draw, 1=p1 won, 2=p2 won, 3=game not over

		#4 convolutions = 4 directions you can get 4 in a row (horizontal, vert, 2xdiag)
		conv1 = scipy.signal.convolve2d(self.state, self.my_filter1, mode='full', boundary='fill', fillvalue=0)
		conv2 = scipy.signal.convolve2d(self.state, self.my_filter2, mode='full', boundary='fill', fillvalue=0)
		conv3 = scipy.signal.convolve2d(self.state, self.my_filter3, mode='full', boundary='fill', fillvalue=0)
		conv4 = scipy.signal.convolve2d(self.state, self.my_filter4, mode='full', boundary='fill', fillvalue=0)
		if (np.sum(conv1==4)>0 or np.sum(conv2==4)>0 or np.sum(conv3==4)>0 or np.sum(conv4==4)>0):
			return 1
		elif (np.sum(conv1==-4)>0 or np.sum(conv2==-4)>0 or np.sum(conv3==-4)>0 or np.sum(conv4==-4)>0):
			return 2
		elif np.sum(self.state==0)==0:
			#board is filled and no one has 4 in a row. Draw
			return 0
		else:
			#board is not filled an no one has 4 in a row yet.
			return 3

	def draw_board(self):
		for row in self.state:
			for val in row:
				if val == -1:
					print("O", end=" ")
				elif val == 1:
					print("X", end=" ")
				else:
					print("-", end=" ")
			print()  # Newline after each row

	def get_observation(self):
		#make it 1's for self and -1's for opponent pieces
		#could also experiment with other setups
		# if self.turn==1:
		# 	return self.state.reshape(1,6,7)
		# elif self.turn==2:
		# 	return (-1*self.state).reshape(1,6,7)
		##############
		#two channels: 1 for own pieces, 1 for opponent pieces
		if self.turn==1:
			first_channel = self.state==1
			second_channel = self.state==-1
		elif self.turn==2:
			first_channel = self.state==-1
			second_channel = self.state==1
		observation = np.zeros((2,6,7))
		observation[0,:,:]=first_channel
		observation[1,:,:]=second_channel
		return observation

	def get_legal_moves(self):
		return self.state[0,:]==0
		# return np.sum(self.state==0, axis=0) > 0


	def reset(self):
		self.state = np.zeros((6,7)) #starting state of the game. All empty
		self.turn=1
		self.illegal_move_made=False