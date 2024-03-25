from game_ConnectFour import ConnectFourGameAI
from agent import Agent
from agent_neuRD import Agent_neuRD
import torch

game = ConnectFourGameAI()
human_player = 2 #player 1 or 2
agent = Agent_neuRD()
agent_file_path = "models_experiment24/c4_model_194.pth"
agent.actor.load_state_dict(torch.load(agent_file_path))

while True:
	#Get move
	game.draw_board()
	print("")

	if (game.turn==1 and human_player==1) or (game.turn==2 and human_player==2):
		move = int(input("What move do you play? (0-6)"))
	else:
		state = agent.get_state(game)	
		move = agent.get_action(state, game)

	#play move
	game.play_step(move)
	if game.illegal_move_made == True:
		print("ILLEGAL MOVE MADE")
		break

	#check if game has ended. Break if so.
	result = game.check_game_end()
	if not result==3: #game finished			
		break
game.draw_board()
print(result)