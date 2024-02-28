from utils import write_to_json, read_from_json

def run_experiment(experiment_name):

    # get timestamp / date

    # create folder in experiments/

    # play_all_games()

    # write metrics to experiments/experiment_name.json
    # write playback to experiments/experiment_name_playback.json

def play_all_games():
    
    # for all filenames in game dir

    # gamestate, playback = play_game(filename)

    # return metrics, history


def play_game(filename, max_steps=500):

    # playback = []

    # setup FrotzEnv

    # while game not finished (env done or not dead)

        # get observation
    
        # prompt agent
    
        # take action
    
        # history.append((observation, action))
    
        # update gamestate
    
    # return gamestate, playback


# Metrics

def get_score (gamestate):
    # return score

def get_lives (gamestate):     
    # return lives
    
def get_steps (gamestate):
    # return steps