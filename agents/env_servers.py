import retro
import retro_contest
import gym
from gym_remote.server import RemoteEnvWrapper
from pathlib import Path
from multiprocessing import Process
import argparse
import pandas as pd


def start_game(game, state, directory='tmp', steps=1000000, discrete_actions=False, bk2dir=None):
    use_restricted_actions = retro.ACTIONS_FILTERED
    if discrete_actions:
        use_restricted_actions = retro.ACTIONS_DISCRETE

    try:
        env = retro.make(game, state, scenario='contest', use_restricted_actions=use_restricted_actions)
    except Exception:
        env = retro.make(game, state, use_restricted_actions=use_restricted_actions)

    if bk2dir:
        env.auto_record(bk2dir)

    env = retro_contest.StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=4500)

    env = RemoteEnvWrapper(env, directory)
    env.serve(steps)


def start_servers(game_states, socket_dir, steps):
    socket_dir = Path(socket_dir)

    workers = []
    game_dirs = []
    for game, state in game_states:
        game_state_dir = socket_dir / "{}_{}".format(game, state)
        game_state_dir.mkdir(parents=True, exist_ok=True)

        sock_fname = game_state_dir / 'sock'
        if sock_fname.exists():
            sock_fname.unlink()

        game_state_dir = str(game_state_dir)
        w = Process(target=start_game, args=(game, state, game_state_dir, steps))
        w.daemon = True
        w.start()
        workers.append(w)
        game_dirs.append(game_state_dir)

    return workers, game_dirs


def main():

    def _parse_args():
        parser = argparse.ArgumentParser(description="Run commands")
        parser.add_argument('--socket_dir', type=str, default='/tmp/retro', help="Base directory for sockers.")
        parser.add_argument('--csv_file', type=str, default='train_small.csv', help="Csv file with train games.")
        parser.add_argument('--steps', type=int, default=None, help="Number of timestemps for each environment.")
        return parser.parse_args()

    args = _parse_args()
    game_states = pd.read_csv(args.csv_file).values

    start_servers(game_states, args.socket_dir, args.steps)


if __name__ == '__main__':
    main()
