import cloudpickle
import tensorflow as tf
import argparse
import gym_remote.exceptions as gre
import sonic_util


def load_model(model_path, weights_path=None):
    with open(model_path, 'rb') as f:
        make_model = cloudpickle.load(f)

    model = make_model()

    if weights_path is not None:
        model.load(weights_path)

    return model


def main():

    def _parse_args():
        parser = argparse.ArgumentParser(description="Run commands")
        parser.add_argument(
            '--model_path', type=str, default='make_model.pkl',
            help="Filename with pickled model_make function.")
        parser.add_argument(
            '--weights_path', type=str, default='weights.pkl',
            help="Filename with weights")
        return parser.parse_args()

    args = _parse_args()
    with tf.Session():
        model = load_model(args.model_path, args.weights_path)

        env = sonic_util.make_remote_env(socket_dir='tmp/sock')
        ob = env.reset()

        while True:
            # TODO: in ppo we used 10 environments, remove it later
            ob = [ob] * 10
            action = model.step(ob, None, None)[0][0]

            ob, reward, done, _ = env.step(action)
            if done:
                print('episode complete')
                env.reset()


if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as e:
        print('exception', e)