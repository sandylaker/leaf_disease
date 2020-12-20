from leaf_disease.datasets import kfold
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Stratified KFold')
    parser.add_argument('csv', help='file path of train csv')
    parser.add_argument('--save-dir', type=str, required=True, help='directory to save the txt files')
    parser.add_argument('--n-splits', type=int, default=5, help='number of splits')
    parser.add_argument('--shuffle', help='whether to shuffle', action='store_false')
    parser.add_argument('--random-state', type=int, help='random seed')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    kfold(args.csv,
          args.save_dir,
          n_splits=args.n_splits,
          shuffle=args.shuffle,
          random_state=args.random_state)

if __name__ == '__main__':
    main()