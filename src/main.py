from exploration.latent import generate
from dataset.scrapper import scrap
from dataset.midi import build
from model.train import train

def scrap_dataset(dataset_dir):
    if dataset_dir is None:
        return
    scrap(dataset_dir)

def build_dataset(dataset_dir, build_dir):
    if dataset_dir is None or build_dir is None:
        return
    build(dataset_dir, build_dir)

def train_model(epochs, batch_size, learning_rate, weight_decay,
                beta_1, beta_2, latent_size, momentum, num_workers, build_dir,
                experience_name, saving_rate, checkpoint):
    if epochs is None or batch_size is None or learning_rate is None or weight_decay is None:
        return
    if beta_1 is None or beta_2 is None or latent_size is None or momentum is None or num_workers is None or build_dir is None:
        return
    if experience_name is None or saving_rate is None:
        return

    train(epochs, batch_size, learning_rate, weight_decay,
          beta_1, beta_2, latent_size, momentum, num_workers, build_dir,
          experience_name, saving_rate, None if checkpoint is None else checkpoint)

def generate_examples(output_dir, latent_size, momentum, checkpoint, n, cpu):
    if output_dir is None or latent_size is None or momentum is None or checkpoint is None or n is None:
        return

    generate(output_dir, latent_size, momentum, checkpoint, n, False if cpu is None else True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--build_dir',   type=str)
    parser.add_argument('--checkpoint',  type=str)
    parser.add_argument('--output_dir',  type=str)

    parser.add_argument('--download', dest='download', action='store_true')
    parser.add_argument('--build',    dest='build',    action='store_true')
    parser.add_argument('--train',    dest='train',    action='store_true')
    parser.add_argument('--test',     dest='test',     action='store_true')
    parser.add_argument('--cpu',      dest='test',     action='store_true')

    parser.add_argument('--epochs',          default=2000,         type=int)
    parser.add_argument('--batch_size',      default=32,           type=int)
    parser.add_argument('--learning_rate',   default=1e-3,         type=float)
    parser.add_argument('--weight_decay',    default=0.,           type=float)
    parser.add_argument('--beta_1',          default=.02,          type=float)
    parser.add_argument('--beta_2',          default=.1,           type=float)
    parser.add_argument('--momentum',        default=.9,           type=float)
    parser.add_argument('--latent_size',     default=128,          type=int)

    parser.add_argument('--num_workers',     default=4,            type=int)
    parser.add_argument('--experience_name', default='experience', type=str)
    parser.add_argument('--saving_rate',     default=2,            type=int)
    parser.add_argument('--n_examples',      default=6,            type=int)


    args = parser.parse_args()

    if args.download:
        scrap_dataset(args.dataset_dir)

    if args.build:
        build_dataset(args.dataset_dir, args.build_dir)

    if args.train and not args.cpu:
        train_model(args.epochs, args.batch_size, args.learning_rate, args.weight_decay,
                    args.beta_1, args.beta_2, args.latent_size, args.momentum, args.num_workers, args.build_dir,
                    args.experience_name, args.saving_rate, args.checkpoint)

    if args.test:
        generate_examples(args.output_dir, args.latent_size, args.momentum, args.checkpoint, args.n_examples, args.cpu)
