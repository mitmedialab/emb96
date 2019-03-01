from dataset.midi import build_transpose, build_images
from dataset.scrapper import scrap
from model.train import train

def build_dataset(dataset_dir):
    if dataset_dir is None:
        return
    scrap(dataset_dir)

def build_t_dataset(dataset_dir, t_dataset_dir):
    if dataset_dir is None or t_dataset_dir is None:
        return
    build_transpose(dataset_dir, t_dataset_dir)

def generate_dataset(dataset_dir, t_dataset_dir, generated_dir):
    if dataset_dir is None or t_dataset_dir is None or generated_dir is None:
        return
    build_images(dataset_dir, t_dataset_dir, generated_dir)

def train_model(epochs, batch_size, learning_rate, weight_decay, beta, num_workers,
                dataset_dir, experience_name, saving_rate):
    if epochs is None or batch_size is None or learning_rate is None or weight_decay is None or beta is None or num_workers is None or dataset_dir is None or experience_name is None or saving_rate is None:
        return

    train(epochs, batch_size, learning_rate, weight_decay, beta, num_workers,
          dataset_dir, experience_name, saving_rate)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir',   type=str)
    parser.add_argument('--t_dataset_dir', type=str)
    parser.add_argument('--generated_dir', type=str)

    parser.add_argument('--download',  dest='download',  action='store_true')
    parser.add_argument('--transpose', dest='transpose', action='store_true')
    parser.add_argument('--generate',  dest='generate',  action='store_true')
    parser.add_argument('--train',     dest='train',     action='store_true')

    parser.add_argument('--epochs',          default=250,          type=int)
    parser.add_argument('--batch_size',      default=16,           type=int)
    parser.add_argument('--learning_rate',   default=1e-4,         type=float)
    parser.add_argument('--weight_decay',    default=0.,           type=float)
    parser.add_argument('--beta',            default=4.,           type=float)
    parser.add_argument('--num_workers',     default=6,            type=int)
    parser.add_argument('--experience_name', default='experience', type=str)
    parser.add_argument('--saving_rate',     default=2,            type=int)

    args = parser.parse_args()

    if args.download:
        build_dataset(args.dataset_dir)

    if args.transpose:
        build_t_dataset(args.dataset_dir, args.t_dataset_dir)

    if args.generate:
        generate_dataset(args.dataset_dir, args.t_dataset_dir, args.generated_dir)

    if args.train:
        train_model(args.epochs, args.batch_size, args.learning_rate,
                    args.weight_decay, args.beta, args.num_workers,
                    args.experience_name, args.saving_rate)
