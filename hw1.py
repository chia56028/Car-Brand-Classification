import argparse
from train import Trainer

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', default='./cs-t0828-2020-hw1/', type=str, dest='working_dir')
    parser.add_argument('--train', default='train/', type=str, dest='training_dir')
    parser.add_argument('--test', default='testing_data/', type=str, dest='testing_dir')
    parser.add_argument('-l','--label', default='training_labels.csv', type=str, dest='label_path')
    parser.add_argument('-n','--model_name', default='resnetCars.pt', type=str, dest='model_name')
    parser.add_argument('-d', '--device', default=None, type=str)

    args = parser.parse_args()

    trainer = Trainer(args.working_dir, args.training_dir, args.testing_dir, args.label_path, args.model_name)
    # trainer.train()
    trainer.predict()