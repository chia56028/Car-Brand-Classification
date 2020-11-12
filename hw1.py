import argparse
from train import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', default='./cs-t0828-2020-hw1/',
                        type=str, dest='working_dir', help='path to dataset')
    parser.add_argument('-tr', '--train_dir', default='train/',
                        type=str, dest='training_dir', help='path to training set')
    parser.add_argument('-te', '--test_dir', default='testing_data/',
                        type=str, dest='testing_dir', help='path to testing set')
    parser.add_argument('-l', '--label', default='training_labels.csv',
                        type=str, dest='label_path', help='path to label file')
    parser.add_argument('-n', '--model_name', default='resnetCars.pt',
                        type=str, dest='model_name', help='name the model')
    parser.add_argument('-t', '--train', default=True,
                        type=bool, dest='is_train', help='train')
    parser.add_argument('-p', '--predict', default=True,
                        type=bool, dest='is_predict', help='predict')
    parser.add_argument('-d', '--device', default=None, type=str)

    args = parser.parse_args()

    print(args.is_train)

    trainer = Trainer(args.working_dir, args.training_dir,
                      args.testing_dir, args.label_path, args.model_name)

    if args.is_train:
        trainer.train()
    if args.is_predict:
        trainer.predict()
