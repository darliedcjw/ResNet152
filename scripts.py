import torch

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.datasets.mnist import MNIST

import argparse

from train import Train


def main(
    train_path,
    val_path,
    log_path,
    num_classes,
    epochs,
    batch_size,
    learning_rate,
    lr_scheduler,
    momentum,
    optimizer,
    num_workers,
    device,
    use_tensorboard,
    checkpoint,
    mnist,
    ):

    transform = T.Compose([
        T.ToTensor(),
        T.Resize((28, 28), antialias=True),
        ])

    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    
    # ds_train = ImageFolder(root=train_path, transform=transform)
    # ds_val = ImageFolder(root=val_path, transform=transform)
    if mnist:
        ds_train = MNIST(root='datasets', train=True, transform=transform)
        ds_val = MNIST(root='datasets', train=False, transform=transform)
    else:
        ds_train = ImageFolder(root=train_path, transform=transform)
        ds_val = ImageFolder(root=val_path, transform=transform)

    with open('class_idx.txt', 'w') as f:
        for idx in ds_train.class_to_idx:
            f.writelines('{}: {}\n'.format(idx, ds_train.class_to_idx[idx]))

    train = Train(
                ds_train, 
                ds_val,
                log_path,
                num_classes,
                epochs,
                batch_size,
                learning_rate,
                lr_scheduler,
                momentum,
                optimizer,
                num_workers,
                device,
                use_tensorboard,
                checkpoint
                )
    
    train.run()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', '-tp', help='Train folder', type=str, default=None)
    parser.add_argument('--val_path', '-vp', help='Val folder', type=str, default=None)
    parser.add_argument('--log_path', '-lp', help='Log folder', type=str, default='logs')
    parser.add_argument('--num_classes', '-c', help='Number of classes', type=int, default=10)
    parser.add_argument('--epochs', '-e', help='Number of epochs', type=int, default=20) 
    parser.add_argument('--batch_size', '-b', help='Training batch size', type=int, default=128)
    parser.add_argument('--learning_rate', '-lr', help='Specify learning rate', type=float, default=0.0001)
    parser.add_argument('--lr_scheduler', '-lrs', help='Specify schedule', type=int, nargs='*', action='store', default=None)
    parser.add_argument('--momentum', '-m', help='Momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', '-o', help='Specify optimizer: SGD, Adam, RMSprop', type=str, default='SGD')
    parser.add_argument('--num_workers', '-w', help='Number of workers', type=int, default=8)
    parser.add_argument('--device', '-d', help='Device', type=str, default=None)
    parser.add_argument('--use_tensorboard', '-tb', help='Use tensorboard', type=bool, default=True)
    parser.add_argument('--checkpoint', '-cp', help='continue from checkpoint', type=str, default=None)
    parser.add_argument('--mnist', help="Use MNIST Dataset", default=True)
    args = parser.parse_args()

    main(**args.__dict__)