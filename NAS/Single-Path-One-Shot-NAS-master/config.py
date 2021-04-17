import argparse


def get_args():
    parser = argparse.ArgumentParser("Single_Path_One_Shot")
    parser.add_argument('--exp_name', type=str, default='spos_cifar10', required=True, help='experiment name')
    parser.add_argument('--data_dir', type=str, default='/home/pdluser/project/lushun', help='path to the dataset')
    parser.add_argument('--classes', type=int, default=10, help='dataset classes')
    parser.add_argument('--layers', type=int, default=20, help='batch size')
    parser.add_argument('--num_choices', type=int, default=4, help='number choices per layer')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--epochs', type=int, default=300, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--val_interval', type=int, default=5, help='validate and save frequency')
    parser.add_argument('--random_search', type=int, default=500, help='validate and save frequency')
    # ******************************* dataset *******************************#
    parser.add_argument('--dataset', type=str, default='cifar10', help='path to the dataset')
    parser.add_argument('--cutout', action='store_true', help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto augmentation')
    parser.add_argument('--resize', action='store_true', default=False, help='use resize')
    args = parser.parse_args()
    print(args)
    return args
