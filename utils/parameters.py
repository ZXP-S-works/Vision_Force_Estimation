import numpy as np
import torch
import argparse

def strToBool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def parse_args():
    parser = argparse.ArgumentParser()

    # neural network
    parser.add_argument('--network', type=str, default='transformerresnet',
                        choices=['mlpresnet', 'transformerresnet'])
    parser.add_argument('--n_input_channel', type=int, default=3)
    parser.add_argument('--n_output_channel', type=int, default=2)
    parser.add_argument('--n_hidden', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--n_history', type=int, default=10)
    parser.add_argument('--n_servo_info', type=int, default=0)
    parser.add_argument('--history_interval', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--h', type=int, default=128)
    parser.add_argument('--w', type=int, default=128)
    parser.add_argument('--use_position_feature', type=bool, default=False)
    parser.add_argument('--use_position_embedding', type=bool, default=False)
    parser.add_argument('--position_embedding_dim', type=int, default=4)
    parser.add_argument('--resolution', type=int, default=128, help='imagined square img size for building the cnn')
    parser.add_argument('--starting_epoch', type=int, default=None, help='starting epoch when continuing training')
    parser.add_argument('--prev_min_val', type=float, default=None, help='previous min val when continuing training')


    # training/testing
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--l1_weight', type=float, default=1)
    parser.add_argument('--l2_weight', type=float, default=1)
    parser.add_argument('--l1_loss', type=str, default='CE', choices=['CE', 'BCE', 'MSE'])
    parser.add_argument('--l2_loss', type=str, default='MSE',
                        choices=['MSE', 'smooth_l1', 'MSE_contrastive'])
    parser.add_argument('--valid', type=strToBool, default=True)
    parser.add_argument('--valid_continuous_sampling', type=strToBool, default=False)
    parser.add_argument('--valid_sample_in_order', type=strToBool, default=True)
    parser.add_argument('--train', type=strToBool, default=True)
    parser.add_argument('--gradient_accumulation', type=strToBool, default=False)
    parser.add_argument('--grad_on_one_frame', type=strToBool, default=False)
    parser.add_argument('--loss_on_entire_history', type=strToBool, default=False)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--segmented_imgs', type=strToBool, default=True)
    parser.add_argument('--background_img_folder', type=str, \
        default='/vast/palmer/home.grace/yz2379/project/Data/MIT_Indoor/indoorCVPR_09/Images/office/')
    
    ## augmentation
    parser.add_argument('--segmentation_aug', type=strToBool, default=False)
    parser.add_argument('--speed_augmentation', type=strToBool, default=False)
        
    # dataset
    parser.add_argument('--train_data_dir', type=str, default=None)
    parser.add_argument('--valid_data_dir', type=str, default=None)
    parser.add_argument('--test_data_dir', type=str, default=None)
    parser.add_argument('--train_csv_fn', type=str, default='train_result.csv')
    parser.add_argument('--valid_csv_fn', type=str, default='val_result.csv')
    parser.add_argument('--test_csv_fn', type=str, default='test_result.csv')
    parser.add_argument('--train_set_size', type=float, default=None)
    parser.add_argument('--load_from_disk', type=strToBool, default=True)

    # logging
    parser.add_argument('--note', type=str, default=None)

    args = parser.parse_args()
    hyper_parameters = {}
    for key in sorted(vars(args)):
        hyper_parameters[key] = vars(args)[key]

    for key in hyper_parameters:
        print('{}: {}'.format(key, hyper_parameters[key]))
    return args, hyper_parameters
