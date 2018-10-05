import argparse


args_list = []
parser = argparse.ArgumentParser()


def add_arg_group(name):
    """
    :param name: argument group, str
    :return: list (argument)
    """
    arg = parser.add_argument_group(name)
    args_list.append(arg)
    return arg


def get_config():
    cfg, un_parsed = parser.parse_known_args()
    return cfg, un_parsed


# Network
network_arg = add_arg_group('Network')
network_arg.add_argument('--kernel_size', type=int, default=3)
network_arg.add_argument('--filter_size', type=int, default=64)
network_arg.add_argument('--image_scaling_factor', type=int, default=4)
network_arg.add_argument('--activation', type=str, default='relu')
network_arg.add_argument('--res_scale', type=float, default=1.)
network_arg.add_argument('--n_res_blocks', type=int, default=20)
network_arg.add_argument('--n_res_groups', type=int, default=10)
network_arg.add_argument('--reduction', type=int, default=16)
network_arg.add_argument('--use_bn', type=bool, default=False)
network_arg.add_argument('--l2_reg', type=float, default=5e-4)

# Train/Test hyper-parameters
train_arg = add_arg_group('Training')
train_arg.add_argument('--batch_size', type=int, default=16)
train_arg.add_argument('--epochs', type=int, default=1000)
train_arg.add_argument('--logging_step', type=int, default=100)
train_arg.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
train_arg.add_argument('--lr', type=float, default=1e-4)
train_arg.add_argument('--lr_decay', type=float, default=.5)
train_arg.add_argument('--lr_decay_step', type=int, default=2e5)
train_arg.add_argument('--momentum', type=float, default=.9)
train_arg.add_argument('--beta1', type=float, default=.9)
train_arg.add_argument('--beta2', type=float, default=.999)
train_arg.add_argument('--lr_lower_boundary', type=float, default=1e-5)
train_arg.add_argument('--opt_epsilon', type=float, default=1e-8)

# Train/Test hyper-parameters
data_arg = add_arg_group('Data')
data_arg.add_argument('--data_dir', type=str, default="D://DataSet//DIV2K//")
data_arg.add_argument('--output_dir', type=str, default='./output/')
data_arg.add_argument('--summary', type=str, default='./log/')
data_arg.add_argument('--patch_size', type=int, default=16)

# Misc
misc_arg = add_arg_group('Misc')
misc_arg.add_argument('--device', type=str, default='gpu')
misc_arg.add_argument('--n_gpu', type=int, default=1)
misc_arg.add_argument('--n_threads', type=int, default=8)
misc_arg.add_argument('--pre_trained', type=str, default='./model/')
misc_arg.add_argument('--seed', type=int, default=1)
