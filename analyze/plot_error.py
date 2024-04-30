import os
import sys
import time
import copy
import torch
import numpy as np
import collections
from tqdm import tqdm
import sys
sys.path.append("../")
from utils.parameters import parse_args
from utils.logger import Logger
from utils.dataset import Dataset, ImgForce, VisionForceDataset
from model.vision_force_estimator import create_estimator


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_step(estimator, dataset, bs, logger, num_train):
    results = (0, 0, 0)
    for _ in range(num_train):
        batch = dataset.sample(bs)
        result = estimator.train_batch(batch)
        results = tuple(map(lambda x, y: x + y, results, result))
    results = tuple(map(lambda x: x / num_train, results))
    logger.trainingBookkeeping(*results)


def valid_step(estimator, dataset, bs, logger, num_valid, sample_continuous, sample_in_order):
    results = (0, 0, 0)
    if sample_in_order:
        current_idx = 0
        while current_idx <= len(dataset) - dataset._n_history * dataset._history_interval:
            batch, current_idx = dataset.sample_by_index(bs, current_idx)
            result = estimator.test(batch, sample_continuous)
            # results = tuple(map(lambda x, y: x + y, results, result))
    else:
        for batch_idx in range(num_valid):
            batch = dataset.sample_continuous(bs) if sample_continuous else dataset.sample(bs)
            result = estimator.test(batch, sample_continuous)
            # results = tuple(map(lambda x, y: x + y, results, result))
    # results = tuple(map(lambda x: x / num_valid, results))
    # logger.validatingBookkeeping(*results)



def saveModelAndInfo(logger, agent, epoch):
    logger.saveModel(epoch, agent)
    logger.saveLossCurve()


def display_info(logger, epoch, t, pbar):
    description = 'Epoch:{} t:{:.02f}s | Train loss:{:.03f} |' \
                  ' Valid l2_f:{:.03f} - rela_err {:.02f}%, avg_err {:.03f}' \
        .format(epoch, time.time() - t, logger.l2[-1],
                logger.data['l2_f'][-1], logger.data['rela_err'][-1], logger.data['avg_err'][-1])
    pbar.set_description(description)
    pbar.update(1)



def collect_errors_step(estimator, dataset, bs, num_valid, flip_forces):
    GTs = None
    predictions = None
    indeces = []
    current_idx = 0
    while current_idx <= len(dataset) - dataset._n_history * dataset._history_interval:
        batch, current_idx, indeces_batch = dataset.sample_by_index_verbose(bs, current_idx)
        result = estimator.test(batch, False)
        GT, pred = estimator.test_verbose(batch)
        if flip_forces:
            pred = -pred
        if GTs is None:
            GTs = GT
            predictions = pred
            indeces += indeces_batch
        else:
            GTs = torch.cat((GTs, GT))
            predictions = torch.cat((predictions, pred))
            indeces += indeces_batch
        # ic(len(GTs))
    return GTs.detach().numpy(), predictions.detach().numpy(), np.array(indeces)


def collect_errors(flip_forces = False):
    args, hyper_parameters = parse_args()
    # logger = Logger(args.note, args.max_epoch)
    if args.seed is not None:
        set_seed(args.seed)

    # setup estimator
    estimator = create_estimator(args)
    estimator.train()
    if args.load_model:
        estimator.loadModel(args.load_model)

    if args.load_from_disk:
        if args.test_data_dir is not None:
            test_data_dir = args.test_data_dir
            fn = args.test_csv_fn
        else:
            test_data_dir = args.valid_data_dir
            fn = args.valid_csv_fn
        test_set = VisionForceDataset(test_data_dir, fn, \
                    n_history=args.n_history, history_interval=args.history_interval )
    else:
        test_set = Dataset()
        test_data_dir = args.test_data_dir if args.test_data_dir is not None else args.valid_data_dir
        test_set.load(test_data_dir, n_history=args.n_history, history_interval=args.history_interval)

    num_test = len(test_set) // args.bs

    #valid_step(estimator, test_set, args.bs, logger, num_test, True, False) #for plotting
    #logger.saveLossCurve()
    # print('Test: best epoch:{} l2_f:{:.03f} - rela_err {:.02f}%, avg_err {:.03f}'
    #       .format(0, logger.data['l2_f'][-1], logger.data['rela_err'][-1], logger.data['avg_err'][-1]))
    return collect_errors_step(estimator, test_set, args.bs, num_test, flip_forces = flip_forces)

# def smooth_forces(history = 2):


def plot_quantitative_results(f, f_hat, indeces, save_path = None):
    # valid f vs f_hat
    import matplotlib.pyplot as plt
    print(indeces.shape, f.shape, f_hat.shape)
    starting_index = 0
    shift_index = 2
    print(starting_index)
    plt.figure(figsize=(15, 3), dpi=300)
    # print(f.shape, f_hat.shape)

    plt.plot(indeces[starting_index:-shift_index],f[starting_index:-shift_index, 0], 'r', label=r'$f_x$')
    plt.plot(indeces[starting_index:-shift_index],f[starting_index:-shift_index, 1], 'b', label=r'$f_z$')
    plt.plot(indeces[starting_index:-shift_index],f_hat[starting_index+shift_index:, 0], 'r:', label=r'$\hat{f}_x$', alpha=0.6)
    plt.plot(indeces[starting_index:-shift_index],f_hat[starting_index+shift_index:, 1], 'b:', label=r'$\hat{f}_z$', alpha=0.6)

    # plt.ylim(-2, 2)
    plt.xlabel('image frame')
    plt.ylabel('force magnitude (N)')
    plt.legend()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

if __name__ == '__main__':
    args, hyper_parameters = parse_args()
    import matplotlib.pyplot as plt
    from icecream import ic
    GTs, predictions, indeces = collect_errors(flip_forces = False)
    print(GTs.shape, predictions.shape)

    ## Get GT stats
    all_force_norms = np.linalg.norm(GTs, axis = 1)
    all_force_norms = np.sort(all_force_norms)
    threshold_10_percent = all_force_norms[int(len(all_force_norms)/10)]
    ic(threshold_10_percent)
    average_force_norm = np.mean(all_force_norms)
    ## recalculate some performance stats
    # Relative error 
    total_relative_error = 0.
    for i in range(len(GTs)):
        if np.linalg.norm(GTs[i]) > threshold_10_percent:
            total_relative_error += np.linalg.norm(GTs[i] - predictions[i])/np.linalg.norm(GTs[i])
    ic(total_relative_error/len(GTs))

    # Scaled MSE
    total_scaled_MAE = 0.
    for i in range(len(GTs)):
        total_scaled_MAE += np.linalg.norm(GTs[i] - predictions[i])
    ic(total_scaled_MAE/len(GTs)/average_force_norm)
    ic(average_force_norm)


    plot_quantitative_results(GTs, predictions, indeces, './test_results.png')
    
    inds = [0, 100]
    for idx in inds:
        ic(GTs[idx])
        ic(predictions[idx])


    # plt.title('Fy GT vs prediction')
    # plt.xlabel('GT (N)')
    # plt.ylabel('Prediction (N)')
    # plt.scatter(GTs[:,0], predictions[:,0])
    # plt.show()    

    # plt.title('Fz GT vs prediction')
    # plt.xlabel('GT (N)')
    # plt.ylabel('Prediction (N)')
    # plt.scatter(GTs[:,1], predictions[:,1])
    # plt.show()    

    #plt.plot()