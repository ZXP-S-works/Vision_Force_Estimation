import os
import sys
import time
import copy
import torch
import numpy as np
import collections
from tqdm import tqdm
from utils.parameters import parse_args
from utils.logger import Logger
from utils.dataset import Dataset, ImgForce, VisionForceDataset
from model.vision_force_estimator import create_estimator
import cProfile
def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_step(estimator, dataset, bs, logger, num_train, gradient_accumulation, speed_augmentation):
    results = (0, 0, 0)
    if gradient_accumulation:
        ## sample in order
        current_idx = 0
        while current_idx <= len(dataset) - dataset._n_history * dataset._history_interval:
            batch, current_idx = dataset.sample_by_index(bs, current_idx)
            result = estimator.train_batch(batch, not gradient_accumulation, num_train)
            results = tuple(map(lambda x, y: x + y, results, result))
    else:
        for i in range(num_train):
            batch = dataset.sample(bs, speed_augmentation)
            if batch is not None:
                result = estimator.train_batch(batch, not gradient_accumulation)
                results = tuple(map(lambda x, y: x + y, results, result))
    if gradient_accumulation:
        for param in estimator.network.parameters():
            param.grad.data.clamp_(-1, 1)
        estimator.optimizer.step()
        estimator.optimizer.zero_grad()
    results = tuple(map(lambda x: x / num_train, results))
    logger.trainingBookkeeping(*results)


def valid_step(estimator, dataset, bs, logger, num_valid, sample_continuous, sample_in_order):
    results = (0, 0, 0)
    if sample_in_order:
        current_idx = 0
        while current_idx <= len(dataset) - dataset._n_history * dataset._history_interval:
            batch, current_idx = dataset.sample_by_index(bs, current_idx)
            result = estimator.test(batch, False)
            results = tuple(map(lambda x, y: x + y, results, result))
    else:
        for batch_idx in range(num_valid):
            batch = dataset.sample_continuous(bs) if sample_continuous else dataset.sample(bs)
            result = estimator.test(batch, sample_continuous)
            results = tuple(map(lambda x, y: x + y, results, result))
    results = tuple(map(lambda x: x / num_valid, results))
    logger.validatingBookkeeping(*results)


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


def train():
    args, hyper_parameters = parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    # setup estimator
    estimator = create_estimator(args)
    estimator.train()
    if args.load_model:
        estimator.loadModel(args.load_model)

    # setup dataset
    if args.load_from_disk:
        train_set = VisionForceDataset(args.train_data_dir, args.train_csv_fn,\
                n_history=args.n_history, history_interval=args.history_interval)
        valid_set = VisionForceDataset(args.valid_data_dir, args.valid_csv_fn,\
                n_history=args.n_history, history_interval=args.history_interval)
        if args.test_data_dir is not None:
            test_data_dir = args.test_data_dir
            fn = args.test_csv_fn
        else:
            test_data_dir = args.valid_data_dir
            fn = args.valid_csv_fn
        test_set = VisionForceDataset(test_data_dir, fn,\
                n_history=args.n_history, history_interval=args.history_interval)
    else:
        train_set = Dataset()
        train_set.load(args.train_data_dir, max_size=args.train_set_size,
                    n_history=args.n_history, history_interval=args.history_interval)
        valid_set = Dataset()
        valid_set.load(args.valid_data_dir, n_history=args.n_history, history_interval=args.history_interval)
        test_set = Dataset()
        # if test_data_dir is not specified, use args.valid_data_dir
        test_data_dir = args.test_data_dir if args.test_data_dir is not None else args.valid_data_dir
        test_set.load(test_data_dir, n_history=args.n_history, history_interval=args.history_interval)

    num_train = len(train_set) // args.bs
    num_valid = len(valid_set) // args.bs
    num_test = len(test_set) // args.bs

    # setup logger
    logger = Logger(args.note, args.max_epoch)
    hyper_parameters['model_shape'] = estimator.getModelStr()
    logger.saveParameters(hyper_parameters)

    pbar = tqdm(total=int(args.max_epoch))
    pbar.set_description('Epoch:0 t: 0s | Train l1:0, l2:0 | Valid x_error:0 f_error:0')

    # early stop
    patience =  args.patience
    earlystop_counter = 0
    lowest_val_loss = float('inf')

    # train loop
    estimator.optimizer.zero_grad()
    if args.load_model:
        starting_epoch = args.starting_epoch
        lowest_val_loss = args.prev_min_val
    else:
        starting_epoch = 0
    for epoch in range(starting_epoch, args.max_epoch):

        t_start = time.time()
        if args.train:
            train_step(estimator, train_set, args.bs, logger, num_train, args.gradient_accumulation, args.speed_augmentation)

        if args.valid:
            valid_step(estimator, valid_set, args.bs, logger, num_valid, args.valid_continuous_sampling, args.valid_sample_in_order)

        latest_val_loss = logger.df[-1]
        if latest_val_loss < lowest_val_loss:
            lowest_val_loss = latest_val_loss
            logger.saveModelBestVal(estimator)
            earlystop_counter = 0
        else:
            earlystop_counter += 1
        print(f'Early stopping counter is {earlystop_counter}')

        saveModelAndInfo(logger, estimator, epoch)
        display_info(logger, epoch, t_start, pbar)
        
        if earlystop_counter >= patience:
            print(f'Val loss has not imporved in {patience} epochs. Early stopping...')
            break

    best_epoch = np.argmin(np.asarray(logger.data['l2_f']))
    # best_model_path = os.path.join(logger.checkpoint_dir, str(best_epoch)+'.pt')
    best_model_path = os.path.join(logger.checkpoint_dir, 'best_val.pt')
    estimator.loadModel(best_model_path)
    valid_step(estimator, test_set, args.bs, logger, num_test, args.valid_continuous_sampling, args.valid_sample_in_order)
    saveModelAndInfo(logger, estimator, epoch+1)
    print('Test: best epoch:{} l2_f:{:.03f} - rela_err {:.02f}%, avg_err {:.03f}'
          .format(best_epoch, logger.data['l2_f'][-1], logger.data['rela_err'][-1], logger.data['avg_err'][-1]))


if __name__ == '__main__':
    train()
    # cProfile.run('train()','./train_stats2')
