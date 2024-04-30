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
from plot_error import collect_errors_step
import csv

class obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, obj(v) if isinstance(v, dict) else v)

def collect_errors(args, flip_forces = False):
    # setup estimator
    estimator = create_estimator(args)
    estimator.train()
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


def plot_quantitative_results(f, f_hat, indeces, save_path = None):
    # valid f vs f_hat
    import matplotlib.pyplot as plt
    import matplotlib
    plt.style.use('seaborn-v0_8-darkgrid')
    matplotlib.rcParams.update({'font.size': 14})
    print('------')
    print(indeces.shape, f.shape, f_hat.shape, save_path)
    # starting_index = 0
    # shift_index = 1
    # print(starting_index)
    # plt.figure(figsize=(18, 3), dpi=300)
    plt.figure(figsize=(12, 3), dpi=300)
    # plt.figure()
    # print(f.shape, f_hat.shape)

    plt.plot(indeces/10,f[:, 0], 'r:', label='Ground truth Fx')
    plt.plot(indeces/10,f[:, 1], 'b:', label='Ground truth Fz')
    plt.plot(indeces/10,f_hat[:,0], 'r', label=r'Predicted Fx') #, alpha=0.6)
    plt.plot(indeces/10,f_hat[:, 1], 'b', label=r'Predicted Fz') #, alpha=0.6)

    # plt.plot(indeces[starting_index:-shift_index],f[starting_index:-shift_index, 0], 'r', label=r'$f_x$')
    # plt.plot(indeces[starting_index:-shift_index],f[starting_index:-shift_index, 1], 'b', label=r'$f_z$')
    # plt.plot(indeces[starting_index:-shift_index],f_hat[starting_index+shift_index:, 0], 'r:', label=r'$\hat{f}_x$', alpha=0.6)
    # plt.plot(indeces[starting_index:-shift_index],f_hat[starting_index+shift_index:, 1], 'b:', label=r'$\hat{f}_z$', alpha=0.6)

    # plt.ylim(-2, 2)
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    # plt.legend(loc = "upper right")
    plt.legend(loc=(1.0, 0.5))
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches="tight")

def get_stats(GTs, predictions):
    ## Get GT stats
    all_force_norms = np.linalg.norm(GTs, axis = 1)
    all_force_norms = np.sort(all_force_norms)
    threshold_10_percent = all_force_norms[int(len(all_force_norms)/10)]
    # ic(threshold_10_percent)
    average_force_norm = np.mean(all_force_norms)
    ## recalculate some performance stats
    # Relative error 
    total_relative_error = 0.
    for i in range(len(GTs)):
        if np.linalg.norm(GTs[i]) > threshold_10_percent:
            total_relative_error += np.linalg.norm(GTs[i] - predictions[i])/np.linalg.norm(GTs[i])
    # ic(total_relative_error/len(GTs))

    # Scaled MSE
    total_scaled_MAE = 0.
    total_scaled_MAE_1 = 0.
    total_scaled_MAE_2 = 0.
    total_scaled_MAE_2plus = 0.
    N_1, N_2, N_2plus = 0,0,0
    for i in range(len(GTs)):
        error = np.linalg.norm(GTs[i] - predictions[i])
        total_scaled_MAE += error
        if np.linalg.norm(GTs[i]) >= 0 and np.linalg.norm(GTs[i]) < 1:
            total_scaled_MAE_1 += error
            N_1 += 1
        elif np.linalg.norm(GTs[i]) >= 1 and np.linalg.norm(GTs[i]) < 2:
            total_scaled_MAE_2 += error
            N_2 += 1
        elif np.linalg.norm(GTs[i]) >= 2:
            total_scaled_MAE_2plus += error
            N_2plus += 1

    # ic(total_scaled_MAE/len(GTs)/average_force_norm)
    # ic(average_force_norm)
    avg_relative_error = total_relative_error/len(GTs)
    avg_scaled_MAE = total_scaled_MAE/len(GTs)/average_force_norm
    avg_MAE = total_scaled_MAE/len(GTs)

    if N_2plus == 0:
        avg_2plus = 0
    else:
        avg_2plus = total_scaled_MAE_2plus/N_2plus
    if N_2 == 0:
        avg_2 = 0
    else:
        avg_2 = total_scaled_MAE_2/N_2
    return avg_relative_error, avg_scaled_MAE, avg_MAE, total_scaled_MAE_1/N_1, avg_2, avg_2plus

if __name__ == '__main__':
    import json
    from icecream import ic


    # folder_lists = ['Feb1018:43_final_data_h_20_int_10_seed_1','Feb1018:43_final_data_h_20_int_10_seed_2', 'Feb1018:43_final_data_h_20_int_10_seed_3',\
    #             'Feb1423:19_final_data_h_30_int_10_seed_1_posSpeed','Feb1423:25_final_data_h_30_int_10_seed_2_posSpeed','Feb1500:01_final_data_h_30_int_10_seed_3_posSpeed',\
    #             'Feb1416:54_final_data_h_20_int_5_seed_1_posSpeed','Feb1416:54_final_data_h_20_int_5_seed_2_posSpeed','Feb1420:20_final_data_h_20_int_5_seed_3_posSpeed']
    
    folder_lists =  ['Feb1018:43_final_data_h_20_int_10_seed_1'] #,'Feb1018:43_final_data_h_20_int_10_seed_2', 'Feb1018:43_final_data_h_20_int_10_seed_3'】
                    #['Mar2617:48_final_data_h_20_int_10_seed_1_segAug9_fulldata','Mar2419:59_final_data_h_20_int_10_seed_2_segAug9_fulldata','Mar2419:59_final_data_h_20_int_10_seed_3_segAug9_fulldata']
                    #['Mar2316:24_final_data_h_20_int_10_seed_1_segAug7','Mar2316:36_final_data_h_20_int_10_seed_1_segAug8','Mar2316:46_final_data_h_20_int_10_seed_1_segAug9']
                    #['Mar2001:09_final_data_h_30_int_10_seed_1_mlp','Mar2001:09_final_data_h_30_int_10_seed_2_mlp','Mar2001:06_final_data_h_30_int_10_seed_3_mlp']
                    #['Mar1900:33_final_data_h_20_int_10_seed_1_mlp','Mar1900:32_final_data_h_20_int_10_seed_2_mlp','Mar1900:33_final_data_h_20_int_10_seed_3_mlp']
    
                    # ['Mar2001:38_final_data_h_5_int_10_seed_1_mlp','Mar2019:15_final_data_h_5_int_10_seed_2_mlp','Mar2022:29_final_data_h_5_int_10_seed_3_mlp',\
                    # 'Mar2001:06_final_data_h_10_int_10_seed_1_mlp', 'Mar2001:06_final_data_h_10_int_10_seed_2_mlp','Mar2001:06_final_data_h_10_int_10_seed_3_mlp',\
                    # 'Mar1900:33_final_data_h_20_int_10_seed_1_mlp','Mar1900:32_final_data_h_20_int_10_seed_2_mlp','Mar1900:33_final_data_h_20_int_10_seed_3_mlp',\
                    # 'Mar2001:09_final_data_h_30_int_10_seed_1_mlp','Mar2001:09_final_data_h_30_int_10_seed_2_mlp','Mar2001:06_final_data_h_30_int_10_seed_3_mlp']
                    #['Mar2102:30_final_data_h_20_int_10_seed_1_segAug6_fulldata']
    
                    # ['Mar1719:52_final_data_h_20_int_10_seed_1_segAug3', 'Mar1720:55_final_data_h_20_int_10_seed_1_segAug3_pos',\
                    # 'Mar1822:24_final_data_h_20_int_10_seed_1_segAug4_pos']
                    #['Feb1018:43_final_data_h_20_int_10_seed_1','Feb1018:43_final_data_h_20_int_10_seed_2', 'Feb1018:43_final_data_h_20_int_10_seed_3',\
                    # 'Feb1423:19_final_data_h_30_int_10_seed_1_posSpeed','Feb1423:25_final_data_h_30_int_10_seed_2_posSpeed','Feb1500:01_final_data_h_30_int_10_seed_3_posSpeed',\
                    # 'Feb1620:30_final_data_h_5_int_10_seed_1_posSpeed', 'Feb1621:39_final_data_h_5_int_10_seed_2_posSpeed', 'Feb1621:45_final_data_h_5_int_10_seed_3_posSpeed',\
                    # 'Feb1219:17_final_data_h_10_int_10_seed_1_posSpeed', 'Feb1302:32_final_data_h_10_int_10_seed_2_posSpeed','Feb1302:32_final_data_h_10_int_10_seed_3_posSpeed'] # ,\
                    # 'Feb1020:16_final_data_h_20_int_10_seed_1_pos', 'Feb1022:06_final_data_h_20_int_10_seed_2_pos','Feb1022:06_final_data_h_20_int_10_seed_3_pos',\
                    # 'Feb1022:03_final_data_h_20_int_10_seed_1_posSpeed','Feb1022:03_final_data_h_20_int_10_seed_2_posSpeed','Feb1021:57_final_data_h_20_int_10_seed_3_posSpeed',\
                    # 'Feb1219:17_final_data_h_10_int_10_seed_1_posSpeed', 'Feb1302:32_final_data_h_10_int_10_seed_2_posSpeed','Feb1302:32_final_data_h_10_int_10_seed_3_posSpeed']
    test_data_dir = '/vast/palmer/home.grace/yz2379/project/Data/final_test_segmented/'
    # test_csv_fns = ['white_BG_occlusion4']
    test_csv_fns = [ 'white_BG_bigClamp_slow', 'white_BG_paint_slow', 'white_BG_jelloRed_slow', 'white_BG_yellow3_slow2','white_BG_yellow3_slow3',\
                    'lab_BG_bigClamp_slow', 'lab_BG_paint_slow', 'lab_BG_jelloRed_slow', 'lab_BG_yellow3_slow','square_peg_smooth2','wipe_plate_16'] #,'lab_BG_paint','wipe_plate_16','square_peg']

    saving_folder = 'results/'
    result_log = 'results.csv'
    shifts = [0] #, 5] #[1, 2, 3, 4, 5, 6]
    with open(result_log, 'w') as csvfile:
        logger = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        logger.writerow(['Model_name', 'Problem', 'shift', 'avg_relative_error', 'avg_scaled_MAE', 'avg_MAE'\
            ,'avg_MAE_1','avg_MAE_2','avg_MAE_2plus'])
    

    flip_forces = [False]*9 + [True]*2 #[False]*2 + [True]*2 
    shorten_sequences = [100]*3 #[250]*3 + [200]*3 + [100]*3 + [0]*3 #+ [100]*3 + 
    for shift_index in shifts:
        for folder, shorten_amount in zip(folder_lists,shorten_sequences):
            for test_csv_fn, flip_force in zip(test_csv_fns, flip_forces):
                with open('../runs/' + folder + '/info/parameters.json') as f:
                        params = json.load(f)
                params['test_data_dir'] = test_data_dir
                params['test_csv_fn'] = test_csv_fn + '.csv'
                params['load_model'] = '../runs/' + folder + '/checkpoint/best_val.pt'
                params['bs'] = 32 ##
                args = obj(params)
                fn = saving_folder + folder + '_' + test_csv_fn + '.npz'
                if not os.path.exists(fn):
                    GTs, predictions, indeces = collect_errors(args, flip_force)
                    np.savez(saving_folder + folder + '_' + test_csv_fn, GTs = GTs, predictions = predictions, indeces = indeces)

                result = np.load(fn)
                GTs, predictions, indeces = result['GTs'], result['predictions'], result['indeces']

                if test_csv_fn == 'wipe_plate_16':
                    actual_shorten_amount = 0
                # shorten to make sure the same testing sequence is used
                # indeces = indeces[actual_shorten_amount:]
                if actual_shorten_amount > 0:
                    indeces = indeces[0:-actual_shorten_amount]
                    GTs = GTs[actual_shorten_amount:, :]
                    predictions = predictions[actual_shorten_amount:, :]

                if actual_shift_index > 0:
                    indeces = indeces[0:-actual_shift_index]
                    GTs = GTs[0:-actual_shift_index, :]
                    predictions = predictions[actual_shift_index:, :]

                plot_quantitative_results(GTs, predictions, indeces, saving_folder + folder + '_' +test_csv_fn + '.png')
                avg_relative_error, avg_scaled_MAE, avg_MAE,avg_MAE_1,avg_MAE_2,avg_MAE_2plus = get_stats(GTs, predictions)

                with open(result_log, 'a') as csvfile:
                    logger = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    logger.writerow([folder, test_csv_fn, shift_index, avg_relative_error, avg_scaled_MAE, avg_MAE, avg_MAE_1,avg_MAE_2,avg_MAE_2plus])


# ### settled results
# folder_lists = ['Feb1018:43_final_data_h_20_int_10_seed_1'] #,'Feb1018:43_final_data_h_20_int_10_seed_2', 'Feb1018:43_final_data_h_20_int_10_seed_3' ,\
#                     # 'Feb1020:16_final_data_h_20_int_10_seed_1_pos', 'Feb1022:06_final_data_h_20_int_10_seed_2_pos','Feb1022:06_final_data_h_20_int_10_seed_3_pos',\
#                     # 'Feb1022:03_final_data_h_20_int_10_seed_1_posSpeed','Feb1022:03_final_data_h_20_int_10_seed_2_posSpeed','Feb1021:57_final_data_h_20_int_10_seed_3_posSpeed',\
#                     # 'Feb1219:17_final_data_h_10_int_10_seed_1_posSpeed', 'Feb1302:32_final_data_h_10_int_10_seed_2_posSpeed','Feb1302:32_final_data_h_10_int_10_seed_3_posSpeed']
# test_data_dir = '/vast/palmer/home.grace/yz2379/project/Data/final_test_settled_segmented/'
# test_csv_fns = ['25mm_settled', '30mm_settled', 'bigClamp_settled', 'paint_settled', 'redJello_settled', 'yellow2_settled']


# saving_folder = 'settled_results/'
# result_log = 'results.csv'
# with open(result_log, 'w') as csvfile:
#     logger = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     logger.writerow(['Model_name', 'Problem', 'avg_relative_error', 'avg_scaled_MAE', 'avg_MAE'])

# flip_forces = [True]*4
# shorten_sequences = [0]*9 + [100]*3
# for folder, shorten_amount in zip(folder_lists,shorten_sequences):
#     for test_csv_fn, flip_force in zip(test_csv_fns, flip_forces):
#         with open('../runs/' + folder + '/info/parameters.json') as f:
#                 params = json.load(f)
#         params['test_data_dir'] = test_data_dir
#         params['test_csv_fn'] = test_csv_fn + '.csv'
#         params['load_model'] = '../runs/' + folder + '/checkpoint/best_val.pt'
#         args = obj(params)
#         fn = saving_folder + folder + '_' + test_csv_fn + '.npz'
#         if not os.path.exists(fn):
#             GTs, predictions, indeces = collect_errors(args, flip_force)
#             np.savez(saving_folder + folder + '_' + test_csv_fn, GTs = GTs, predictions = predictions, indeces = indeces)


#         result = np.load(saving_folder + folder + '_' + test_csv_fn + '.npz')
#         GTs, predictions, indeces = result['GTs'], result['predictions'], result['indeces']
        

#         shift_index = 1
#         indeces = indeces[0:-shift_index]
#         GTs = GTs[0:-shift_index, :]
#         predictions = predictions[shift_index:, :]
        
#         # shorten to make sure the same testing sequence is used
#         indeces = indeces[shorten_amount:]
#         GTs = GTs[shorten_amount:, :]
#         predictions = predictions[shorten_amount:, :]

#         plot_quantitative_results(GTs, predictions, indeces, saving_folder + folder + '_' +test_csv_fn + '.png')
#         # avg_relative_error, avg_scaled_MAE, avg_MAE = get_stats(GTs, predictions)
#         #avg_relative_error, avg_scaled_MAE, avg_MAE = get_stats_settled(GTs, predictions, indeces)

#         # with open(result_log, 'a') as csvfile:
#         #     logger = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         #     logger.writerow([folder, test_csv_fn, avg_relative_error, avg_scaled_MAE, avg_MAE])


