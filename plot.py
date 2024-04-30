import os
import numpy as np
import matplotlib.pyplot as plt
import inspect

data_size = ['train_set_size_300',
             'train_set_size_1000',
             'train_set_size_3000',
             'train_set_size_10000']
hidden = ['n_hidden_8',
          'n_hidden_16',
          'n_hidden_32',
          'n_hidden_64']
aug = ['n_hidden_32',
       'rot_aug15_trans_aug0',
       'rot_aug0_trans_aug10',
       'rot_aug0_trans_aug0']
res = ['resolution_64',
       'resolution_128',
       'resolution_256',
       'resolution_512']
res_aug = ['res_64_taug_5',
           'resolution_128',
           'res_256_taug_20',
           'res_512_taug_40']
expose = ['no_robot_base',
          'n_hidden_32']

if __name__ == '__main__':
    data_dict = {}

    # Path to the root directory containing the folders
    experiment = '0630_sim_res'
    root_dir = "./server_runs/" + experiment

    # Loop through the folders
    for method in os.listdir(root_dir):
        method_path = os.path.join(root_dir, method)
        if os.path.isdir(method_path):
            for folder_name in os.listdir(method_path):
                folder_path = os.path.join(root_dir, method, folder_name, 'info')

            # if os.path.isdir(folder_path):

                # Loop through files in the folder
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)

                    if file_name.endswith("df%.npy"):
                    # if file_name.endswith("df_normal_f.npy"):
                        # Load the npy file
                        data = np.load(file_path)

                        if method in data_dict:
                            data_dict[method].append(data)
                        else:
                            data_dict[method] = [data]

    fig, ax = plt.subplots(1)
    sorted_methods = list(data_dict.keys())
    sorted_methods.sort()
    for method in sorted_methods:
        error = np.stack(data_dict[method], 1).min(axis=0)
        print(method, error)
        plt.bar(method, error.mean(), yerr=error.std(), capsize=4)
        fig.autofmt_xdate()
    plt.ylabel('average error (%)')
    # plt.ylabel('relative error')
    plt.savefig(root_dir + '/' + experiment + '_avg_errors.pdf')
    # plt.savefig(root_dir + '/' + experiment + '_relative_errors.pdf')
    plt.close()
