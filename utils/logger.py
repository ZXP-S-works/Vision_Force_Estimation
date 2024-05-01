import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from collections import namedtuple
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import json
from tqdm import tqdm
import collections

ImgForce = collections.namedtuple('ImgForce', 'img, x, f')
plt.style.use('ggplot')


class Logger(object):
    '''
    Logger for train/test runs.

    Args:
      - log_dir: Directory to write log
      - num_envs: Number of environments running concurrently
    '''

    def __init__(self, note, max_epoch):
        # Logging variables
        self.data = {}
        self.max_epoch = max_epoch
        # Create directory in the logging directory
        log_dir = './runs/' + datetime.now().strftime('%b%d%H:%M') # + str(time.time())
        if note is not None:
            log_dir += '_'
            log_dir += note
        self.base_dir = log_dir
        print('Creating logging session at: {}'.format(self.base_dir))

        # Create subdirs to save important run info
        self.info_dir = os.path.join(self.base_dir, 'info')
        self.checkpoint_dir = os.path.join(self.base_dir, 'checkpoint')

        os.makedirs(self.info_dir, exist_ok =True)
        os.makedirs(self.checkpoint_dir, exist_ok =True)

        # Variables to store information
        self.num_epochs = 0
        self.loss, self.l1, self.l2, self.df, self.norm_std_f, self.df_normal_f = [], [], [], [], [], []

    def trainingBookkeeping(self, loss, l1, l2):
        self.loss.append(loss)
        self.l1.append(l1)
        self.l2.append(l2)

    def validatingBookkeeping(self, df, norm_std_f, df_normal_f):
        self.df.append(df)
        self.norm_std_f.append(norm_std_f)
        self.df_normal_f.append(df_normal_f)

    def saveLossCurve(self):
        df = np.asarray(self.df)
        dfpercent = df / np.asarray(self.norm_std_f) * 100
        self.data = {
            'loss': np.asarray(self.loss),
            'l1': np.asarray(self.l1),
            'l2': np.asarray(self.l2),
            'l2_f': df,
            'rela_err': dfpercent,
            'avg_err': np.asarray(self.df_normal_f)
        }

        for key, value in self.data.items():
            if key in ['l2_f', 'rela_err', 'avg_err']:
                if len(value) <= self.max_epoch:
                    plt.plot(value, c='b', label='valid')
                else:
                    plt.plot(value[:-1], c='b', label='valid')
                    plt.plot(np.repeat(value[-1], len(value[:-1])), c='r', label='test')
            else:
                plt.plot(value, label='train')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            # plt.yscale('log')
            plt.legend()
            plt.title(str(key))
            plt.tick_params(axis='y', which='minor')
            plt.grid(True, which='both', axis='y')
            ax = plt.gca()
            ax.yaxis.set_minor_formatter(FormatStrFormatter("%.4f"))
            plt.savefig(os.path.join(self.info_dir, key + '.pdf'))
            plt.close()
            np.save(os.path.join(self.info_dir, key + '.npy'), value)

    def saveModel(self, steps, agent, create_dir=False):
        '''
        Save PyTorch model to log directory

        Args:
          - steps: steps of the current run
          - name: Name to save model as
          - agent: Agent containing model to save
        '''
        if create_dir:
            save_model_path = os.path.join(self.checkpoint_dir, str(steps))
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            agent.saveModel(save_model_path)
        else:
            agent.saveModel(os.path.join(self.checkpoint_dir, str(steps)))

    def saveModelBestVal(self, agent):
        '''
        Save PyTorch model to log directory

        Args:
          - steps: steps of the current run
          - name: Name to save model as
          - agent: Agent containing model to save
        '''
        agent.saveModel(os.path.join(self.checkpoint_dir, 'best_val'))

    def saveParameters(self, parameters):
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        with open(os.path.join(self.info_dir, "parameters.json"), 'w') as f:
            json.dump(parameters, f, cls=NumpyEncoder, indent=2)
