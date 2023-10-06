import copy
import logging
import os
import shutil
import signal
import sys
import threading
import time
from typing import Optional, List
from typing import Union

from omegaconf import DictConfig
import gc
import numpy as np
import psutil
import torch
import pandas as pd
from yarr.agents.agent import Agent
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer
from yarr.runners.env_runner import EnvRunner
from yarr.runners.train_runner import TrainRunner
from yarr.utils.log_writer import LogWriter
from yarr.utils.stat_accumulator import StatAccumulator
from yarr.replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer

from uncertainty_module.src.base.calib_scaling import CalibScaler
# from uncertainty_module.temperature_scaling import TemperatureScaler
from uncertainty_module.action_selection import ActionSelection
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from torch.utils.tensorboard import SummaryWriter

class OfflineTrainRunner():

    def __init__(self,
                 agent: Agent,
                 wrapped_replay_buffer: PyTorchReplayBuffer,
                 train_device: torch.device,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 iterations: int = int(6e6),
                 logdir: str = '/tmp/yarr/logs',
                 logging_level: int = logging.INFO,
                 log_freq: int = 10,
                 weightsdir: str = '/tmp/yarr/weights',
                 num_weights_to_keep: int = 60,
                 save_freq: int = 100,
                 tensorboard_logging: bool = True,
                 csv_logging: bool = False,
                 load_existing_weights: bool = True,
                 rank: int = None,
                 world_size: int = None,
                 task_name: str = None,
                 calib_scaler: CalibScaler = None,
                 action_selection: ActionSelection = None):
        self._agent = agent
        self._wrapped_buffer = wrapped_replay_buffer
        self._stat_accumulator = stat_accumulator
        self._iterations = iterations
        self._logdir = logdir
        self._logging_level = logging_level
        self._log_freq = log_freq
        self._weightsdir = weightsdir
        self._num_weights_to_keep = num_weights_to_keep
        self._save_freq = save_freq

        self._wrapped_buffer = wrapped_replay_buffer
        self._train_device = train_device
        self._tensorboard_logging = tensorboard_logging
        self._csv_logging = csv_logging
        self._load_existing_weights = load_existing_weights
        self._rank = rank
        self._world_size = world_size
        self._task_name = task_name
        self._calib_scaler = calib_scaler
        self._action_selection = action_selection
        print('OfflineTrainRunner constructing')
        self._writer = None
        # if logdir is None:
        #     logging.info("'logdir' was None. No logging will take place.")
        # else:
        #     self._writer = LogWriter(
        #         self._logdir, tensorboard_logging, csv_logging)
        print('logdir',logdir)
        if weightsdir is None:
            logging.info(
                "'weightsdir' was None. No weight saving will take place.")
        else:
            os.makedirs(self._weightsdir, exist_ok=True)
            
        print('OfflineTrainRunner constructed')

    def _save_model(self, i):
        d = os.path.join(self._weightsdir, str(i))
        os.makedirs(d, exist_ok=True)
        self._agent.save_weights(d)

        # remove oldest save
        prev_dir = os.path.join(self._weightsdir, str(
            i - self._save_freq * self._num_weights_to_keep))
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)

    def _step(self, i, sampled_batch):
        update_dict, update_conf = self._agent.update(i, sampled_batch)
        total_losses = update_dict['total_losses'].item()
        total_conf = update_conf['total_conf']
        true_pred = update_conf['total_true_pred']
        total_scaler_loss = update_conf['total_scaler_loss']
        scaler = update_conf['scaler']
        return total_losses, total_conf, true_pred, total_scaler_loss, scaler

    def _get_resume_eval_epoch(self):
        starting_epoch = 0
        eval_csv_file = self._weightsdir.replace('weights', 'eval_data.csv') # TODO(mohit): check if it's supposed be 'env_data.csv'
        if os.path.exists(eval_csv_file):
             eval_dict = pd.read_csv(eval_csv_file).to_dict()
             epochs = list(eval_dict['step'].values())
             return epochs[-1] if len(epochs) > 0 else starting_epoch
        else:
            return starting_epoch

    def start(self):
        logging.getLogger().setLevel(self._logging_level)
        self._agent = copy.deepcopy(self._agent)
        if self._calib_scaler.training:
            if self._calib_scaler.calib_type == 'vector':
                run = wandb.init(project='vector_train_v7', entity='your_entity', name=self._task_name
                                                                                +'_'+str(self._calib_scaler.lr)
                                                                                +'_'+str(self._calib_scaler.div_penalty))
            else:
                un = wandb.init(project='temp_train_v7', entity='your_entity', name=self._task_name)
        
        self._agent.build(training=True, device=self._train_device, calib_scaler=self._calib_scaler, action_selection = self._action_selection)
        if not self._calib_scaler.training:
            self._calib_scaler.load_parameter(task_name=self._task_name)
        
        if self._weightsdir is not None:
            existing_weights = sorted([int(f) for f in os.listdir(self._weightsdir)])
            if (not self._load_existing_weights) or len(existing_weights) == 0:
                self._save_model(0)
                start_iter = 0
            else:
                resume_iteration = existing_weights[-1]
                print(os.path.join(self._weightsdir, str(resume_iteration)))
                self._agent.load_weights(os.path.join(self._weightsdir, str(resume_iteration)))
                start_iter = resume_iteration + 1
                if self._rank == 0:
                    logging.info(f"Resuming training from iteration {resume_iteration} ...")
        # import pdb
        # pdb.set_trace()
        
        # print('existing_weights', existing_weights)
        # print('resume iteration', resume_iteration)
        # exit()
        dataset = self._wrapped_buffer.dataset()
        data_iter = iter(dataset)

        process = psutil.Process(os.getpid())
        num_cpu = psutil.cpu_count()
        if not self._calib_scaler.training:
            self._iterations = start_iter+1000 #1000 #1000 #1000 #1000 #TODO: add support for this
        else:
            self._iterations = 600000+self._calib_scaler.training_iter #1000 #1000 #1000 #1000 #TODO: add support for this
        logging.info('start_iter {}'.format(start_iter))
        logging.info('self._iterations {}'.format(self._iterations))

        reliability_results = {
            'confidence': [],
            'matching_labels': []
        }
        
        for i in range(start_iter, self._iterations):
            print('iterations: {}'.format(i))
            log_iteration = i % self._log_freq == 0 and i > 0

            if log_iteration:
                process.cpu_percent(interval=None)

            t = time.time()
            sampled_batch = next(data_iter)
            sample_time = time.time() - t

            batch = {k: v.to(self._train_device) for k, v in sampled_batch.items() if type(v) == torch.Tensor}
            t = time.time()
            loss, total_conf, true_pred, total_scaler_loss, scaler = self._step(i, batch)
            reliability_results['confidence'].append(total_conf[0])
            reliability_results['matching_labels'].append(true_pred)
            step_time = time.time() - t
            logging.info('self._rank {}'.format(self._rank))
            # print('total_scaler_loss', total_scaler_loss)
            if self._calib_scaler.training:
                if self._calib_scaler.calib_type == 'temperature':
                    wandb.log({'epoch loss': total_scaler_loss.cpu().item(), 'temperature': scaler.cpu().item()})
                else:
                    wandb.log({'epoch loss': total_scaler_loss.cpu().item(), 
                               'non_scaled_penalty': scaler['non_scaled_penalty'],
                               'weight_mean': scaler['weight_mean'],
                               'weight_min': scaler['weight_min'],
                               'weight_max': scaler['weight_max'],
                               'bias_mean': scaler['bias_mean'],
                               'bias_min': scaler['bias_min'],
                               'bias_max': scaler['bias_max']
                               })
                # self._temp_writer.add_scalar('epoch loss', total_temp_scaler_loss, t)
                # self._temp_writer.add_scalar('temperature', temp_scaler, t)
                
            if self._rank == 0:
                if log_iteration and self._writer is not None:
                    agent_summaries = self._agent.update_summaries()
                    self._writer.add_summaries(i, agent_summaries)

                    self._writer.add_scalar(
                        i, 'monitoring/memory_gb',
                        process.memory_info().rss * 1e-9)
                    self._writer.add_scalar(
                        i, 'monitoring/cpu_percent',
                        process.cpu_percent(interval=None) / num_cpu)

                    logging.info(f"Train Step {i:06d} | Loss: {loss:0.5f} | Sample time: {sample_time:0.6f} | Step time: {step_time:0.4f}.")
                    logging.info(f"Train Step {i:06d} | Scaling Loss: {total_scaler_loss:0.5f} | Sample time: {sample_time:0.6f} | Step time: {step_time:0.4f}.")
                    
                # self._writer.end_iteration()

                if i % self._save_freq == 0 and self._weightsdir is not None:
                    # torch.save(self._agent.scaler.temperature, 'temperature.pth')
                    # torch.save(temp_scaler, temp_log_path+self._task_name+'_temperature.pth')
                    self._save_model(i)
                    
                    
        # if self._temperature_scaler.training:
        #     self._temp_writer.close()
        if self._calib_scaler.training:
            # torch.save(temp_scaler, temp_log_path+self._task_name+'_temperature.pth')
            self._calib_scaler.save_parameter(task_name=self._task_name)
            run.finish()
        if self._rank == 0 and self._writer is not None:
            self._writer.close()
            logging.info('Stopping envs ...')

            self._wrapped_buffer.replay_buffer.shutdown()

        for k, v in reliability_results.items():

            reliability_results[k] = np.array(v)

        thresholds = [0.1*i for i in range(100)]
        confidence_bin = np.zeros(len(reliability_results['confidence']))
        for thresh in thresholds:
            mask = reliability_results['confidence'] > thresh
            indices = np.where(mask)[0]
            confidence_bin[indices] += 1

        num_bins = 100
        accuracy = np.zeros(num_bins)
        for i in range(num_bins):
            print(i)
            if (sum(reliability_results['matching_labels'][confidence_bin == i])) == 0:
                accuracy[i] = 0
            else:
                accuracy[i] = sum(reliability_results['matching_labels'][confidence_bin == i]) / len(reliability_results['matching_labels'][confidence_bin == i])
        print('logging confidence-accuracy plot')
        print('accuracy', accuracy)
        fig, ax = plt.subplots()

        bars = ax.bar(thresholds, accuracy, width=1/num_bins, edgecolor='black')

        # Add some labels and title
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(self._task_name + ' Reliability Diagram')

        # Display the plot
        plt.tight_layout()
        
        # # create the log folder
        # path = "Set this DIR" 
        # if os.path.exists(path):
        #     shutil.rmtree(path)
        # os.makedirs(path)

        # plt.savefig(path+'/random_reliab.png') 
        # print('logging confidence-accuracy plot done! ')
        # plt.close()
        # torch.save(reliability_results, path+'/results.pth')

        # # make the success/failure plots
        # colors = ['blue' if b else 'red' for b in reliability_results['matching_labels']]
        # plt.scatter(reliability_results['confidence'], reliability_results['matching_labels'], c=colors, s=2)
        # plt.yticks([0, 1], ['False', 'True'])
        # plt.xlabel('Confidence')
        # plt.ylabel('Step Success')
        # plt.title(self._task_name + ' Confidence vs Step Success')
        # plt.savefig(path+'/random_success_fail.png') 
        # plt.close()
        # # plt.show()
        # for i in range(10):
        #     print(len(reliability_results['matching_labels'][confidence_bin == i]))

        def count_child_processes():
            current_process = psutil.Process(os.getpid())
            return len(current_process.children(recursive=True))

        print('num_processes', count_child_processes())

        def kill_child_processes():
            current_process = psutil.Process(os.getpid())
            children = current_process.children(recursive=True)  # Get all child processes
            for child in children:
                child.kill()
                try:
                    child.wait(timeout=3)  # Wait up to 3 seconds for process to terminate
                except psutil.TimeoutExpired:
                    print(f"Child process {child.pid} did not terminate in time.")
                
                if child.is_running():
                    print(f"Child process {child.pid} is still running.")
                else:
                    print(f"Child process {child.pid} terminated successfully.")


        kill_child_processes() ## !!! DIRTY FIX
        print('num_processes', count_child_processes())