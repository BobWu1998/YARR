import copy
import logging
import os
import time
import pandas as pd

from multiprocessing import Process, Manager
from multiprocessing import get_start_method, set_start_method
from typing import Any

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.agents.agent import ScalarSummary
from yarr.agents.agent import Summary
from yarr.envs.env import Env
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.log_writer import LogWriter
from yarr.utils.process_str import change_case
from yarr.utils.video_utils import CircleCameraMotion, TaskRecorder

from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from yarr.runners._env_runner import _EnvRunner

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

class _IndependentEnvRunner(_EnvRunner):

    def __init__(self,
                 train_env: Env,
                 eval_env: Env,
                 agent: Agent,
                 timesteps: int,
                 train_envs: int,
                 eval_envs: int,
                 rollout_episodes: int,
                 eval_episodes: int,
                 training_iterations: int,
                 eval_from_eps_number: int,
                 episode_length: int,
                 kill_signal: Any,
                 step_signal: Any,
                 num_eval_episodes_signal: Any,
                 eval_epochs_signal: Any,
                 eval_report_signal: Any,
                 log_freq: int,
                 rollout_generator: RolloutGenerator,
                 save_load_lock,
                 current_replay_ratio,
                 target_replay_ratio,
                 weightsdir: str = None,
                 logdir: str = None,
                 env_device: torch.device = None,
                 previous_loaded_weight_folder: str = '',
                 num_eval_runs: int = 1,
                 wrapped_replay = None,
                 calib_scaler = None,
                 action_selection = None
                 ):

            super().__init__(train_env, eval_env, agent, timesteps,
                             train_envs, eval_envs, rollout_episodes, eval_episodes,
                             training_iterations, eval_from_eps_number, episode_length,
                             kill_signal, step_signal, num_eval_episodes_signal,
                             eval_epochs_signal, eval_report_signal, log_freq,
                             rollout_generator, save_load_lock, current_replay_ratio,
                             target_replay_ratio, weightsdir, logdir, env_device,
                             previous_loaded_weight_folder, num_eval_runs, wrapped_replay, 
                             calib_scaler, action_selection)

    def _load_save(self):
        if self._weightsdir is None:
            logging.info("'weightsdir' was None, so not loading weights.")
            return
        while True:
            weight_folders = []
            with self._save_load_lock:
                if os.path.exists(self._weightsdir):
                    weight_folders = os.listdir(self._weightsdir)
                if len(weight_folders) > 0:
                    weight_folders = sorted(map(int, weight_folders))
                    # only load if there has been a new weight saving
                    if self._previous_loaded_weight_folder != weight_folders[-1]:
                        self._previous_loaded_weight_folder = weight_folders[-1]
                        d = os.path.join(self._weightsdir, str(weight_folders[-1]))
                        try:
                            self._agent.load_weights(d)
                        except FileNotFoundError:
                            # rare case when agent hasn't finished writing.
                            time.sleep(1)
                            self._agent.load_weights(d)
                        logging.info('Agent %s: Loaded weights: %s' % (self._name, d))
                        self._new_weights = True
                    else:
                        self._new_weights = False
                    break
            logging.info('Waiting for weights to become available.')
            time.sleep(1)

    def _get_task_name(self):
        if hasattr(self._eval_env, '_task_class'):
            eval_task_name = change_case(self._eval_env._task_class.__name__)
            multi_task = False
        elif hasattr(self._eval_env, '_task_classes'):
            if self._eval_env.active_task_id != -1:
                task_id = (self._eval_env.active_task_id) % len(self._eval_env._task_classes)
                eval_task_name = change_case(self._eval_env._task_classes[task_id].__name__)
            else:
                eval_task_name = ''
            multi_task = True
        else:
            raise Exception('Neither task_class nor task_classes found in eval env')
        return eval_task_name, multi_task


    def draw_episodes(self, log_dir, reliability_results):
        scores, binaries = reliability_results['confidence'], reliability_results['matching_labels']
        # _dir = log_dir + '/100_episodes'
        # if not os.path.exists(_dir):
        #     os.makedirs(_dir)

        def get_max_step(scores):
            max_step = 0
            for score in scores:
                if len(score) > max_step:
                    max_step = len(score)
            return max_step

        # For each time step, plot the scores and binary values for all episodes in a single plot
        max_step = get_max_step(scores)
        for i in range(max_step):
            scores_at_step = []
            binaries_at_step = []

            for j, score in enumerate(scores):
                if i < len(score):
                    scores_at_step.append(score[i])
                    binaries_at_step.append(binaries[j][i])
            
            plt.scatter(scores_at_step, binaries_at_step)
            plt.yticks([0, 1], ['False', 'True'])
            plt.xlabel('Confidence')
            plt.ylabel('Episode Success')
            plt.title('Confidence vs Episode Success')

            # plt.show()
            # save the plot with name 'step_{}.png'
            plt.savefig(log_dir + '/step_{}.png'.format(i))
            plt.close()

    def _run_eval_independent(self, name: str,
                              stats_accumulator,
                              weight,
                              writer_lock,
                              eval=True,
                              device_idx=0,
                              save_metrics=True,
                              cinematic_recorder_cfg=None,
                              ):

        self._name = name
        self._save_metrics = save_metrics
        self._is_test_set = type(weight) == dict

        self._agent = copy.deepcopy(self._agent)

        device = torch.device('cuda:%d' % device_idx) if torch.cuda.device_count() > 1 else torch.device('cuda:0')
        with writer_lock: # hack to prevent multiple CLIP downloads ... argh should use a separate lock
            # self._agent.build(training=False, device=device)
            self._agent.build(training=False, device=device, calib_scaler=self._calib_scaler, action_selection=self._action_selection)

        logging.info('%s: Launching env.' % name)
        # np.random.seed(0) #np.random.seed()
        # torch.manual_seed(0)
        # torch.cuda.manual_seed(0)

        logging.info('Agent information:')
        logging.info(self._agent)

        env = self._eval_env
        env.eval = eval
        env.launch()

        # initialize cinematic recorder if specified
        rec_cfg = cinematic_recorder_cfg
        if rec_cfg.enabled:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam = VisionSensor.create(rec_cfg.camera_resolution)
            cam.set_pose(cam_placeholder.get_pose())
            cam.set_parent(cam_placeholder)

            cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), rec_cfg.rotate_speed)
            tr = TaskRecorder(env, cam_motion, fps=rec_cfg.fps)

            env.env._action_mode.arm_action_mode.set_callable_each_step(tr.take_snap)

        if not os.path.exists(self._weightsdir):
            raise Exception('No weights directory found.')

        # to save or not to save evaluation metrics (set as False for recording videos)
        if self._save_metrics:
            csv_file = 'eval_data.csv' if not self._is_test_set else 'test_data.csv'
            writer = LogWriter(self._logdir, True, True,
                               env_csv=csv_file)

        # one weight for all tasks (used for validation)
        if type(weight) == int:
            logging.info('Evaluating weight %s' % weight)
            weight_path = os.path.join(self._weightsdir, str(weight))
            seed_path = self._weightsdir.replace('/weights', '')
            self._agent.load_weights(weight_path)
            weight_name = str(weight)

        new_transitions = {'train_envs': 0, 'eval_envs': 0}
        total_transitions = {'train_envs': 0, 'eval_envs': 0}
        current_task_id = -1


        reliability_results = {
            'confidence': [],
            'matching_labels': []
        }
        for n_eval in range(self._num_eval_runs):
            if rec_cfg.enabled:
                tr._cam_motion.save_pose()

            # best weight for each task (used for test evaluation)
            if type(weight) == dict:
                task_name = list(weight.keys())[n_eval]
                task_weight = weight[task_name]
                weight_path = os.path.join(self._weightsdir, str(task_weight))
                seed_path = self._weightsdir.replace('/weights', '')
                self._agent.load_weights(weight_path)
                weight_name = str(task_weight)
                print('Evaluating weight %s for %s' % (weight_name, task_name))

            # print('self._wrapped_replay', self._wrapped_replay.dataset())
            if self._wrapped_replay != None:
                dataset = self._wrapped_replay.dataset()
                data_iter = iter(dataset)
            
            # evaluate on N tasks * M episodes per task = total eval episodes
            for ep in range(self._eval_episodes):
                episode_confidence = []
                print('eval episodes {}'.format(ep))
                eval_demo_seed = ep + self._eval_from_eps_number
                logging.info('%s: Starting episode %d, seed %d.' % (name, ep, eval_demo_seed))
                
                # the current task gets reset after every M episodes
                episode_rollout = []

                if self._wrapped_replay != None:
                    sampled_batch = next(data_iter)
                    # print('-------------')
                    # print(sampled_batch)
                    print('iterating...')
                else: 
                    sampled_batch = None
                print('self._episode_length', self._episode_length)
                # batch = {k: v.to(self._train_device) for k, v in sampled_batch.items() if type(v) == torch.Tensor}
                generator = self._rollout_generator.generator(
                    self._step_signal, env, self._agent,
                    self._episode_length, self._timesteps,
                    eval, eval_demo_seed=eval_demo_seed,
                    record_enabled=rec_cfg.enabled, sampled_batch=sampled_batch)
                try:
                    for replay_transition, total_conf in generator:
                        while True:
                            if self._kill_signal.value:
                                env.shutdown()
                                return
                            if (eval or self._target_replay_ratio is None or
                                    self._step_signal.value <= 0 or (
                                            self._current_replay_ratio.value >
                                            self._target_replay_ratio)):
                                break
                            time.sleep(1)
                            logging.debug(
                                'Agent. Waiting for replay_ratio %f to be more than %f' %
                                (self._current_replay_ratio.value, self._target_replay_ratio))

                        with self.write_lock:
                            if len(self.agent_summaries) == 0:
                                # Only store new summaries if the previous ones
                                # have been popped by the main env runner.
                                for s in self._agent.act_summaries():
                                    self.agent_summaries.append(s)
                        episode_rollout.append(replay_transition)
                        episode_confidence.append(total_conf)
                except StopIteration as e:
                    continue
                except Exception as e:
                    env.shutdown()
                    raise e
                print('rollout_length', len(episode_rollout))
                # for epro in episode_rollout:
                #     print(epro.action)

                with self.write_lock:
                    for transition in episode_rollout:
                        self.stored_transitions.append((name, transition, eval))

                        new_transitions['eval_envs'] += 1
                        total_transitions['eval_envs'] += 1
                        stats_accumulator.step(transition, eval)
                        current_task_id = transition.info['active_task_id']

                self._num_eval_episodes_signal.value += 1
                # for ep_ro in episode_rollout:
                #     print('action', ep_ro.action)

                task_name, _ = self._get_task_name()
                reward = episode_rollout[-1].reward
                lang_goal = env._lang_goal
                print(f"Evaluating {task_name} | Episode {ep} | Score: {reward} | Lang Goal: {lang_goal}")

                # add the confidence to the matching labels
                reliability_results['confidence'].append(episode_confidence)
                if reward == 100:
                    reliability_results['matching_labels'].append([1 for _ in range(len(episode_confidence))])
                else:
                    reliability_results['matching_labels'].append([0 for _ in range(len(episode_confidence))])

                # exit()
                # save recording
                print('rec_cfg', rec_cfg)
                if rec_cfg.enabled:
                    success = reward > 0.99
                    record_file = os.path.join(seed_path, 'videos',
                                               '%s_w%s_s%s_%s.mp4' % (task_name,
                                                                      weight_name,
                                                                      eval_demo_seed,
                                                                      'succ' if success else 'fail'))

                    lang_goal = self._eval_env._lang_goal
                    print('record_file', record_file)
                    tr.save(record_file, lang_goal, reward)
                    tr._cam_motion.restore_pose()
                # success = reward > 0.99
                # record_file = os.path.join(seed_path, 'videos',
                #                             '%s_w%s_s%s_%s.mp4' % (task_name,
                #                                                     weight_name,
                #                                                     eval_demo_seed,
                #                                                     'succ' if success else 'fail'))

                # lang_goal = self._eval_env._lang_goal
                # print('record_file', record_file)
                # tr.save(record_file, lang_goal, reward)
                # tr._cam_motion.restore_pose()
            # make the success/failure plots
            ### v1
            # colors = ['blue' if b else 'red' for b in reliability_results['matching_labels']]
            # plt.scatter(reliability_results['confidence'], reliability_results['matching_labels'], c=colors)#, s=2)
            # plt.yticks([0, 1], ['False', 'True'])
            # plt.xlabel('Scores')
            # plt.ylabel('Binary Value')
            # plt.title('Scores vs Binary Value')
            # plt.savefig('/home/DIR/shared/results/success_fail_episodes.png') 
            
            # eval_task_name, multi_task = self._get_task_name()
            # dir_path = 'Set DIR' + eval_task_name
            # os.makedirs(dir_path, exist_ok=True)

            # # # Draw the T/F for all episodes
            # # cmap = cm.rainbow(np.linspace(0, 1, len(reliability_results['confidence'])))

            # # for i, score in enumerate(reliability_results['confidence']):
            # #     color = cmap[i]
                
            # #     for j, s in enumerate(score):
            # #         plt.scatter(s, reliability_results['matching_labels'][i][j], color=color)
            # #         plt.annotate(str(j), (s, reliability_results['matching_labels'][i][j]))
                    
            # #     plt.yticks([0, 1], ['False', 'True'])
            # #     plt.xlabel('Confidence')
            # #     plt.ylabel('Episode Success')
            # #     plt.title('Confidence vs Episode Success')
            # #     handles = [plt.Rectangle((0,0),1,1, color=c) for c in cmap]
            # #     plt.legend(handles, ['Episode {}'.format(i+1) for i in range(len(reliability_results['confidence']))], loc='center right')
            # # plt.savefig(dir_path + '/' + eval_task_name + 'success_fail_episodes_colored.png')
            # # plt.close() 

            # self.draw_episodes(dir_path, reliability_results)
            # report summaries
            summaries = []
            summaries.extend(stats_accumulator.pop())

            eval_task_name, multi_task = self._get_task_name()

            if eval_task_name and multi_task:
                for s in summaries:
                    if 'eval' in s.name:
                        s.name = '%s/%s' % (s.name, eval_task_name)

            if len(summaries) > 0:
                if multi_task:
                    task_score = [s.value for s in summaries if f'eval_envs/return/{eval_task_name}' in s.name][0]
                else:
                    task_score = [s.value for s in summaries if f'eval_envs/return' in s.name][0]
            else:
                task_score = "unknown"

            print(f"Finished {eval_task_name} | Final Score: {task_score}\n")
            
            
            ## adding log for current episode
            tau = self._action_selection._tau
            search_size = self._action_selection._search_size
            search_step = self._action_selection._search_step
            # Create a file name based on tau, search_size, and search_step
            if self._action_selection.enabled:
                action_type = 'safe'
            else:
                action_type = 'best'
            file_name = f"tau_{tau}_search_size_{search_size}_search_step_{search_step}_{action_type}.txt"

            # Full path to the file
            file_path = self._action_selection.log_dir
            # Ensure the directory exists
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Write the output to the file
            with open(file_path+file_name, 'w') as f:
                f.write(f"tau: {tau}\n")
                f.write(f"search_size: {search_size}\n")
                f.write(f"search_step: {search_step}\n")
                f.write(f"Finished {eval_task_name} | Final Score: {task_score}\n")

            if self._save_metrics:
                with writer_lock:
                    writer.add_summaries(weight_name, summaries)

            self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
            self.agent_summaries[:] = []
            self.stored_transitions[:] = []

        if self._save_metrics:
            with writer_lock:
                writer.end_iteration()

        logging.info('Finished evaluation.')
        env.shutdown()

    def kill(self):
        self._kill_signal.value = True
