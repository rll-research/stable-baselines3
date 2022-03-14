import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np

from stable_baselines3.common import base_class  # pytype: disable=pyi-error
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

from stable_baselines3.common.callbacks import * 
from os.path import join
import time 

class LogEvalCallback(EvalCallback):
    """
    Modify the eval callback to eval based on training steps

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super(LogEvalCallback, self).__init__(
            eval_env=eval_env,
            callback_on_new_best=None,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=None, # just save all models
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
            )
        
        self.evaluations_trainsteps = [] 
        self.model_save_path = join(model_save_path, 'models')


    def _init_callback(self) -> None:
        super()._init_callback()
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # skip this function 
        self.num_timesteps = self.model.num_timesteps
        return True 

    def on_training_end(self):
        if self.model_save_path is not None:
            self.model.save(os.path.join(self.model_save_path, f"{int(self.model._n_updates)}"))

    def _on_log_step(self) -> bool:
        """ Run Eval of model after fixed update steps """
        update_count = int(self.model.iterations) # NOTE(Mandi): for PPO this ignores n_epochs 
        self.num_timesteps = self.model.num_timesteps
        if self.verbose > 2:
            print('Evaluating after trained steps:', self.model._n_updates, update_count)
        if self.eval_freq > 0 and update_count % self.eval_freq == 0: 
            if self.verbose > 1:
                print(f'Evaluating after {update_count} model update steps')
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )
            # Reset success rate buffer
            self._is_success_buffer = []
            start_eval_time = time.time()
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_trainsteps.append(update_count)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval trained iterations ={update_count}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("eval/after_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.record("eval/after_iterations", update_count, exclude="tensorboard")
            self.logger.record("eval/time_elapsed_per_eval", int(time.time() - start_eval_time), exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print(f"New best mean reward! {mean_reward}") 
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()
            if self.model_save_path is not None:
                self.model.save(os.path.join(self.model_save_path, f"{update_count}"))
                if self.verbose > 1:
                    print(
                        'LogEvalCallback saving model to: ',
                        os.path.join(self.model_save_path, f"{update_count}")
                    )
        return True

class NormalizeBufferRewardCallback(BaseCallback):
    """ Normalize the rewards for each task by its std in current rollout buffer """

    def __init__(
        self, 
        rollout_buffer, 
        task_to_envs, 
        subtract_mean=False, 
        memory_length=None,
        verbose=0):
        super(NormalizeBufferRewardCallback, self).__init__(verbose=verbose)
        self.rollout_buffer = rollout_buffer
        self.task_to_envs = task_to_envs
        self.subtract_mean = subtract_mean
        self.task_memory = None
        if memory_length is not None:
            self.task_memory = np.zeros(
                (len(task_to_envs), memory_length))
            self.pos = 0 

    def _init_callback(self) -> None:
        return 

    def _on_step(self) -> bool:
        return True

    def on_rollout_end(self) -> None:
        assert self.rollout_buffer.full, "Rollout buffer is not full"
        assert not self.rollout_buffer.generator_ready, "Need to normalize before mini batch sample!"
        buffer = self.rollout_buffer
        for task_id, env_ids in self.task_to_envs.items():
            env_rewards = buffer.rewards[:, env_ids].flatten()
            # if self.task_memory is not None:
            #     self.task_memory[task_id, self.pos] = env_rewards
            std = np.std(env_rewards)
            if np.isnan(std) or std == 0:
                std = 1 
            if self.subtract_mean:
                buffer.rewards[:, env_ids] -= np.mean(env_rewards)
            buffer.rewards[:, env_ids] /= std
             