import os
from typing import Callable
from collections import defaultdict
from gym.wrappers.monitoring import video_recorder

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv


class VecEvalVideoRecorder(VecEnvWrapper):
    """
    Custom recorder, records a fix number of videos per reward level 
    Wraps a VecEnv or VecEnvWrapper object to record rendered image as mp4 video.
    It requires ffmpeg or avconv to be installed on the machine.

    :param venv:
    :param video_folder: Where to save videos
    :param record_video_trigger: Function that defines when to start recording.
                                        The function takes the current number of step,
                                        and returns whether we should start recording or not.
    :param video_length:  Length of recorded videos
    :param name_prefix: Prefix to the video name
    """

    def __init__(
        self,
        venv: VecEnv,
        video_folder: str,
        num_episodes: int = 1,
        video_length: int = 200,
        name_prefix: str = "procgen-video",
        frames_per_sec: int = 20, 
    ):

        VecEnvWrapper.__init__(self, venv)

        self.env = venv
        # Temp variable to retrieve metadata
        temp_env = venv

        # Unwrap to retrieve metadata dict
        # that will be used by gym recorder
        while isinstance(temp_env, VecEnvWrapper):
            temp_env = temp_env.venv

        if isinstance(temp_env, DummyVecEnv) or isinstance(temp_env, SubprocVecEnv):
            metadata = temp_env.get_attr("metadata")[0]
        else:
            metadata = temp_env.metadata

        self.env.metadata = metadata

        self.num_episodes = num_episodes
        self.episode_count = defaultdict()

        self.video_recorder = None 
        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.traj_id = 0
        self.record_count = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0
        self.frames_per_sec = frames_per_sec

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.start_video_recorder()
        return obs

    def start_video_recorder(self) -> None:
        self.close_video_recorder() 
        video_name = f"{self.name_prefix}-traj-{self.traj_id}-record-{self.record_count}"
        base_path = os.path.join(self.video_folder, video_name)
        # self.env.metadata["render_modes"] = ["rgb_array"]
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env, base_path=base_path, enabled=True, 
            metadata={"traj_id": self.traj_id, "record_count": self.record_count},
        )
        # print(self.env.metadata.get("render_modes"))
        # assert  "rgb_array" in self.env.metadata["render_modes"]
        # # self.video_recorder.enabled = True
        # # self.video_recorder.broken = False
        # # self.video_recorder.path = base_path 
        # self.video_recorder.env = self.env 

        self.video_recorder.frames_per_sec = self.frames_per_sec
        self.video_recorder.output_frames_per_sec = self.frames_per_sec

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self) -> bool:
        return True 

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()
        
        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if True in dones: # self.recorded_frames > self.video_length or 
                self.traj_id += 1
                if self.traj_id == 5:
                # print(self.video_recorder)
                # fname = self.video_recorder.path.split("/")[-1]
                # fname = f'rew-{rews[0]}-' + fname 
                #self.video_recorder.path = "/".join(self.video_recorder.path.split("/")[:-1] + [fname])
                # print(f"Saving video to {self.video_recorder.path}")
                    print("Saving video, reward", rews)
                    self.record_count += 1
                    self.close_video_recorder()
                # for idx, reward in enumerate(rews):
                #     self.episode_count[reward] += 1
        # elif self._video_enabled():
                    self.start_video_recorder()
                    self.traj_id = 0
                    

        return obs, rews, dones, infos

    def close_video_recorder(self) -> None:
        if self.recording:
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 0

    def close(self) -> None:
        VecEnvWrapper.close(self)
        self.close_video_recorder()

    def __del__(self):
        self.close()
