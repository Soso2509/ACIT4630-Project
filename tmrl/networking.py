# standard library imports
import datetime
import os
import socket
import time
import atexit
import json
import shutil
import tempfile
import itertools
import csv
from os.path import exists

# third-party imports
import numpy as np
from requests import get
from tlspyo import Relay, Endpoint
import keyboard
import csv

import torch
import torch.nn as nn

import pygame
pygame.init()
pygame.joystick.init()

# local imports
from tmrl.actor import ActorModule
from tmrl.util import dump, load, partial_to_dict
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj

import logging
import wandb


__docformat__ = "google"


# PRINT: ============================================


def print_with_timestamp(s):
    x = datetime.datetime.now()
    sx = x.strftime("%x %X ")
    logging.info(sx + str(s))


def print_ip():
    public_ip = get('http://api.ipify.org').text
    local_ip = socket.gethostbyname(socket.gethostname())
    print_with_timestamp(f"public IP: {public_ip}, local IP: {local_ip}")


# BUFFER: ===========================================

# Imitation Learning Neural Networks ================

class BCNetSmall(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class BCNetMedium(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class BCNetLarge(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# =================================================

class Buffer:
    """
    Buffer of training samples.

    `Server`, `RolloutWorker` and `Trainer` all have their own `Buffer` to store and send training samples.

    Samples are tuples of the form (`act`, `new_obs`, `rew`, `terminated`, `truncated`, `info`)
    """
    def __init__(self, maxlen=cfg.BUFFERS_MAXLEN):
        """
        Args:
            maxlen (int): buffer length
        """
        self.memory = []
        self.stat_train_return = 0.0  # stores the train return
        self.stat_test_return = 0.0  # stores the test return
        self.stat_train_steps = 0  # stores the number of steps per training episode
        self.stat_test_steps = 0  # stores the number of steps per test episode
        self.maxlen = maxlen

    def clip_to_maxlen(self):
        lenmem = len(self.memory)
        if lenmem > self.maxlen:
            print_with_timestamp("buffer overflow. Discarding old samples.")
            self.memory = self.memory[(lenmem - self.maxlen):]

    def append_sample(self, sample):
        """
        Appends `sample` to the buffer.

        Args:
            sample (Tuple): a training sample of the form (`act`, `new_obs`, `rew`, `terminated`, `truncated`, `info`)
        """
        self.memory.append(sample)
        self.clip_to_maxlen()

    def clear(self):
        """
        Clears the buffer but keeps train and test returns.
        """
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def __iadd__(self, other):
        self.memory += other.memory
        self.clip_to_maxlen()
        self.stat_train_return = other.stat_train_return
        self.stat_test_return = other.stat_test_return
        self.stat_train_steps = other.stat_train_steps
        self.stat_test_steps = other.stat_test_steps
        return self


# SERVER SERVER: =====================================


class Server:
    """
    Central server.

    The `Server` lets 1 `Trainer` and n `RolloutWorkers` connect.
    It buffers experiences sent by workers and periodically sends these to the trainer.
    It also receives the weights from the trainer and broadcasts these to the connected workers.
    """
    def __init__(self,
                 port=cfg.PORT,
                 password=cfg.PASSWORD,
                 local_port=cfg.LOCAL_PORT_SERVER,
                 header_size=cfg.HEADER_SIZE,
                 security=cfg.SECURITY,
                 keys_dir=cfg.CREDENTIALS_DIRECTORY,
                 max_workers=cfg.NB_WORKERS):
        """
        Args:
            port (int): tlspyo public port
            password (str): tlspyo password
            local_port (int): tlspyo local communication port
            header_size (int): tlspyo header size (bytes)
            security (Union[str, None]): tlspyo security type (None or "TLS")
            keys_dir (str): tlspyo credentials directory
            max_workers (int): max number of accepted workers
        """
        self.__relay = Relay(port=port,
                             password=password,
                             accepted_groups={
                                 'trainers': {
                                     'max_count': 1,
                                     'max_consumables': None},
                                 'workers': {
                                     'max_count': max_workers,
                                     'max_consumables': None}},
                             local_com_port=local_port,
                             header_size=header_size,
                             security=security,
                             keys_dir=keys_dir)


# TRAINER: ==========================================


class TrainerInterface:
    """
    This is the trainer's network interface
    This connects to the server
    This receives samples batches and sends new weights
    """
    def __init__(self,
                 server_ip=None,
                 server_port=cfg.PORT,
                 password=cfg.PASSWORD,
                 local_com_port=cfg.LOCAL_PORT_TRAINER,
                 header_size=cfg.HEADER_SIZE,
                 max_buf_len=cfg.BUFFER_SIZE,
                 security=cfg.SECURITY,
                 keys_dir=cfg.CREDENTIALS_DIRECTORY,
                 hostname=cfg.HOSTNAME,
                 model_path=cfg.MODEL_PATH_TRAINER):

        self.model_path = model_path
        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'
        self.__endpoint = Endpoint(ip_server=self.server_ip,
                                   port=server_port,
                                   password=password,
                                   groups="trainers",
                                   local_com_port=local_com_port,
                                   header_size=header_size,
                                   max_buf_len=max_buf_len,
                                   security=security,
                                   keys_dir=keys_dir,
                                   hostname=hostname)

        print_with_timestamp(f"server IP: {self.server_ip}")

        self.__endpoint.notify(groups={'trainers': -1})  # retrieve everything

    def broadcast_model(self, model: ActorModule):
        """
        model must be an ActorModule
        broadcasts the model's weights to all connected RolloutWorkers
        """
        model.save(self.model_path)
        with open(self.model_path, 'rb') as f:
            weights = f.read()
        self.__endpoint.broadcast(weights, "workers")

    def retrieve_buffer(self):
        """
        returns the TrainerInterface's buffer of training samples
        """
        buffers = self.__endpoint.receive_all()
        res = Buffer()
        for buf in buffers:
            res += buf
        self.__endpoint.notify(groups={'trainers': -1})  # retrieve everything
        return res


def log_environment_variables():
    """
    add certain relevant environment variables to our config
    usage: `LOG_VARIABLES='HOME JOBID' python ...`
    """
    return {k: os.environ.get(k, '') for k in os.environ.get('LOG_VARIABLES', '').strip().split()}


def load_run_instance(checkpoint_path):
    """
    Default function used to load trainers from checkpoint path
    Args:
        checkpoint_path: the path where instances of run_cls are checkpointed
    Returns:
        An instance of run_cls loaded from checkpoint_path
    """
    return load(checkpoint_path)


def dump_run_instance(run_instance, checkpoint_path):
    """
    Default function used to dump trainers to checkpoint path
    Args:
        run_instance: the instance of run_cls to checkpoint
        checkpoint_path: the path where instances of run_cls are checkpointed
    """
    dump(run_instance, checkpoint_path)


def iterate_epochs(run_cls,
                   interface: TrainerInterface,
                   checkpoint_path: str,
                   dump_run_instance_fn=dump_run_instance,
                   load_run_instance_fn=load_run_instance,
                   epochs_between_checkpoints=1,
                   updater_fn=None):
    """
    Main training loop (remote)
    The run_cls instance is saved in checkpoint_path at the end of each epoch
    The model weights are sent to the RolloutWorker every model_checkpoint_interval epochs
    Generator yielding episode statistics (list of pd.Series) while running and checkpointing
    """
    checkpoint_path = checkpoint_path or tempfile.mktemp("_remove_on_exit")

    try:
        logging.debug(f"checkpoint_path: {checkpoint_path}")
        if not exists(checkpoint_path):
            logging.info(f"=== specification ".ljust(70, "="))
            run_instance = run_cls()
            dump_run_instance_fn(run_instance, checkpoint_path)
            logging.info(f"")
        else:
            logging.info(f"Loading checkpoint...")
            t1 = time.time()
            run_instance = load_run_instance_fn(checkpoint_path)
            logging.info(f" Loaded checkpoint in {time.time() - t1} seconds.")
            if updater_fn is not None:
                logging.info(f"Updating checkpoint...")
                t1 = time.time()
                run_instance = updater_fn(run_instance, run_cls)
                logging.info(f"Checkpoint updated in {time.time() - t1} seconds.")

        while run_instance.epoch < run_instance.epochs:
            # time.sleep(1)  # on network file systems writing files is asynchronous and we need to wait for sync
            yield run_instance.run_epoch(interface=interface)  # yield stats data frame (this makes this function a generator)
            if run_instance.epoch % epochs_between_checkpoints == 0:
                logging.info(f" saving checkpoint...")
                t1 = time.time()
                dump_run_instance_fn(run_instance, checkpoint_path)
                logging.info(f" saved checkpoint in {time.time() - t1} seconds.")
                # we delete and reload the run_instance from disk to ensure the exact same code runs regardless of interruptions
                # del run_instance
                # gc.collect()  # garbage collection
                # run_instance = load_run_instance_fn(checkpoint_path)

    finally:
        if checkpoint_path.endswith("_remove_on_exit") and exists(checkpoint_path):
            os.remove(checkpoint_path)


def run_with_wandb(entity, project, run_id, interface, run_cls, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None, updater_fn=None):
    """
    Main training loop (remote).

    saves config and stats to https://wandb.com
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    wandb_dir = tempfile.mkdtemp()  # prevent wandb from polluting the home directory
    atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)  # clean up after wandb atexit handler finishes
    logging.debug(f" run_cls: {run_cls}")
    config = partial_to_dict(run_cls)
    config['environ'] = log_environment_variables()
    # config['git'] = git_info()  # TODO: check this for bugs
    resume = checkpoint_path and exists(checkpoint_path)
    wandb_initialized = False
    err_cpt = 0
    while not wandb_initialized:
        try:
            wandb.init(dir=wandb_dir, entity=entity, project=project, id=run_id, resume=resume, config=config)
            wandb_initialized = True
        except Exception as e:
            err_cpt += 1
            logging.warning(f"wandb error {err_cpt}: {e}")
            if err_cpt > 10:
                logging.warning(f"Could not connect to wandb, aborting.")
                exit()
            else:
                time.sleep(10.0)
    # logging.info(config)
    for stats in iterate_epochs(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, 1, updater_fn):
        [wandb.log(json.loads(s.to_json())) for s in stats]


def run(interface, run_cls, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None, updater_fn=None):
    """
    Main training loop (remote).
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
    load_run_instance_fn = load_run_instance_fn or load_run_instance
    for stats in iterate_epochs(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, 1, updater_fn):
        pass


class Trainer:
    """
    Training entity.

    The `Trainer` object is where RL training happens.
    Typically, it can be located on a HPC cluster.
    """
    def __init__(self,
                 training_cls=cfg_obj.TRAINER,
                 server_ip=cfg.SERVER_IP_FOR_TRAINER,
                 server_port=cfg.PORT,
                 password=cfg.PASSWORD,
                 local_com_port=cfg.LOCAL_PORT_TRAINER,
                 header_size=cfg.HEADER_SIZE,
                 max_buf_len=cfg.BUFFER_SIZE,
                 security=cfg.SECURITY,
                 keys_dir=cfg.CREDENTIALS_DIRECTORY,
                 hostname=cfg.HOSTNAME,
                 model_path=cfg.MODEL_PATH_TRAINER,
                 checkpoint_path=cfg.CHECKPOINT_PATH,
                 dump_run_instance_fn: callable = None,
                 load_run_instance_fn: callable = None,
                 updater_fn: callable = None):
        """
        Args:
            training_cls (type): training class (subclass of tmrl.training_offline.TrainingOffline)
            server_ip (str): ip of the central `Server`
            server_port (int): public port of the central `Server`
            password (str): password of the central `Server`
            local_com_port (int): port used by `tlspyo` for local communication
            header_size (int): number of bytes used for `tlspyo` headers
            max_buf_len (int): maximum number of messages queued by `tlspyo`
            security (str): `tlspyo security type` (None or "TLS")
            keys_dir (str): custom credentials directory for `tlspyo` TLS security
            hostname (str): custom TLS hostname
            model_path (str): path where a local copy of the model will be saved
            checkpoint_path: path where the `Trainer` will be checkpointed (`None` = no checkpointing)
            dump_run_instance_fn (callable): custom serializer (`None` = pickle.dump)
            load_run_instance_fn (callable): custom deserializer (`None` = pickle.load)
            updater_fn (callable): custom updater (`None` = no updater). If provided, this must be a function \
            that takes a checkpoint and training_cls as argument and returns an updated checkpoint. \
            The updater is called after a checkpoint is loaded, e.g., to update your checkpoint with new arguments.
        """
        self.checkpoint_path = checkpoint_path
        self.dump_run_instance_fn = dump_run_instance_fn
        self.load_run_instance_fn = load_run_instance_fn
        self.updater_fn = updater_fn
        self.training_cls = training_cls
        self.interface = TrainerInterface(server_ip=server_ip,
                                          server_port=server_port,
                                          password=password,
                                          local_com_port=local_com_port,
                                          header_size=header_size,
                                          max_buf_len=max_buf_len,
                                          security=security,
                                          keys_dir=keys_dir,
                                          hostname=hostname,
                                          model_path=model_path)

    def run(self):
        """
        Runs training.
        """
        run(interface=self.interface,
            run_cls=self.training_cls,
            checkpoint_path=self.checkpoint_path,
            dump_run_instance_fn=self.dump_run_instance_fn,
            load_run_instance_fn=self.load_run_instance_fn,
            updater_fn=self.updater_fn)

    def run_with_wandb(self,
                       entity=cfg.WANDB_ENTITY,
                       project=cfg.WANDB_PROJECT,
                       run_id=cfg.WANDB_RUN_ID,
                       key=None):
        """
        Runs training while logging metrics to wandb_.

        .. _wandb: https://wandb.ai

        Args:
            entity (str): wandb entity
            project (str): wandb project
            run_id (str): name of the run
            key (str): wandb API key
        """
        if key is not None:
            os.environ['WANDB_API_KEY'] = key
        run_with_wandb(entity=entity,
                       project=project,
                       run_id=run_id,
                       interface=self.interface,
                       run_cls=self.training_cls,
                       checkpoint_path=self.checkpoint_path,
                       dump_run_instance_fn=self.dump_run_instance_fn,
                       load_run_instance_fn=self.load_run_instance_fn,
                       updater_fn=self.updater_fn)


# ROLLOUT WORKER: ===================================


class RolloutWorker:
    """Actor.

    A `RolloutWorker` deploys the current policy in the environment.
    A `RolloutWorker` may connect to a `Server` to which it sends buffered experience.
    Alternatively, it may exist in standalone mode for deployment.
    """
    def __init__(
            self,
            env_cls,
            actor_module_cls,
            sample_compressor: callable = None,
            device="cpu",
            max_samples_per_episode=np.inf,
            model_path=cfg.MODEL_PATH_WORKER,
            obs_preprocessor: callable = None,
            crc_debug=False,
            model_path_history=cfg.MODEL_PATH_SAVE_HISTORY,
            model_history=cfg.MODEL_HISTORY,
            standalone=False,
            server_ip=None,
            server_port=cfg.PORT,
            password=cfg.PASSWORD,
            local_port=cfg.LOCAL_PORT_WORKER,
            header_size=cfg.HEADER_SIZE,
            max_buf_len=cfg.BUFFER_SIZE,
            security=cfg.SECURITY,
            keys_dir=cfg.CREDENTIALS_DIRECTORY,
            hostname=cfg.HOSTNAME,
            model_path_IL="bc_model.pth"
    ):
        """
        Args:
            env_cls (type): class of the Gymnasium environment (subclass of tmrl.envs.GenericGymEnv)
            actor_module_cls (type): class of the module containing the policy (subclass of tmrl.actor.ActorModule)
            sample_compressor (callable): compressor for sending samples over the Internet; \
            when not `None`, `sample_compressor` must be a function that takes the following arguments: \
            (prev_act, obs, rew, terminated, truncated, info), and that returns them (modified) in the same order: \
            when not `None`, a `sample_compressor` works with a corresponding decompression scheme in the `Memory` class
            device (str): device on which the policy is running
            max_samples_per_episode (int): if an episode gets longer than this, it is reset
            model_path (str): path where a local copy of the policy will be stored
            obs_preprocessor (callable): utility for modifying observations retrieved from the environment; \
            when not `None`, `obs_preprocessor` must be a function that takes an observation as input and outputs the \
            modified observation
            crc_debug (bool): useful for debugging custom pipelines; leave to False otherwise
            model_path_history (str): (include the filename but omit .tmod) path to the saved history of policies; \
            we recommend you leave this to the default
            model_history (int): policies are saved every `model_history` new policies (0: not saved)
            standalone (bool): if True, the worker will not try to connect to a server
            server_ip (str): ip of the central server
            server_port (int): public port of the central server
            password (str): tlspyo password
            local_port (int): tlspyo local communication port; usually, leave this to the default
            header_size (int): tlspyo header size (bytes)
            max_buf_len (int): tlspyo max number of messages in buffer
            security (str): tlspyo security type (None or "TLS")
            keys_dir (str): tlspyo credentials directory; usually, leave this to the default
            hostname (str): tlspyo hostname; usually, leave this to the default
        """
        self.obs_preprocessor = obs_preprocessor
        self.get_local_buffer_sample = sample_compressor
        self.env = env_cls()
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.model_path = model_path
        self.model_path_history = model_path_history
        self.device = device
        self.actor = actor_module_cls(observation_space=obs_space, action_space=act_space).to_device(self.device)
        self.standalone = standalone
        if os.path.isfile(self.model_path):
            logging.debug(f"Loading model from {self.model_path}")
            self.actor = self.actor.load(self.model_path, device=self.device)
        else:
            logging.debug(f"No model found at {self.model_path}")
        self.buffer = Buffer()
        self.max_samples_per_episode = max_samples_per_episode
        self.crc_debug = crc_debug
        self.model_history = model_history
        self._cur_hist_cpt = 0
        self.model_cpt = 0

        self.debug_ts_cpt = 0
        self.debug_ts_res_cpt = 0

        self.IL_chance = 0.7
        self.prev_episode_reward = 0.0
        self.episode_start_time = time.time()


        self.device = torch.device(device)


        self.max_samples_per_episode = max_samples_per_episode

        # === Setup BC model ===
        # Observation shape: assuming (velocity + lidar)
        obs_sample, _ = self.env.reset()
        flat_obs = self.flatten_obs(obs_sample)
        self.model_path_IL = model_path_IL  # e.g., "bc_model.pth"
        self.model_IL = self._load_best_bcnet_model(len(flat_obs), self.model_path_IL)
        self.model_IL.load_state_dict(torch.load(self.model_path_IL, map_location=self.device))
        self.model_IL.eval()

        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'

        print_with_timestamp(f"server IP: {self.server_ip}")

        if not self.standalone:
            self.__endpoint = Endpoint(ip_server=self.server_ip,
                                       port=server_port,
                                       password=password,
                                       groups="workers",
                                       local_com_port=local_port,
                                       header_size=header_size,
                                       max_buf_len=max_buf_len,
                                       security=security,
                                       keys_dir=keys_dir,
                                       hostname=hostname,
                                       deserializer_mode="synchronous")
        else:
            self.__endpoint = None


    def _load_best_bcnet_model(self, input_dim, path, output_dim=3):
        model_classes = [BCNetSmall, BCNetMedium, BCNetLarge]
        for ModelCls in model_classes:
            try:
                model = ModelCls(input_dim, output_dim).to(self.device)
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()
                print(f"[INFO] Successfully loaded: {ModelCls.__name__}")
                return model
            except Exception as e:
                print(f"[WARNING] Failed to load {ModelCls.__name__}: {e}")
        raise RuntimeError("No compatible BCNet model variant could be loaded.")


    def flatten_obs(self, obs):
        velocity, lidar = obs[0], obs[1]
        return velocity.tolist() + lidar.flatten().tolist()


    def act(self, obs, rew=0.0, test=False):
        """
        Select an action based on observation obs

        Args:
            obs (nested structure): observation
            test (bool): directly passed to the act() method of the ActorModule

        Returns:
            numpy.array: action computed by the ActorModule
        """
        # if self.obs_preprocessor is not None:
        #     obs = self.obs_preprocessor(obs)
        flat_obs = self.flatten_obs(obs)
        x = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            IL_act = self.model_IL(x).squeeze().cpu().numpy()
        IL_act = np.clip(IL_act, -1.0, 1.0)

        RL_act = self.actor.act_(obs, test=test)

        print(f"IL_chance: {np.round(self.IL_chance, 2)}, RL: {np.round(RL_act, 2)}, IL: {np.round(IL_act, 2)}")

        return IL_act if np.random.rand() < self.IL_chance else RL_act


    def reset(self, collect_samples):
        """
        Starts a new episode.

        Args:
            collect_samples (bool): if True, samples are buffered and sent to the `Server`

        Returns:
            Tuple:
            (nested structure: observation retrieved from the environment,
            dict: information retrieved from the environment)
        """
        
        RL_chance = self.prev_episode_reward * 0.005
        self.IL_chance = max(0.0, min(1.0, 0.8 - RL_chance))  # Keep within [0, 1]

        print_with_timestamp(f"New episode: IL_chance set to {np.round(self.IL_chance, 3)} based on previous reward: {np.round(self.prev_episode_reward, 2)}")
      
        obs = None
        try:
            # Faster than hasattr() in real-time environments
            act = self.env.unwrapped.default_action  # .astype(np.float32)
        except AttributeError:
            # In non-real-time environments, act is None on reset
            act = None
        new_obs, info = self.env.reset()
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        rew = 0.0
        terminated, truncated = False, False
        if collect_samples:
            if self.crc_debug:
                self.debug_ts_cpt += 1
                self.debug_ts_res_cpt = 0
                info['crc_sample'] = (obs, act, new_obs, rew, terminated, truncated)
                info['crc_sample_ts'] = (self.debug_ts_cpt, self.debug_ts_res_cpt)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(act, new_obs, rew, terminated, truncated, info)
            else:
                sample = act, new_obs, rew, terminated, truncated, info
            self.buffer.append_sample(sample)
        self.episode_start_time = time.time()
        return new_obs, info

    def step(self, obs, test, collect_samples, last_step=False):
        """
        Performs a full RL transition.

        A full RL transition is `obs` -> `act` -> `new_obs`, `rew`, `terminated`, `truncated`, `info`.
        Note that, in the Real-Time RL setting, `act` is appended to a buffer which is part of `new_obs`.

        Args:
            obs (nested structure): previous observation
            test (bool): passed to the `act()` method of the `ActorModule`
            collect_samples (bool): if True, samples are buffered and sent to the `Server`
            last_step (bool): if True and `terminated` is False, `truncated` will be set to True

        Returns:
            Tuple:
            (nested structure: new observation,
            float: new reward,
            bool: episode termination signal,
            bool: episode truncation signal,
            dict: information dictionary)
        """
        elapsed_time = round(time.time() - self.episode_start_time, 2)
        #print(f"Elapsed episode time: {elapsed_time} seconds")

        # Pass previous reward to act()
        act = self.act(obs, test=test) #rew=self.prev_rew,

        # Step in environment
        new_obs, rew, terminated, truncated, info = self.env.step(act)

        # Update previous reward for next step
        self.prev_rew = rew

        # Preprocess new observation
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)

        if collect_samples:
            if last_step and not terminated:
                truncated = True

            if self.crc_debug:
                self.debug_ts_cpt += 1
                self.debug_ts_res_cpt += 1
                info['crc_sample'] = (obs, act, new_obs, rew, terminated, truncated)
                info['crc_sample_ts'] = (self.debug_ts_cpt, self.debug_ts_res_cpt)

            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(act, new_obs, rew, terminated, truncated, info)
            else:
                sample = act, new_obs, rew, terminated, truncated, info



            #act_rounded = np.round(act, 2)
            print(rew)
                

        # Save when the agent made it to the goal within time
        if terminated and not truncated and rew > 40:
          
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            elapsed_time = round(time.time() - self.episode_start_time, 2)

            data = {

            "timestamp": timestamp,
            "elapsed_time_seconds": elapsed_time,
            "IL_chance": round(float(self.IL_chance), 3) 
            }

            filename = "goal_timestamps.json"

            # If file exists, load existing list
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    existing_data = json.load(f)
            else:
                existing_data = []


            # Add the new entry
            existing_data.append(data)

            # Write back updated list
            with open(filename, "w") as f:
                json.dump(existing_data, f, indent=4)


        self.buffer.append_sample(sample)

        if terminated == True or truncated == True:
            self.episode_start_time = time.time()


        return new_obs, rew, terminated, truncated, info


    def collect_train_episode(self, max_samples=None):
        """
        Collects a maximum of `max_samples` training transitions (from reset to terminated or truncated)

        This method stores the episode and the training return in the local `Buffer` of the worker
        for sending to the `Server`.

        Args:
            max_samples (int): if the environment is not `terminated` after `max_samples` time steps,
                it is forcefully reset and `truncated` is set to True.
        """
        if max_samples is None:
            max_samples = self.max_samples_per_episode

        iterator = range(max_samples) if max_samples != np.inf else itertools.count()

        ret = 0.0
        steps = 0
        obs, info = self.reset(collect_samples=True)
        for i in iterator:
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=False, collect_samples=True, last_step=i == max_samples - 1)
            ret += rew
            steps += 1
            if terminated or truncated:
                break
        self.buffer.stat_train_return = ret
        self.buffer.stat_train_steps = steps
        self.prev_episode_reward = ret  # Save reward for next episode’s IL chance

    def run_episodes(self, max_samples_per_episode=None, nb_episodes=np.inf, train=False):
        """
        Runs `nb_episodes` episodes.

        Args:
            max_samples_per_episode (int): same as run_episode
            nb_episodes (int): total number of episodes to collect
            train (bool): same as run_episode
        """
        if max_samples_per_episode is None:
            max_samples_per_episode = self.max_samples_per_episode

        iterator = range(nb_episodes) if nb_episodes != np.inf else itertools.count()

        for _ in iterator:
            self.run_episode(max_samples_per_episode, train=train)

    def run_episode(self, max_samples=None, train=False):
        """
        Collects a maximum of `max_samples` test transitions (from reset to terminated or truncated).

        Args:
            max_samples (int): At most `max_samples` samples are collected per episode.
                If the episode is longer, it is forcefully reset and `truncated` is set to True.
            train (bool): whether the episode is a training or a test episode.
                `step` is called with `test=not train`.
        """
        if max_samples is None:
            max_samples = self.max_samples_per_episode

        iterator = range(max_samples) if max_samples != np.inf else itertools.count()

        ret = 0.0
        steps = 0
        obs, info = self.reset(collect_samples=False)
        for _ in iterator:
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=not train, collect_samples=False)
            ret += rew
            steps += 1
            if terminated or truncated:
                break
        self.buffer.stat_test_return = ret
        self.buffer.stat_test_steps = steps

    def run(self, test_episode_interval=0, nb_episodes=np.inf, verbose=True, expert=False):
        """
        Runs the worker for `nb_episodes` episodes.

        This method sends episodes continuously to the Server, and checks for new weights between episodes.
        For synchronous or more fine-grained sampling, use synchronous or lower-level APIs.
        For deployment, use `run_episodes` rather than `run`.

        Args:
            test_episode_interval (int): a test episode is collected for every `test_episode_interval` train episodes;
                set to 0 to not collect test episodes.
            nb_episodes (int): maximum number of train episodes to collect.
            verbose (bool): whether to log INFO messages.
            expert (bool): experts send training samples without updating their model nor running test episodes.
        """

        iterator = range(nb_episodes) if nb_episodes != np.inf else itertools.count()

        if expert:
            if not verbose:
                for _ in iterator:
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.ignore_actor_weights()
            else:
                for _ in iterator:
                    print_with_timestamp("collecting expert episode")
                    self.collect_train_episode(self.max_samples_per_episode)
                    print_with_timestamp("copying buffer for sending")
                    self.send_and_clear_buffer()
                    self.ignore_actor_weights()
        elif not verbose:
            if not test_episode_interval:
                for _ in iterator:
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.update_actor_weights(verbose=False)
            else:
                for episode in iterator:
                    if episode % test_episode_interval == 0 and not self.crc_debug:
                        self.run_episode(self.max_samples_per_episode, train=False)
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.update_actor_weights(verbose=False)
        else:
            for episode in iterator:
                if test_episode_interval and episode % test_episode_interval == 0 and not self.crc_debug:
                    print_with_timestamp("running test episode")
                    self.run_episode(self.max_samples_per_episode, train=False)
                print_with_timestamp("collecting train episode")
                self.collect_train_episode(self.max_samples_per_episode)
                print_with_timestamp("copying buffer for sending")
                self.send_and_clear_buffer()
                print_with_timestamp("checking for new weights")
                self.update_actor_weights(verbose=True)

    def run_synchronous(self,
                        test_episode_interval=0,
                        nb_steps=np.inf,
                        initial_steps=1,
                        max_steps_per_update=np.inf,
                        end_episodes=True,
                        verbose=False):
        """
        Collects `nb_steps` steps while synchronizing with the Trainer.

        This method is useful for traditional (non-real-time) environments that can be stepped fast.
        It also works for rtgym environments with `wait_on_done` enabled, just set `end_episodes` to `True`.

        Note: This method does not collect test episodes. Periodically use `run_episode(train=False)` if you wish to.

        Args:
            test_episode_interval (int): a test episode is collected for every `test_episode_interval` train episodes;
                set to 0 to not collect test episodes. NB: `end_episodes` must be `True` to collect test episodes.
            nb_steps (int): total number of steps to collect (after `initial_steps`).
            initial_steps (int): initial number of steps to collect before waiting for the first model update.
            max_steps_per_update (float): maximum number of steps to collect per model received from the Server
                (this can be a non-integer ratio).
            end_episodes (bool): when True, waits for episodes to end before sending samples and waiting for updates.
                When False (default), pauses whenever the max_steps_per_update ratio is exceeded.
            verbose (bool): whether to log INFO messages.
        """

        # collect initial samples

        if verbose:
            logging.info(f"Collecting {initial_steps} initial steps")

        iteration = 0
        done = False
        while iteration < initial_steps:
            steps = 0
            ret = 0.0
            # reset
            obs, info = self.reset(collect_samples=True)
            done = False
            iteration += 1
            # episode
            while not done and (end_episodes or iteration < initial_steps):
                # step
                obs, rew, terminated, truncated, info = self.step(obs=obs,
                                                                  test=False,
                                                                  collect_samples=True,
                                                                  last_step=steps == self.max_samples_per_episode - 1)
                iteration += 1
                steps += 1
                ret += rew
                done = terminated or truncated
            # send the collected samples to the Server
            self.buffer.stat_train_return = ret
            self.buffer.stat_train_steps = steps
            if verbose:
                logging.info(f"Sending buffer (initial steps)")
            self.send_and_clear_buffer()

        i_model = 1

        # wait for the first updated model if required here
        ratio = (iteration + 1) / i_model
        while ratio > max_steps_per_update:
            if verbose:
                logging.info(f"Ratio {ratio} > {max_steps_per_update}, sending buffer checking updates")
            self.send_and_clear_buffer()
            i_model += self.update_actor_weights(verbose=verbose, blocking=True)
            ratio = (iteration + 1) / i_model

        # collect further samples while synchronizing with the Trainer

        iteration = 0
        episode = 0
        steps = 0
        ret = 0.0

        while iteration < nb_steps:

            if done:
                # test episode
                if test_episode_interval > 0 and episode % test_episode_interval == 0 and end_episodes:
                    if verbose:
                        print_with_timestamp("running test episode")
                    self.run_episode(self.max_samples_per_episode, train=False)
                # reset
                obs, info = self.reset(collect_samples=True)
                done = False
                iteration += 1
                steps = 0
                ret = 0.0
                episode += 1

            while not done and (end_episodes or ratio <= max_steps_per_update):

                # step
                obs, rew, terminated, truncated, info = self.step(obs=obs,
                                                                  test=False,
                                                                  collect_samples=True,
                                                                  last_step=steps == self.max_samples_per_episode - 1)
                iteration += 1
                steps += 1
                ret += rew

                done = terminated or truncated

                if not end_episodes:
                    # check model and send samples after each step
                    ratio = (iteration + 1) / i_model
                    while ratio > max_steps_per_update:
                        if verbose:
                            logging.info(f"Ratio {ratio} > {max_steps_per_update}, sending buffer checking updates (no eoe)")
                        if not done:
                            if verbose:
                                logging.info(f"Sending buffer (no eoe)")
                            self.send_and_clear_buffer()
                        i_model += self.update_actor_weights(verbose=verbose, blocking=True)
                        ratio = (iteration + 1) / i_model

            if end_episodes:
                # check model and send samples only after episodes end
                ratio = (iteration + 1) / i_model
                while ratio > max_steps_per_update:
                    if verbose:
                        logging.info(
                            f"Ratio {ratio} > {max_steps_per_update}, sending buffer checking updates (eoe)")
                    if not done:
                        if verbose:
                            logging.info(f"Sending buffer (eoe)")
                        self.send_and_clear_buffer()
                    i_model += self.update_actor_weights(verbose=verbose, blocking=True)
                    ratio = (iteration + 1) / i_model

            self.buffer.stat_train_return = ret
            self.buffer.stat_train_steps = steps
            if verbose:
                logging.info(f"Sending buffer - DEBUG ratio {ratio} iteration {iteration} i_model {i_model}")
            self.send_and_clear_buffer()

    def run_env_benchmark(self, nb_steps, test=False, verbose=True):
        """
        Benchmarks the environment.

        This method is only compatible with rtgym_ environments.
        Furthermore, the `"benchmark"` option of the rtgym configuration dictionary must be set to `True`.

        .. _rtgym: https://github.com/yannbouteiller/rtgym

        Args:
            nb_steps (int): number of steps to perform to compute the benchmark
            test (int): whether the actor is called in test or train mode
            verbose (bool): whether to log INFO messages
        """
        if nb_steps == np.inf or nb_steps < 0:
            raise RuntimeError(f"Invalid number of steps: {nb_steps}")

        obs, info = self.reset(collect_samples=False)
        for _ in range(nb_steps):
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=test, collect_samples=False)
            if terminated or truncated:
                break
        res = self.env.benchmarks()
        if verbose:
            print_with_timestamp(f"Benchmark results:\n{res}")
        return res

    def send_and_clear_buffer(self):
        """
        Sends the buffered samples to the `Server`.
        """
        self.__endpoint.produce(self.buffer, "trainers")
        self.buffer.clear()

    def update_actor_weights(self, verbose=True, blocking=False):
        """
        Updates the actor with new weights received from the `Server` when available.

        Args:
            verbose (bool): whether to log INFO messages.
            blocking (bool): if True, blocks until a model is received; otherwise, can be a no-op.

        Returns:
            int: number of new actor models received from the Server (the latest is used).
        """
        weights_list = self.__endpoint.receive_all(blocking=blocking)
        nb_received = len(weights_list)
        if nb_received > 0:
            weights = weights_list[-1]
            with open(self.model_path, 'wb') as f:
                f.write(weights)
            if self.model_history:
                self._cur_hist_cpt += 1
                if self._cur_hist_cpt == self.model_history:
                    x = datetime.datetime.now()
                    with open(self.model_path_history + str(x.strftime("%d_%m_%Y_%H_%M_%S")) + ".tmod", 'wb') as f:
                        f.write(weights)
                    self._cur_hist_cpt = 0
                    if verbose:
                        print_with_timestamp("model weights saved in history")
            self.actor = self.actor.load(self.model_path, device=self.device)
            if verbose:
                print_with_timestamp("model weights have been updated")
        return nb_received

    def ignore_actor_weights(self):
        """
        Clears the buffer of weights received from the `Server`.

        This is useful for expert RolloutWorkers, because all RolloutWorkers receive weights.

        Returns:
            int: number of new (ignored) actor models received from the Server.
        """
        weights_list = self.__endpoint.receive_all(blocking=False)
        nb_received = len(weights_list)
        return nb_received



# This is old. Not in use
class BCNet(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim)
    )


    def forward(self, x):
        return self.net(x)

# IMITATION WORKER (This runs the IL model on its own)
class ImitationWorker:
    def __init__(self, env_cls, device="cpu", model_path="bc_model.pth", max_samples_per_episode=np.inf):
        self.device = torch.device(device)
        self.env = env_cls()
        self.model_path = model_path
        self.max_samples_per_episode = max_samples_per_episode

        # === Setup BC model ===
        # Observation shape: assuming (velocity + lidar)
        obs_sample, _ = self.env.reset()
        flat_obs = self.flatten_obs(obs_sample)
        self.model = self._load_best_bcnet_model(len(flat_obs), self.model_path)


    def flatten_obs(self, obs):
        velocity, lidar = obs[0], obs[1]
        return velocity.tolist() + lidar.flatten().tolist()

    def _load_best_bcnet_model(self, input_dim, path, output_dim=3):
        model_classes = [BCNetSmall, BCNetMedium, BCNetLarge]
        for ModelCls in model_classes:
            try:
                model = ModelCls(input_dim, output_dim).to(self.device)
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()
                print(f"[INFO] Successfully loaded: {ModelCls.__name__}")
                return model
            except Exception as e:
                print(f"[WARNING] Failed to load {ModelCls.__name__}: {e}")
        raise RuntimeError("No compatible BCNet model variant could be loaded.")



    def act(self, obs):
        flat_obs = self.flatten_obs(obs)
        x = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act = self.model(x).squeeze().cpu().numpy()
        act = np.clip(act, -1.0, 1.0) 
        #act = [0.51, 0.0, 0]
        print(act)
        return act

    def run(self, nb_episodes=np.inf, verbose=True):
        episode_iter = range(nb_episodes) if nb_episodes != np.inf else iter(int, 1)

        for episode in episode_iter:
            if verbose:
                print(f"\n[EPISODE {episode+1}] Starting episode...")

            # Start timer
            run_start_time = time.time()

            obs, info = self.env.reset()
            done = False
            total_rew = 0.0
            steps = 0

            for _ in range(int(self.max_samples_per_episode)):
                act = self.act(obs)
                obs, rew, terminated, truncated, info = self.env.step(act)
                total_rew += rew
                steps += 1

                if terminated or truncated:
                    break

            # End timer
            run_time = round(time.time() - run_start_time, 2)

            if verbose:
                print(f"[EPISODE {episode+1}] Finished: reward={total_rew:.2f}, steps={steps}, time={run_time}s")

            # Save results to IL_rew.json
            log_entry = {
                "run_ID": episode + 1,
                "total_rew": round(float(total_rew), 2),
                "run_time": run_time
            }

            filename = "IL_rew.json"
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    data = json.load(f)
            else:
                data = []

            data.append(log_entry)

            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
            





# IMITATION learner (data extractor): ===================================
class Imitation:
    def __init__(
        self,
        env_cls,
        actor_module_cls,
        sample_compressor: callable = None,
        device="cpu",
        max_samples_per_episode=np.inf,
        model_path=cfg.MODEL_PATH_WORKER,
        obs_preprocessor: callable = None,
        crc_debug=False,
        model_path_history=cfg.MODEL_PATH_SAVE_HISTORY,
        model_history=cfg.MODEL_HISTORY,
        standalone=False,
        server_ip=None,
        server_port=cfg.PORT,
        password=cfg.PASSWORD,
        local_port=cfg.LOCAL_PORT_WORKER,
        header_size=cfg.HEADER_SIZE,
        max_buf_len=cfg.BUFFER_SIZE,
        security=cfg.SECURITY,
        keys_dir=cfg.CREDENTIALS_DIRECTORY,
        hostname=cfg.HOSTNAME
    ):

        self.obs_preprocessor = obs_preprocessor
        self.get_local_buffer_sample = sample_compressor
        self.env = env_cls()
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.model_path = model_path
        self.device = device
        self.standalone = standalone
        self.actor = actor_module_cls(observation_space=obs_space, action_space=act_space).to_device(self.device)

        if os.path.isfile(self.model_path):
            logging.debug(f"Loading model from {self.model_path}")
            self.actor = self.actor.load(self.model_path, device=self.device)
        else:
            logging.debug(f"No model found at {self.model_path}")

        self.buffer = Buffer()
        self.max_samples_per_episode = max_samples_per_episode
        self.crc_debug = crc_debug

        self.server_ip = server_ip
        print_with_timestamp(f"server IP: {self.server_ip}")


        self.__endpoint = Endpoint(ip_server=self.server_ip,
                                   port=server_port,
                                   password=password,
                                   groups="workers",
                                   local_com_port=local_port,
                                   header_size=header_size,
                                   max_buf_len=max_buf_len,
                                   security=security,
                                   keys_dir=keys_dir,
                                   hostname=hostname,
                                   deserializer_mode="synchronous")
        
        self.current_lap_id = 0
        
        self.controller = None
        if pygame.joystick.get_count() > 0:
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()
            print_with_timestamp("Controller connected.")
        else:
            print_with_timestamp("No controller detected, using keyboard.")


    #def act(self, obs):
        #return self.actor.act_(obs)

    def _get_human_action(self):
        pygame.event.pump()  # Needed to update controller state

        steering = 0.0
        throttle = 0.0         # Not certain these values are correct#####
        brake = 0.0

        if self.controller:
            # Example mapping: adjust these based on your controller model
            steering = self.controller.get_axis(0)  # Left stick horizontal
            trigger = self.controller.get_axis(5)   # Right trigger typically goes from -1 (released) to 1 (pressed)
            brake_trigger = self.controller.get_axis(4)  # Left trigger

            # Normalize trigger values to [0, 1]
            throttle = (trigger + 1) / 2
            brake = brake_trigger
        else:
            # Keyboard fallback
            if keyboard.is_pressed('a'):
                steering = -1.0
            elif keyboard.is_pressed('d'):
                steering = 1.0

            if keyboard.is_pressed('w'):
                throttle = 1.0
            if keyboard.is_pressed('s'):
                brake = 1.0

        return np.array([throttle, brake, steering], dtype=np.float32)



    def reset(self, collect_samples=True):
        self.episode_start_time = time.time()
        obs = None
        try:
            act = self.env.unwrapped.default_action
        except AttributeError:
            act = None  # e.g., if not available on reset

        new_obs, info = self.env.reset()
        if self.obs_preprocessor:
            new_obs = self.obs_preprocessor(new_obs)

        rew = 0.0
        terminated, truncated = False, False

        if collect_samples:
            if self.crc_debug:
                self.debug_ts_cpt += 1
                self.debug_ts_res_cpt = 0
                info['crc_sample'] = (obs, act, new_obs, rew, terminated, truncated)
                info['crc_sample_ts'] = (self.debug_ts_cpt, self.debug_ts_res_cpt)

            # Use keyboard inputs at reset time too
            act = self._get_human_action()

            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(act, new_obs, rew, terminated, truncated, info)
            else:
                sample = act, new_obs, rew, terminated, truncated, info

            #self.buffer.append_sample(sample)

        return new_obs, info





    def step(self, obs, last_step=False):

        new_obs, rew, terminated, truncated, info = self.env.step(None)

        #print(f"Observation received: {new_obs}")
        # print(f"Info received: {info}")
        # print(f"Observation space: {self.env.observation_space}")

        act = self._get_human_action()

        if act is None:
            act = np.zeros(self.env.action_space.shape, dtype=np.float32)
        elif isinstance(act, list):
            act = np.array(act, dtype=np.float32)
        elif isinstance(act, np.ndarray):
            act = act.astype(np.float32)

        #print(act)

        if isinstance(new_obs, tuple):
            obs_to_store = new_obs  # store full tuple
        else:
            obs_to_store = new_obs

        if self.obs_preprocessor:
            obs_to_store = self.obs_preprocessor(obs_to_store)

        if last_step and not terminated:
            truncated = True

        rotation = obs_to_store[2]  
        if np.array_equal(rotation, [0.0, 0.0, 0.0]): # This only happens on a new lap
            self.current_lap_id += 1  
            run_id = self.current_lap_id  
        else:
            run_id = getattr(self, "current_lap_id", 0)  

        self.write_to_csv(run_id, act, obs_to_store, terminated, truncated, info)
        
        '''
        # Save when the agent made it to the goal within time
        if terminated and not truncated and rew > 40:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            elapsed_time = round(time.time() - self.episode_start_time, 2)

            data = {
                "timestamp": timestamp,
                "elapsed_time_seconds": elapsed_time
            }

            filename = "goal_timestamps.json"

            # If file exists, load existing list
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            # Add the new entry
            existing_data.append(data)

            # Write back updated list
            with open(filename, "w") as f:
                json.dump(existing_data, f, indent=4)
        '''

        if self.get_local_buffer_sample:
            sample = self.get_local_buffer_sample(act, obs_to_store, rew, terminated, truncated, info)
        else:
            sample = (act, obs_to_store, rew, terminated, truncated, info)
        print(rew)

        return obs, rew, terminated, truncated, info

    def write_to_csv(self, run_id, act, obs_to_store, terminated, truncated, info):
        velocity = obs_to_store[0] if len(obs_to_store) > 0 else None
        lidar = obs_to_store[1] if len(obs_to_store) > 0 else None
        rotation = obs_to_store[2] if len(obs_to_store) > 0 else None
        position = obs_to_store[3] if len(obs_to_store) > 0 else None
        

        velocity = velocity.tolist() if isinstance(velocity, np.ndarray) else velocity
        lidar = lidar.flatten().tolist() if isinstance(lidar, np.ndarray) else lidar
        rotation = rotation.tolist() if isinstance(rotation, np.ndarray) else rotation
        position = position.tolist() if isinstance(position, np.ndarray) else position

        csv_file = "demonstration_data.csv"
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    "Run_ID", "Forward Throttle", "Backward throttle", "Steering",
                    "Velocity", "Lidar", "Rotation", "Position",
                    "Truncated", "Terminated", "Info"
                ])

            writer.writerow([
                run_id, act[0], act[1], act[2],
                velocity, lidar, rotation, position,
                truncated, terminated, info
            ])

        print(f"Data written to {csv_file}: Run {run_id}")



    #def _build_sample(self, act, obs, rew, terminated, truncated, info):
        #if self.get_local_buffer_sample:
            #return self.get_local_buffer_sample(act, obs, rew, terminated, truncated, info)
        #else:
            #return act, obs, rew, terminated, truncated, info

    def collect_train_episode(self):
        obs, info = self.reset()
        done = False
        ret, steps = 0.0, 0

        for i in range(int(self.max_samples_per_episode)):
            obs, rew, terminated, truncated, info = self.step(obs, last_step=i == self.max_samples_per_episode - 1)
            ret += rew
            steps += 1
            if terminated or truncated:
                break

        self.buffer.stat_train_return = ret
        self.buffer.stat_train_steps = steps

    def run(self, nb_episodes=11, verbose=True):
        iterator = range(nb_episodes) if nb_episodes != np.inf else iter(int, 1)  # infinite loop

        for _ in iterator:
            if verbose:
                print_with_timestamp("Collecting expert episode")
            self.collect_train_episode()

            if verbose:
                print_with_timestamp("Sending buffer to trainer")
            self.send_and_clear_buffer()

            self.ignore_actor_weights()

    def send_and_clear_buffer(self):
        self.__endpoint.produce(self.buffer, "trainers")
        self.buffer.clear()

    def ignore_actor_weights(self):
        _ = self.__endpoint.receive_all(blocking=False)


# Pure RL ROLLOUT WORKER: ===================================

class OriginalRolloutWorker:
    """Actor.

    A `RolloutWorker` deploys the current policy in the environment.
    A `RolloutWorker` may connect to a `Server` to which it sends buffered experience.
    Alternatively, it may exist in standalone mode for deployment.
    """
    def __init__(
            self,
            env_cls,
            actor_module_cls,
            sample_compressor: callable = None,
            device="cpu",
            max_samples_per_episode=np.inf,
            model_path=cfg.MODEL_PATH_WORKER,
            obs_preprocessor: callable = None,
            crc_debug=False,
            model_path_history=cfg.MODEL_PATH_SAVE_HISTORY,
            model_history=cfg.MODEL_HISTORY,
            standalone=False,
            server_ip=None,
            server_port=cfg.PORT,
            password=cfg.PASSWORD,
            local_port=cfg.LOCAL_PORT_WORKER,
            header_size=cfg.HEADER_SIZE,
            max_buf_len=cfg.BUFFER_SIZE,
            security=cfg.SECURITY,
            keys_dir=cfg.CREDENTIALS_DIRECTORY,
            hostname=cfg.HOSTNAME
    ):
        """
        Args:
            env_cls (type): class of the Gymnasium environment (subclass of tmrl.envs.GenericGymEnv)
            actor_module_cls (type): class of the module containing the policy (subclass of tmrl.actor.ActorModule)
            sample_compressor (callable): compressor for sending samples over the Internet; \
            when not `None`, `sample_compressor` must be a function that takes the following arguments: \
            (prev_act, obs, rew, terminated, truncated, info), and that returns them (modified) in the same order: \
            when not `None`, a `sample_compressor` works with a corresponding decompression scheme in the `Memory` class
            device (str): device on which the policy is running
            max_samples_per_episode (int): if an episode gets longer than this, it is reset
            model_path (str): path where a local copy of the policy will be stored
            obs_preprocessor (callable): utility for modifying observations retrieved from the environment; \
            when not `None`, `obs_preprocessor` must be a function that takes an observation as input and outputs the \
            modified observation
            crc_debug (bool): useful for debugging custom pipelines; leave to False otherwise
            model_path_history (str): (include the filename but omit .tmod) path to the saved history of policies; \
            we recommend you leave this to the default
            model_history (int): policies are saved every `model_history` new policies (0: not saved)
            standalone (bool): if True, the worker will not try to connect to a server
            server_ip (str): ip of the central server
            server_port (int): public port of the central server
            password (str): tlspyo password
            local_port (int): tlspyo local communication port; usually, leave this to the default
            header_size (int): tlspyo header size (bytes)
            max_buf_len (int): tlspyo max number of messages in buffer
            security (str): tlspyo security type (None or "TLS")
            keys_dir (str): tlspyo credentials directory; usually, leave this to the default
            hostname (str): tlspyo hostname; usually, leave this to the default
        """
        self.obs_preprocessor = obs_preprocessor
        self.get_local_buffer_sample = sample_compressor
        self.env = env_cls()
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.model_path = model_path
        self.model_path_history = model_path_history
        self.device = device
        self.actor = actor_module_cls(observation_space=obs_space, action_space=act_space).to_device(self.device)
        self.standalone = standalone
        if os.path.isfile(self.model_path):
            logging.debug(f"Loading model from {self.model_path}")
            self.actor = self.actor.load(self.model_path, device=self.device)
        else:
            logging.debug(f"No model found at {self.model_path}")
        self.buffer = Buffer()
        self.max_samples_per_episode = max_samples_per_episode
        self.crc_debug = crc_debug
        self.model_history = model_history
        self._cur_hist_cpt = 0
        self.model_cpt = 0

        self.debug_ts_cpt = 0
        self.debug_ts_res_cpt = 0

        self.episode_start_time = None

        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'

        print_with_timestamp(f"server IP: {self.server_ip}")

        if not self.standalone:
            self.__endpoint = Endpoint(ip_server=self.server_ip,
                                       port=server_port,
                                       password=password,
                                       groups="workers",
                                       local_com_port=local_port,
                                       header_size=header_size,
                                       max_buf_len=max_buf_len,
                                       security=security,
                                       keys_dir=keys_dir,
                                       hostname=hostname,
                                       deserializer_mode="synchronous")
        else:
            self.__endpoint = None


    def act(self, obs, test=False):
        """
        Select an action based on observation `obs`

        Args:
            obs (nested structure): observation
            test (bool): directly passed to the `act()` method of the `ActorModule`

        Returns:
            numpy.array: action computed by the `ActorModule`
        """
        # if self.obs_preprocessor is not None:
        #     obs = self.obs_preprocessor(obs)
        action = self.actor.act_(obs, test=test)
        return action

    def reset(self, collect_samples):
        """
        Starts a new episode.

        Args:
            collect_samples (bool): if True, samples are buffered and sent to the `Server`

        Returns:
            Tuple:
            (nested structure: observation retrieved from the environment,
            dict: information retrieved from the environment)
        """
        self.episode_start_time = time.time()
        obs = None
        try:
            # Faster than hasattr() in real-time environments
            act = self.env.unwrapped.default_action  # .astype(np.float32)
        except AttributeError:
            # In non-real-time environments, act is None on reset
            act = None
        new_obs, info = self.env.reset()
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        rew = 0.0
        terminated, truncated = False, False
        if collect_samples:
            if self.crc_debug:
                self.debug_ts_cpt += 1
                self.debug_ts_res_cpt = 0
                info['crc_sample'] = (obs, act, new_obs, rew, terminated, truncated)
                info['crc_sample_ts'] = (self.debug_ts_cpt, self.debug_ts_res_cpt)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(act, new_obs, rew, terminated, truncated, info)
            else:
                sample = act, new_obs, rew, terminated, truncated, info
            self.buffer.append_sample(sample)
        return new_obs, info

    def step(self, obs, test, collect_samples, last_step=False):
        """
        Performs a full RL transition.

        A full RL transition is `obs` -> `act` -> `new_obs`, `rew`, `terminated`, `truncated`, `info`.
        Note that, in the Real-Time RL setting, `act` is appended to a buffer which is part of `new_obs`.
        This is because is does not directly affect the new observation, due to real-time delays.

        Args:
            obs (nested structure): previous observation
            test (bool): passed to the `act()` method of the `ActorModule`
            collect_samples (bool): if True, samples are buffered and sent to the `Server`
            last_step (bool): if True and `terminated` is False, `truncated` will be set to True

        Returns:
            Tuple:
            (nested structure: new observation,
            float: new reward,
            bool: episode termination signal,
            bool: episode truncation signal,
            dict: information dictionary)
        """
        act = self.act(obs, test=test)
        new_obs, rew, terminated, truncated, info = self.env.step(act)
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        if collect_samples:
            if last_step and not terminated:
                truncated = True
            if self.crc_debug:
                self.debug_ts_cpt += 1
                self.debug_ts_res_cpt += 1
                info['crc_sample'] = (obs, act, new_obs, rew, terminated, truncated)
                info['crc_sample_ts'] = (self.debug_ts_cpt, self.debug_ts_res_cpt)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(act, new_obs, rew, terminated, truncated, info)
            else:
                sample = act, new_obs, rew, terminated, truncated, info
            act_rounded = np.round(act, 2)
            print(act_rounded)
            self.buffer.append_sample(sample)  # CAUTION: in the buffer, act is for the PREVIOUS transition (act, obs(act))
        
         # Save when the agent made it to the goal within time
        if rew > 60:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            elapsed_time = round(time.time() - self.episode_start_time, 2)

            data = {
                "timestamp": timestamp,
                "elapse_time_seconds": elapsed_time
                    }

            filename = "goal_timestamps.json"

            # If file exists, load existing list
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            # Add the new timestamp
            existing_data.append(data)

            # Write back updated list
            with open(filename, "w") as f:
                json.dump(existing_data, f, indent=4)

        return new_obs, rew, terminated, truncated, info

    def collect_train_episode(self, max_samples=None):
        """
        Collects a maximum of `max_samples` training transitions (from reset to terminated or truncated)

        This method stores the episode and the training return in the local `Buffer` of the worker
        for sending to the `Server`.

        Args:
            max_samples (int): if the environment is not `terminated` after `max_samples` time steps,
                it is forcefully reset and `truncated` is set to True.
        """
        if max_samples is None:
            max_samples = self.max_samples_per_episode

        iterator = range(max_samples) if max_samples != np.inf else itertools.count()

        ret = 0.0
        steps = 0
        obs, info = self.reset(collect_samples=True)
        for i in iterator:
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=False, collect_samples=True, last_step=i == max_samples - 1)
            ret += rew
            steps += 1
            if terminated or truncated:
                break
        self.buffer.stat_train_return = ret
        self.buffer.stat_train_steps = steps

    def run_episodes(self, max_samples_per_episode=None, nb_episodes=np.inf, train=False):
        """
        Runs `nb_episodes` episodes.

        Args:
            max_samples_per_episode (int): same as run_episode
            nb_episodes (int): total number of episodes to collect
            train (bool): same as run_episode
        """
        if max_samples_per_episode is None:
            max_samples_per_episode = self.max_samples_per_episode

        iterator = range(nb_episodes) if nb_episodes != np.inf else itertools.count()

        for _ in iterator:
            self.run_episode(max_samples_per_episode, train=train)

    def run_episode(self, max_samples=None, train=False):
        """
        Collects a maximum of `max_samples` test transitions (from reset to terminated or truncated).

        Args:
            max_samples (int): At most `max_samples` samples are collected per episode.
                If the episode is longer, it is forcefully reset and `truncated` is set to True.
            train (bool): whether the episode is a training or a test episode.
                `step` is called with `test=not train`.
        """
        if max_samples is None:
            max_samples = self.max_samples_per_episode

        iterator = range(max_samples) if max_samples != np.inf else itertools.count()

        ret = 0.0
        steps = 0
        obs, info = self.reset(collect_samples=False)
        for _ in iterator:
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=not train, collect_samples=False)
            ret += rew
            steps += 1
            if terminated or truncated:
                break
        self.buffer.stat_test_return = ret
        self.buffer.stat_test_steps = steps

    def run(self, test_episode_interval=0, nb_episodes=np.inf, verbose=True, expert=False):
        """
        Runs the worker for `nb_episodes` episodes.

        This method sends episodes continuously to the Server, and checks for new weights between episodes.
        For synchronous or more fine-grained sampling, use synchronous or lower-level APIs.
        For deployment, use `run_episodes` rather than `run`.

        Args:
            test_episode_interval (int): a test episode is collected for every `test_episode_interval` train episodes;
                set to 0 to not collect test episodes.
            nb_episodes (int): maximum number of train episodes to collect.
            verbose (bool): whether to log INFO messages.
            expert (bool): experts send training samples without updating their model nor running test episodes.
        """

        iterator = range(nb_episodes) if nb_episodes != np.inf else itertools.count()

        if expert:
            if not verbose:
                for _ in iterator:
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.ignore_actor_weights()
            else:
                for _ in iterator:
                    print_with_timestamp("collecting expert episode")
                    self.collect_train_episode(self.max_samples_per_episode)
                    print_with_timestamp("copying buffer for sending")
                    self.send_and_clear_buffer()
                    self.ignore_actor_weights()
        elif not verbose:
            if not test_episode_interval:
                for _ in iterator:
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.update_actor_weights(verbose=False)
            else:
                for episode in iterator:
                    if episode % test_episode_interval == 0 and not self.crc_debug:
                        self.run_episode(self.max_samples_per_episode, train=False)
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.update_actor_weights(verbose=False)
        else:
            for episode in iterator:
                if test_episode_interval and episode % test_episode_interval == 0 and not self.crc_debug:
                    print_with_timestamp("running test episode")
                    self.run_episode(self.max_samples_per_episode, train=False)
                print_with_timestamp("collecting train episode")
                self.collect_train_episode(self.max_samples_per_episode)
                print_with_timestamp("copying buffer for sending")
                self.send_and_clear_buffer()
                print_with_timestamp("checking for new weights")
                self.update_actor_weights(verbose=True)

    def run_synchronous(self,
                        test_episode_interval=0,
                        nb_steps=np.inf,
                        initial_steps=1,
                        max_steps_per_update=np.inf,
                        end_episodes=True,
                        verbose=False):
        """
        Collects `nb_steps` steps while synchronizing with the Trainer.

        This method is useful for traditional (non-real-time) environments that can be stepped fast.
        It also works for rtgym environments with `wait_on_done` enabled, just set `end_episodes` to `True`.

        Note: This method does not collect test episodes. Periodically use `run_episode(train=False)` if you wish to.

        Args:
            test_episode_interval (int): a test episode is collected for every `test_episode_interval` train episodes;
                set to 0 to not collect test episodes. NB: `end_episodes` must be `True` to collect test episodes.
            nb_steps (int): total number of steps to collect (after `initial_steps`).
            initial_steps (int): initial number of steps to collect before waiting for the first model update.
            max_steps_per_update (float): maximum number of steps to collect per model received from the Server
                (this can be a non-integer ratio).
            end_episodes (bool): when True, waits for episodes to end before sending samples and waiting for updates.
                When False (default), pauses whenever the max_steps_per_update ratio is exceeded.
            verbose (bool): whether to log INFO messages.
        """

        # collect initial samples

        if verbose:
            logging.info(f"Collecting {initial_steps} initial steps")

        iteration = 0
        done = False
        while iteration < initial_steps:
            steps = 0
            ret = 0.0
            # reset
            obs, info = self.reset(collect_samples=True)
            done = False
            iteration += 1
            # episode
            while not done and (end_episodes or iteration < initial_steps):
                # step
                obs, rew, terminated, truncated, info = self.step(obs=obs,
                                                                  test=False,
                                                                  collect_samples=True,
                                                                  last_step=steps == self.max_samples_per_episode - 1)
                iteration += 1
                steps += 1
                ret += rew
                done = terminated or truncated
            # send the collected samples to the Server
            self.buffer.stat_train_return = ret
            self.buffer.stat_train_steps = steps
            if verbose:
                logging.info(f"Sending buffer (initial steps)")
            self.send_and_clear_buffer()

        i_model = 1

        # wait for the first updated model if required here
        ratio = (iteration + 1) / i_model
        while ratio > max_steps_per_update:
            if verbose:
                logging.info(f"Ratio {ratio} > {max_steps_per_update}, sending buffer checking updates")
            self.send_and_clear_buffer()
            i_model += self.update_actor_weights(verbose=verbose, blocking=True)
            ratio = (iteration + 1) / i_model

        # collect further samples while synchronizing with the Trainer

        iteration = 0
        episode = 0
        steps = 0
        ret = 0.0

        while iteration < nb_steps:

            if done:
                # test episode
                if test_episode_interval > 0 and episode % test_episode_interval == 0 and end_episodes:
                    if verbose:
                        print_with_timestamp("running test episode")
                    self.run_episode(self.max_samples_per_episode, train=False)
                # reset
                obs, info = self.reset(collect_samples=True)
                done = False
                iteration += 1
                steps = 0
                ret = 0.0
                episode += 1

            while not done and (end_episodes or ratio <= max_steps_per_update):

                # step
                obs, rew, terminated, truncated, info = self.step(obs=obs,
                                                                  test=False,
                                                                  collect_samples=True,
                                                                  last_step=steps == self.max_samples_per_episode - 1)
                iteration += 1
                steps += 1
                ret += rew

                done = terminated or truncated

                if not end_episodes:
                    # check model and send samples after each step
                    ratio = (iteration + 1) / i_model
                    while ratio > max_steps_per_update:
                        if verbose:
                            logging.info(f"Ratio {ratio} > {max_steps_per_update}, sending buffer checking updates (no eoe)")
                        if not done:
                            if verbose:
                                logging.info(f"Sending buffer (no eoe)")
                            self.send_and_clear_buffer()
                        i_model += self.update_actor_weights(verbose=verbose, blocking=True)
                        ratio = (iteration + 1) / i_model

            if end_episodes:
                # check model and send samples only after episodes end
                ratio = (iteration + 1) / i_model
                while ratio > max_steps_per_update:
                    if verbose:
                        logging.info(
                            f"Ratio {ratio} > {max_steps_per_update}, sending buffer checking updates (eoe)")
                    if not done:
                        if verbose:
                            logging.info(f"Sending buffer (eoe)")
                        self.send_and_clear_buffer()
                    i_model += self.update_actor_weights(verbose=verbose, blocking=True)
                    ratio = (iteration + 1) / i_model

            self.buffer.stat_train_return = ret
            self.buffer.stat_train_steps = steps
            if verbose:
                logging.info(f"Sending buffer - DEBUG ratio {ratio} iteration {iteration} i_model {i_model}")
            self.send_and_clear_buffer()

    def run_env_benchmark(self, nb_steps, test=False, verbose=True):
        """
        Benchmarks the environment.

        This method is only compatible with rtgym_ environments.
        Furthermore, the `"benchmark"` option of the rtgym configuration dictionary must be set to `True`.

        .. _rtgym: https://github.com/yannbouteiller/rtgym

        Args:
            nb_steps (int): number of steps to perform to compute the benchmark
            test (int): whether the actor is called in test or train mode
            verbose (bool): whether to log INFO messages
        """
        if nb_steps == np.inf or nb_steps < 0:
            raise RuntimeError(f"Invalid number of steps: {nb_steps}")

        obs, info = self.reset(collect_samples=False)
        for _ in range(nb_steps):
            obs, rew, terminated, truncated, info = self.step(obs=obs, test=test, collect_samples=False)
            if terminated or truncated:
                break
        res = self.env.benchmarks()
        if verbose:
            print_with_timestamp(f"Benchmark results:\n{res}")
        return res

    def send_and_clear_buffer(self):
        """
        Sends the buffered samples to the `Server`.
        """
        self.__endpoint.produce(self.buffer, "trainers")
        self.buffer.clear()

    def update_actor_weights(self, verbose=True, blocking=False):
        """
        Updates the actor with new weights received from the `Server` when available.

        Args:
            verbose (bool): whether to log INFO messages.
            blocking (bool): if True, blocks until a model is received; otherwise, can be a no-op.

        Returns:
            int: number of new actor models received from the Server (the latest is used).
        """
        weights_list = self.__endpoint.receive_all(blocking=blocking)
        nb_received = len(weights_list)
        if nb_received > 0:
            weights = weights_list[-1]
            with open(self.model_path, 'wb') as f:
                f.write(weights)
            if self.model_history:
                self._cur_hist_cpt += 1
                if self._cur_hist_cpt == self.model_history:
                    x = datetime.datetime.now()
                    with open(self.model_path_history + str(x.strftime("%d_%m_%Y_%H_%M_%S")) + ".tmod", 'wb') as f:
                        f.write(weights)
                    self._cur_hist_cpt = 0
                    if verbose:
                        print_with_timestamp("model weights saved in history")
            self.actor = self.actor.load(self.model_path, device=self.device)
            if verbose:
                print_with_timestamp("model weights have been updated")
        return nb_received

    def ignore_actor_weights(self):
        """
        Clears the buffer of weights received from the `Server`.

        This is useful for expert RolloutWorkers, because all RolloutWorkers receive weights.

        Returns:
            int: number of new (ignored) actor models received from the Server.
        """
        weights_list = self.__endpoint.receive_all(blocking=False)
        nb_received = len(weights_list)
        return nb_received