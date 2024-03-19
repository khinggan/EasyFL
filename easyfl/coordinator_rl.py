import logging
import os
import random
import sys
import time
from os import path

import numpy as np
import torch
from omegaconf import OmegaConf

from easyfl.client.dqn_client import DQN_Client
from easyfl.server.dqn_server import DQNServer
from easyfl.datasets import TEST_IN_SERVER
from easyfl.distributed import dist_init, get_device
from easyfl.models.model import load_model
from easyfl.server.base import BaseServer
from easyfl.coordinator import Coordinator
import gymnasium as gym

logger = logging.getLogger(__name__)


class Coordinator_RL(Coordinator):
    """Coordinator_RL manages federated reinforcement learning server and client.
    A single instance of coordinator_rl is initialized for each federated learning task
    when the package is imported.
    """

    def __init__(self):
        self.envs = None
        self.policy_net = None
        self.target_net = None

    def init(self, conf, init_all=True):
        """Initialize coordinator

        Args:
            conf (omegaconf.dictconfig.DictConfig): Internal configurations for federated learning.
            init_all (bool): Whether initialize dataset, model, server, and client other than configuration.
        """
        self.init_conf(conf)

        _set_random_seed(conf.seed)

        if init_all:
            self.init_envs()      # not init dataset, init envs

            self.init_model()

            self.init_server()    # same

            self.init_clients()   # RL clients

    def run(self):
        """Run the coordinator and the federated learning process.
        Initialize `torch.distributed` if distributed training is configured.
        """
        start_time = time.time()

        if self.conf.is_distributed:
            dist_init(
                self.conf.distributed.backend,
                self.conf.distributed.init_method,
                self.conf.distributed.world_size,
                self.conf.distributed.rank,
                self.conf.distributed.local_rank,
            )
        self.server.start(self.model, self.clients)
        self.print_("Total training time {:.1f}s".format(time.time() - start_time))

    def init_conf(self, conf):
        """Initialize coordinator configuration.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Configurations.
        """
        self.conf = conf
        self.conf.is_distributed = (self.conf.gpu > 1)
        if self.conf.gpu == 0:
            self.conf.device = "cpu"
        elif self.conf.gpu == 1:
            self.conf.device = "cuda"
        else:
            self.conf.device = get_device(self.conf.gpu, self.conf.distributed.world_size,
                                          self.conf.distributed.local_rank)
        self.print_("Configurations: {}".format(self.conf))

    def init_envs(self):
        """Initialize datasets. Use provided datasets if not registered."""
        if self.registerd_env:
            return

        if self.conf.env.num_of_clients <=0:
            return

        if self.conf.env.env_dist == "same" and self.conf.env.env_type == "gym":
            self.envs = [gym.make(self.conf.env.env_name) for _ in range(self.conf.env.num_of_clients)]
            self.print_(f"Build {len(self.envs)} {self.conf.env.env_name} environments!")
        else:
            print("INITIALIZE ERROR")    # todo init other kinds of environments

    def init_model(self):
        """Initialize model instance."""
        if self.envs:
            state, _ = self.envs[0].reset()
            n_actions = self.envs[0].action_space.n
            n_observation = len(state)
        else: 
            # default env's size
            n_actions = 2
            n_observation = 4
            
        model = load_model(self.conf.model)
        self.policy_net = model(n_observation, n_actions).to(self.conf.device)
        self.target_net = model(n_observation, n_actions).to(self.conf.device)


    def init_server(self):
        """Initialize a server instance."""
        self._server_class = DQNServer

        kwargs = {
            "is_remote": self.conf.is_remote,
            "local_port": self.conf.local_port
        }

        self.server = self._server_class(self.conf, **kwargs)
        self._server_class.policy_net = self.policy_net
        self._server_class.target_net = self.target_net

    def init_clients(self):
        """Initialize client instances, each represent a federated reinforcement learning clients."""
        if not self.registered_client:
            self._client_class = DQN_Client

        if self.conf.model == "dqn":
            self._client_class = DQN_Client

        # Enforce system heterogeneity of clients.
        # sleep_time = [0 for _ in self.train_data.users]
        # if self.conf.resource_heterogeneous.simulate:
        #     sleep_time = resource_hetero_simulation(self.conf.resource_heterogeneous.fraction,
        #                                             self.conf.resource_heterogeneous.hetero_type,
        #                                             self.conf.resource_heterogeneous.sleep_group_num,
        #                                             self.conf.resource_heterogeneous.level,
        #                                             self.conf.resource_heterogeneous.total_time,
        #                                             len(self.train_data.users))

        # client_test_data = self.test_data
        # if self.conf.test_mode == TEST_IN_SERVER:
        #     client_test_data = None

        # all clients have same environments
        self.clients = [self._client_class(u, self.conf.client, None, None, self.conf.device, self.envs[u]) for u in range(self.conf.env.num_of_clients)]

        self.print_("Clients in total: {}".format(len(self.clients)))

    def init_client(self):
        """Initialize client instance.

        Returns:
            :obj:`BaseClient`: The initialized client instance.
        """
        # Get a random client if not specified
        if self.conf.index is not None:
            user = self.conf.index
        else:
            print("Please initialize the client index in conf.index")

        self._client_class = DQN_Client(
            user,
            self.conf.client,
            None,
            None,
            self.conf.device,
            is_remote=self.conf.is_remote,
            local_port=self.conf.local_port,
            server_addr=self.conf.server_addr,
            tracker_addr=self.conf.tracker_addr, 
            env=self.envs[user] if self.envs else None
        )
        return self._client_class

    def start_server(self, args):
        """Start a server service for remote training.

        Server controls the model and testing dataset if configured to test in server.

        Args:
            args (argparse.Namespace): Configurations passed in as arguments, it is merged with configurations.
        """
        if args:
            self.conf = OmegaConf.merge(self.conf, args.__dict__)

        self.init_model()

        self.init_server()

        self.server.start_service()

    def start_client(self, args):
        """Start a client service for remote training.

        Client controls training datasets.

        Args:
            args (argparse.Namespace): Configurations passed in as arguments, it is merged with configurations.
        """

        if args:
            self.conf = OmegaConf.merge(self.conf, args.__dict__)

        # self.init_dataset()
        # init environment

        client = self.init_client()

        client.start_service()

    def register_dataset(self, train_data, test_data, val_data=None):
        """Register datasets.

        Datasets should inherit from :obj:`FederatedDataset`, e.g., :obj:`FederatedTensorDataset`.

        Args:
            train_data (:obj:`FederatedDataset`): Training dataset.
            test_data (:obj:`FederatedDataset`): Testing dataset.
            val_data (:obj:`FederatedDataset`): Validation dataset.
        """
        self.registered_dataset = True
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

    def register_model(self, model):
        """Register customized model for federated learning.

        Args:
            model (nn.Module): PyTorch model, both class and instance are acceptable.
                Use model class when there is no specific arguments to initialize model.
        """
        self.registered_model = True
        if not isinstance(model, type):
            self.model = model
        else:
            self._model_class = model

    def register_server(self, server):
        """Register a customized federated learning server.

        Args:
            server (:obj:`BaseServer`): Customized federated learning server.
        """
        self.registered_server = True
        self._server_class = server

    def register_client(self, client):
        """Register a customized federated learning client.

        Args:
            client (:obj:`BaseClient`): Customized federated learning client.
        """
        self.registered_client = True
        self._client_class = client

    def print_(self, content):
        """Log the content only when the server is primary server.

        Args:
            content (str): The content to log.
        """
        if self._is_primary_server():
            logger.info(content)

    def _is_primary_server(self):
        """Check whether current running server is the primary server.

        In standalone or remote training, the server is primary.
        In distributed training, the server on `rank0` is primary.
        """
        return not self.conf.is_distributed or self.conf.distributed.rank == 0


def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# Initialize the global coordinator object
_global_coord_rl = Coordinator_RL()


def init_conf(conf=None):
    """Initialize configuration for EasyFL. It overrides and supplements default configuration loaded from config.yaml
    with the provided configurations.

    Args:
        conf (dict): Configurations.

    Returns:
        omegaconf.dictconfig.DictConfig: Internal configurations managed by OmegaConf.
    """
    here = path.abspath(path.dirname(__file__))
    config_file = path.join(here, 'config.yaml')
    return load_config(config_file, conf)


def load_config(file, conf=None):
    """Load and merge configuration from file and input

    Args:
        file (str): filename of the configuration.
        conf (dict): Configurations.

    Returns:
        omegaconf.dictconfig.DictConfig: Internal configurations managed by OmegaConf.
    """
    config = OmegaConf.load(file)
    if conf is not None:
        config = OmegaConf.merge(config, conf)
    return config


def init_logger(log_level):
    """Initialize internal logger of EasyFL.

    Args:
        log_level (int): Logger level, e.g., logging.INFO, logging.DEBUG
    """
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    log_level = logging.INFO if not log_level else log_level
    root_logger.setLevel(log_level)

    file_path = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = path.join(file_path, "train" + time.strftime(".%m_%d_%H_%M_%S") + ".log")
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)


def init_rl(conf=None, init_all=True):
    """Initialize EasyFL.

    Args:
        conf (dict, optional): Configurations.
        init_all (bool, optional): Whether initialize dataset, model, server, and client other than configuration.
    """
    global _global_coord_rl

    config = init_conf(conf)

    init_logger(config.tracking.log_level)

    _set_random_seed(config.seed)

    _global_coord_rl.init(config, init_all)


def run_rl():
    """Run federated learning process."""
    global _global_coord_rl
    _global_coord_rl.run()


def init_dataset():
    """Initialize dataset, either using registered dataset or out-of-the-box datasets set in config."""
    global _global_coord
    _global_coord.init_dataset()


def init_model():
    """Initialize model, either using registered model or out-of–the-box model set in config.

    Returns:
        nn.Module: Model used in federated learning.
    """
    global _global_coord_rl
    _global_coord_rl.init_model()

    return _global_coord_rl.policy_net


def start_server(args=None):
    """Start federated learning server service for remote training.

    Args:
        args (argparse.Namespace): Configurations passed in as arguments.
    """
    global _global_coord_rl

    _global_coord_rl.start_server(args)


def start_client(args=None):
    """Start federated learning client service for remote training.

    Args:
        args (argparse.Namespace): Configurations passed in as arguments.
    """
    global _global_coord_rl

    _global_coord_rl.start_client(args)


def get_coordinator():
    """Get the global coordinator instance.

    Returns:
        :obj:`Coordinator`: global coordinator instance.
    """
    return _global_coord


def register_dataset(train_data, test_data, val_data=None):
    """Register datasets for federated learning training.

    Args:
        train_data (:obj:`FederatedDataset`): Training dataset.
        test_data (:obj:`FederatedDataset`): Testing dataset.
        val_data (:obj:`FederatedDataset`): Validation dataset.
    """
    global _global_coord
    _global_coord.register_dataset(train_data, test_data, val_data)


def register_model(model):
    """Register model for federated learning training.

    Args:
        model (nn.Module): PyTorch model, both class and instance are acceptable.
    """
    global _global_coord
    _global_coord.register_model(model)


def register_server(server):
    """Register federated learning server.

    Args:
        server (:obj:`BaseServer`): Customized federated learning server.
    """
    global _global_coord
    _global_coord.register_server(server)


def register_client(client):
    """Register federated learning client.

    Args:
        client (:obj:`BaseClient`): Customized federated learning client.
    """
    global _global_coord
    _global_coord.register_client(client)
