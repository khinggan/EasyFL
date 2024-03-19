from omegaconf import OmegaConf
from easyfl.communication import grpc_wrapper
# from easyfl.pb.server_service_pb2_grpc import ServerService
from easyfl.server.service import ServerService
from easyfl.protocol import codec
from easyfl.tracking import metric
import copy
import torch
import time
import logging
import concurrent.futures
from easyfl.server import BaseServer
logger = logging.getLogger(__name__)
from easyfl.pb import common_pb2 as common_pb
from easyfl.pb import client_service_pb2 as client_pb


class DQNServer(BaseServer):
    def __init__(self, conf, test_data=None, val_data=None, is_remote=False, local_port=22999):
        super(DQNServer, self).__init__(conf, test_data, val_data, is_remote, local_port)
        self.policy_net = None
        self.target_net = None
        self.compress_policy_net = None
        self.compress_target_net = None

        self.dqn_client_stubs = {}
        
    def distribution_to_train_locally(self):
        """Conduct training sequentially for selected clients in the group."""
        # uploaded_models = {}
        uploaded_weights = {}
        # uploaded_metrics = []
        for client in self.grouped_clients:
            # Update client config before training
            # self.conf.client.task_id = self.conf.task_id
            # self.conf.client.round_id = self._current_round

            weight = client.run_train(self.compress_policy_net, self.conf.client)

            weight = self.decompression(codec.unmarshal(weight))
            uploaded_weights[client.cid] = weight

        self.set_client_uploads_train(uploaded_weights)
    
    def set_client_uploads_train(self, uploaded_weights):
        self._client_uploads['weights'] = uploaded_weights
    
    def aggregation(self):
        """Aggregate training updates from clients.
        Server aggregates trained models from clients via federated averaging.
        """
        uploaded_weights = self._client_uploads["weights"]

        agg_policy_net_weight = self.average_weights(uploaded_weights)
        
        # set model
        self.policy_net.load_state_dict(agg_policy_net_weight)
        self.target_net.load_state_dict(agg_policy_net_weight)
    
    def average_weights(self, w):
        """
        Returns the average of the weights.
        """
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg

    def train(self):
        """Training process of federated learning."""
        self.print_("--- start training ---")

        self.selection(self._clients, self.conf.server.clients_per_round)
        self.grouping_for_distributed()
        self.compression()

        begin_train_time = time.time()
        if self.is_remote: 
            self.distribution_to_train_remotely()
        else:
            self.distribution_to_train_locally()
        self.aggregation()

        train_time = time.time() - begin_train_time
        self.print_("Server train time: {}".format(train_time))
        # self.track(metric.TRAIN_TIME, train_time)
    
    def compression(self):
        """Model compression to reduce communication cost."""
        self.compress_policy_net = self.policy_net
        self.compress_target_net = self.target_net
    
    def start_service(self):
        """Start federated DQN server GRPC service."""
        if self.is_remote:
            grpc_wrapper.start_service(grpc_wrapper.TYPE_SERVER, ServerService(self), self.local_port)
            logger.info("GRPC server started at :{}".format(self.local_port))
    
    def start_remote_training(self, model, clients):
        """Start federated learning in the remote training mode.
        Server establishes gPRC connection with clients that are not connected first before training.

        Args:
            model (nn.Module): The model to train.
            clients (list[str]): Client addresses.
        """
        self.connect_remote_clients(clients)
        self.start(model, clients)

    def connect_remote_clients(self, clients):
        # TODO: This client should be consistent with client started separately.
        for client in clients:
            if client.client_id not in self.dqn_client_stubs:
                self.dqn_client_stubs[client.client_id] = grpc_wrapper.init_stub(grpc_wrapper.TYPE_CLIENT, client.address)
                logger.info("Successfully connected to gRPC client {}".format(client.address))


    def distribution_to_train_remotely(self):
        """Distribute training requests to remote clients through multiple threads.
        The main thread waits for signal to proceed. The signal can be triggered via notification, as below example.

        Example to trigger signal:
            >>> with self.condition():
            >>>     self.notify_all()
        """
        start_time = time.time()
        # should_track = self._tracker is not None and self.conf.client.track
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for client in self.grouped_clients:
                request = client_pb.OperateRequest(
                    type=client_pb.OP_TYPE_TRAIN,
                    model=codec.marshal(self.policy_net),
                    data_index=client.index,
                    config=client_pb.OperateConfig(
                        batch_size=self.conf.client.batch_size,
                        local_epoch=self.conf.client.local_epoch,
                        seed=self.conf.seed,
                        local_test=None,
                        optimizer=client_pb.Optimizer(
                            type=self.conf.client.optimizer.type,
                            lr=self.conf.client.optimizer.lr,
                            momentum=self.conf.client.optimizer.momentum,
                        ),
                        task_id=self.conf.task_id,
                        round_id=self._current_round,
                        track=None,
                    ),
                )
                executor.submit(self._distribution_remotely, client.client_id, request)

            distribute_time = time.time() - start_time
            self.track(metric.TRAIN_DISTRIBUTE_TIME, distribute_time)
            logger.info("Distribute to clients, time: {}".format(distribute_time))
        with self._condition:
            self._condition.wait()
    
    def _distribution_remotely(self, cid, request):
        """Distribute request to the assigned client to conduct operations.

        Args:
            cid (str): Client id.
            request (:obj:`OperateRequest`): gRPC request of specific operations.
        """
        resp = self.client_stubs[cid].Operate(request)
        if resp.status.code != common_pb.SC_OK:
            logger.error("Failed to train/test in client {}, error: {}".format(cid, resp.status.message))
        else:
            logger.info("Distribute to train/test remotely successfully, client: {}".format(cid))


    def start(self, model, clients):
        """Start federated learning process, including training and testing.

        Args:
            model (nn.Module): The model to train.
            clients (list[:obj:`BaseClient`]|list[str]): Available clients.
                Clients are actually client grpc addresses when in remote training.
        """
        # Setup
        self._start_time = time.time()
        self._reset()
        self.set_model(model)
        self.set_clients(clients)

        if self._should_track():
            self._tracker.create_task(self.conf.task_id, OmegaConf.to_container(self.conf))

        while not self.should_stop():
            self._round_time = time.time()

            self._current_round += 1
            self.print_("\n-------- round {} --------".format(self._current_round))

            # Train
            self.pre_train()
            self.train()
            self.post_train()

            # Test
            # if self._do_every(self.conf.server.test_every, self._current_round, self.conf.server.rounds):
            #     self.pre_test()
            #     self.test()
            #     self.post_test()

            # # Save Model
            # self.save_model()

