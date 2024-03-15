from easyfl.protocol import codec
from easyfl.tracking import metric
import copy
import torch
import time
from easyfl.server import BaseServer

class DQNServer(BaseServer):
    def __init__(self, conf, test_data=None, val_data=None, is_remote=False, local_port=22999):
        super(DQNServer, self).__init__(conf, test_data, val_data, is_remote, local_port)
        self.policy_net = None
        self.target_net = None
        self.compress_policy_net = None
        self.compress_target_net = None
        
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

        self.distribution_to_train_locally()
        self.aggregation()

        train_time = time.time() - begin_train_time
        self.print_("Server train time: {}".format(train_time))
        # self.track(metric.TRAIN_TIME, train_time)
    
    def compression(self):
        """Model compression to reduce communication cost."""
        self.compress_policy_net = self.policy_net
        self.compress_target_net = self.target_net