from easyfl.server import BaseServer

class RL_Server(BaseServer):
    def __init__(self, conf, test_data=None, val_data=None, is_remote=False, local_port=22999):
        super(RL_Server, self).__init__(conf, test_data, val_data, is_remote, local_port)
        
        def aggregation(self):
        # Implement customized aggregation method, which overwrites the default aggregation method.
            pass