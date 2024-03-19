from easyfl.coordinator_rl import init_rl, start_server

# Configurations for the remote server.
conf = {"is_remote": True, "local_port": 22999, "model": "dqn"}
# Initialize only the configuration.
init_rl(conf, init_all=False)
# Start remote server service.
# The remote server waits to be connected with the remote client.
start_server()
