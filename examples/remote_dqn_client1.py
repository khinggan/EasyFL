from easyfl.coordinator_rl import init_rl, start_client

# Configurations for the remote client.
conf = {
    "is_remote": True,
    "local_port": 23000,
    "server_addr": "localhost:22999",
    "index": 0,
}
# Initialize only the configuration.
init_rl(conf, init_all=False)
# Start remote client service.
# The remote client waits to be connected with the remote server.
start_client()