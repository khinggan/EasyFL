import easyfl

# Configurations for the remote client.
conf = {
    "is_remote": True,
    "local_port": 23001,
    "server_addr": "localhost:22999",
    "index": 1,
}
# Initialize only the configuration.
easyfl.init(conf, init_all=False)
# Start remote client service.
# The remote client waits to be connected with the remote server.
easyfl.start_client()