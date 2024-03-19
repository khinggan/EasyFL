from easyfl.communication import grpc_wrapper
from easyfl.pb import common_pb2 as common_pb
from easyfl.pb import server_service_pb2 as server_pb

server_addr = "localhost:22999"
stub = grpc_wrapper.init_stub(grpc_wrapper.TYPE_SERVER, server_addr)
# Send request to stop training.
response = stub.Stop(server_pb.StopRequest())
result = "Success" if response.status.code == common_pb.SC_OK else response
print(result)