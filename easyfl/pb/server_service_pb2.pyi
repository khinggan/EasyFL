from easyfl.pb import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class UploadRequest(_message.Message):
    __slots__ = ("task_id", "round_id", "client_id", "content")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    ROUND_ID_FIELD_NUMBER: _ClassVar[int]
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    round_id: int
    client_id: int
    content: UploadContent
    def __init__(self, task_id: _Optional[str] = ..., round_id: _Optional[int] = ..., client_id: _Optional[int] = ..., content: _Optional[_Union[UploadContent, _Mapping]] = ...) -> None: ...

class UploadContent(_message.Message):
    __slots__ = ("data", "extra")
    DATA_FIELD_NUMBER: _ClassVar[int]
    EXTRA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    extra: bytes
    def __init__(self, data: _Optional[bytes] = ..., extra: _Optional[bytes] = ...) -> None: ...

class Performance(_message.Message):
    __slots__ = ("accuracy", "loss")
    ACCURACY_FIELD_NUMBER: _ClassVar[int]
    LOSS_FIELD_NUMBER: _ClassVar[int]
    accuracy: float
    loss: float
    def __init__(self, accuracy: _Optional[float] = ..., loss: _Optional[float] = ...) -> None: ...

class UploadResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ...) -> None: ...

class RunRequest(_message.Message):
    __slots__ = ("model", "clients", "etcd_addresses")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    CLIENTS_FIELD_NUMBER: _ClassVar[int]
    ETCD_ADDRESSES_FIELD_NUMBER: _ClassVar[int]
    model: bytes
    clients: _containers.RepeatedCompositeFieldContainer[Client]
    etcd_addresses: str
    def __init__(self, model: _Optional[bytes] = ..., clients: _Optional[_Iterable[_Union[Client, _Mapping]]] = ..., etcd_addresses: _Optional[str] = ...) -> None: ...

class RunResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ...) -> None: ...

class StopRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StopResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ...) -> None: ...

class Client(_message.Message):
    __slots__ = ("client_id", "address", "index")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    address: str
    index: int
    def __init__(self, client_id: _Optional[str] = ..., address: _Optional[str] = ..., index: _Optional[int] = ...) -> None: ...
