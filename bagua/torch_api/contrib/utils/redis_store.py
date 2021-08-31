import socket
import subprocess
import time
from bagua.torch_api.env import get_rank, get_local_rank, get_world_size, get_local_size

try:
    from redis import Redis
except ImportError:
    print(
        "DEBUG: did not find redis-py. To install it, run `pip install redis` or follow instructions on its website(https://github.com/andymccurdy/redis-py)."
    )
    raise

from typing import List, Dict, Optional, Union
from .store import Store, ClusterStore
import torch.distributed.distributed_c10d as c10d
import json
import logging
import atexit


try:
    p = subprocess.Popen(
        ["redis-server", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
except Exception:
    print(
        "DEBUG: did not find redis-server. Follow instructions on its website(https://redis.io/download) to have it installed."
    )
    raise


__all__ = ["RedisStore"]

_host_ip = None

_global_redis_servers = []


class RedisStore(ClusterStore):
    """
    A Redis-based distributed key-value store implementation, with :meth:`~bagua.torch_api.contrib.utils.store.Store.set`
    and :meth:`~bagua.torch_api.contrib.utils.store.Store.get` API exposed.

    Args:
        hosts (List[Dict[str, str]]): A list of redis servers, defined by a list of dict containing Redis host and
            port information like ``[{"host": "192.168.1.0", "port": "7000"}, {"host": "192.168.1.1", "port": "7000"}]``.
            A new Redis instance will be spawned on each node if :attr:`hosts=None`.
        cluster_mode (bool): If ``True``, data is sharded across all Redis instances. Otherwise, if there are :math:`m`
            Redis instances, the workers on the :math:`n`-th node will use the :math:`n % m`-th Redis instance.
        capacity_per_node (int): Maximum memory limit in bytes when spawning new Redis instances. Old values will be evicted when the limit is reached.
            Default is ``100GB``.

    .. note::
        All Bagua jobs within the same node will share the same local Redis instance if :attr:`hosts=None`. The :attr:`capacity_per_node` only affects
        newly spawned Redis instances, and has no effect on existing ones.
    """

    def __init__(
        self,
        hosts: Optional[List[Dict[str, str]]] = None,
        cluster_mode: bool = True,
        capacity_per_node: int = 107_374_182_400,
    ):

        if hosts is None:
            logging.info("Ready to bootstrap redis server locally.")
            hosts = bootstrap_redis_server(capacity_per_node)
        else:
            assert len(hosts) > 0, "RedisStore hosts should not be empty."
            logging.info("Ready to connect redis servers: {}".format(hosts))

        to_connect = []
        if cluster_mode:
            to_connect.extend(hosts)
        else:
            to_connect.append(hosts[get_node_rank() % len(hosts)])

        stores = []
        for h in to_connect:
            store = _RedisStore(host=h["host"], port=h["port"])
            stores.append(store)

        super(RedisStore, self).__init__(stores)


def _is_bootstrapped():
    global _global_redis_servers

    return _global_redis_servers is not None and len(_global_redis_servers) > 0


def shutdown_redis_server():
    global _global_redis_servers

    hostinfo = _global_redis_servers[get_node_rank() % len(_global_redis_servers)]
    store = _RedisStore(host=hostinfo["host"], port=hostinfo["port"])

    store.shutdown()


def bootstrap_redis_server(capacity_per_node):
    global _global_redis_servers

    if _is_bootstrapped():
        logging.debug("Local redis server has already bootstrapped.")
        return _global_redis_servers

    host, port = get_host_ip(), find_free_port()
    hostinfo = {"host": host, "port": port}
    if get_local_rank() == 0:
        start_redis_server_cli(port, capacity_per_node)
        atexit.register(shutdown_redis_server)

    if get_world_size() > 1:
        default_store = c10d._get_default_store()
        key_pattern = "redis-node{}"

        if get_local_rank() == 0:
            default_store.set(key_pattern.format(get_node_rank()), json.dumps(hostinfo))

        for i in range(get_num_nodes()):
            ret = json.loads(default_store.get(key_pattern.format(i)))
            _global_redis_servers.append(ret)
    else:
        _global_redis_servers.append(hostinfo)

    return _global_redis_servers


class _RedisStore(Store):
    def __init__(self, host, port):
        self.client = create_redis_client(host=host, port=port)
        self.host = host
        self.port = port

        assert self._connect_with_retry(
            retry_times=3
        ), "Could not connect to redis server {}:{}".format(host, port)

    def _connect_with_retry(self, retry_times=3):
        for i in range(retry_times):
            try:
                connected = self.client.ping()
            except Exception:
                if i == retry_times - 1:
                    return False

                time.sleep(10)
            else:
                return connected

        return False

    def set(self, key: str, value: Union[str, bytes]):
        self.client.set(key, value)

    def get(self, key: str) -> Optional[Union[str, bytes]]:
        return self.client.get(key)

    def num_keys(self) -> int:
        return self.client.dbsize()

    def clear(self):
        self.client.flushdb()

    def mset(self, dictionary: Dict[str, Union[str, bytes]]):
        self.client.mset(dictionary)

    def mget(self, keys: List[str]) -> List[Optional[Union[str, bytes]]]:
        return self.client.mget(keys)

    def status(self) -> bool:
        return self.client.ping()

    def shutdown(self):
        if self.host != get_host_ip():
            logging.error("Could not shut down non-local redis servers.")
        else:
            logging.debug(
                f"CLEANUP: shutting down local redis server at port {self.port}."
            )
            self.client.shutdown(nosave=True)  # pytype: disable=wrong-keyword-args


def create_redis_client(host, port):
    logging.debug(f"{get_host_ip()} connect to redis server: {host}:{port}")
    return Redis(port=port) if host == get_host_ip() else Redis(host=host, port=port)


def start_redis_server_cli(port, capacity, *args):
    cmd = [
        "redis-server",
        "--daemonize yes",
        "--port {}".format(port),
        "--maxmemory {}".format(capacity),
        "--maxmemory-policy allkeys-random",  # use random eviction by default
        "--appendonly no",  # disable persistence by default
        '--save ""',
        "--protected-mode no",
    ]

    cmd.extend(list(args))
    logging.debug(f"Start redis server, command: {cmd}")
    subprocess.run(cmd)


def get_node_rank():
    return get_rank() // get_local_size()


def get_num_nodes():
    return get_world_size() // get_local_size()


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("localhost", 0))
    sockname = sock.getsockname()
    sock.close()
    return sockname[1]


def get_host_ip():
    global _host_ip

    if _host_ip is None:
        host_name = socket.gethostname()
        _host_ip = socket.gethostbyname(host_name)

    return _host_ip
