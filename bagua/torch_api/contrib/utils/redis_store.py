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
    A Redis-based Key-Value store implementation.

    The server holds the data, while the client can connect to the server over Redis protocol and perform
    actions such as `set()` to insert a key-value pair, `get()` to retrieve a key-value pair, etc.

    Args:
        hosts (List[Dict[str, str]]): A list of redis servers, defined by a list of dict containing server host and
            port information. Can be ``None``, which means to bootstrap redis servers locally.
        cluster_mode (bool): Redis servers serve as a cluster or not. If True, data is automatically sharded across all
            redis servers, otherwise, data is routed to a specific server.
        capacity_per_node (int): Maximum memory limit in bytes to configure redis servers when bootstrap locally. Redis servers
            will evict keys randomly when maximum memory limit is reached.
        hash_fn: Hash function to compute the shard key. Default is `xxh64`. A `hash_fn` accepts a `str` as
            input, and returns an `int` as output.

    .. note::
        Only one redis server can be bootstrapped on each node, thus the maximum memory limit of it is determined on
        its first initialization.
    """

    def __init__(
        self,
        hosts: Optional[List[Dict[str, str]]] = None,
        cluster_mode: bool = True,
        capacity_per_node: int = 100_000_000_000,
        hash_fn=None,
    ):

        self.hosts = []
        if hosts is None:
            logging.info("Ready to bootstrap redis server locally")
            self.bootstrap = True
        else:
            logging.info("Ready to connect redis servers: {}".format(hosts))
            self.bootstrap = False
            self.hosts.extend(hosts)

        self.cluster_mode = cluster_mode
        self.capacity_per_node = capacity_per_node

        if self.bootstrap:
            bootstrap_redis_server(self.capacity_per_node)
            self.hosts.extend(get_bootstrapped_host_info(self.cluster_mode))

        stores = []
        for h in self.hosts:
            store = _RedisStore(
                host=h["host"], port=h["port"], bootstrap=self.bootstrap
            )
            stores.append(store)

        super(RedisStore, self).__init__(stores, hash_fn)


def _is_bootstrapped():
    global _global_redis_servers

    return _global_redis_servers is not None and len(_global_redis_servers) > 0


def shutdown_redis_server():
    global _global_redis_servers

    hostinfo = get_bootstrapped_host_info(cluster_mode=False)[0]
    store = _RedisStore(host=hostinfo["host"], port=hostinfo["port"], bootstrap=True)

    store.shutdown()


def bootstrap_redis_server(capacity_per_node):
    if _is_bootstrapped():
        logging.debug("local redis server has already bootstrapped")
        return

    ip, port = get_host_ip(), find_free_port()
    hostinfo = {"host": ip, "port": port}
    if get_local_rank() == 0:
        start_redis_server_cli(port, capacity_per_node)

    hosts = []
    nrank = get_rank() // get_local_size()
    if get_world_size() > 1:
        nnodes = get_world_size() // get_local_size()
        default_store = c10d._get_default_store()
        key_pattern = "redis-node{}"

        if get_local_rank() == 0:
            default_store.set(key_pattern.format(nrank), json.dumps(hostinfo))

        for i in range(nnodes):
            ret = json.loads(default_store.get(key_pattern.format(i)))
            hosts.append(ret)
    else:
        hosts.append(hostinfo)

    global _global_redis_servers
    _global_redis_servers.extend(hosts)

    atexit.register(shutdown_redis_server)


def get_bootstrapped_host_info(cluster_mode):
    global _global_redis_servers

    if cluster_mode:
        return _global_redis_servers
    else:
        nrank = get_rank() // get_local_size()
        return [_global_redis_servers[nrank]]


class _RedisStore(Store):
    def __init__(self, host, port, bootstrap):
        self.client = create_redis_client(host=host, port=port)
        self.host = host
        self.port = port
        self.bootstrap = bootstrap

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

    def mset(self, mapping: Dict[str, Union[str, bytes]]):
        self.client.mset(mapping)

    def mget(self, keys: List[str]) -> List[Optional[Union[str, bytes]]]:
        return self.client.mget(keys)

    def status(self) -> bool:
        return self.client.ping()

    def shutdown(self):
        if self.bootstrap:
            logging.debug(f"shutting down local redis server at port {self.port}")
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
    logging.debug(f"start redis server, command: {cmd}")
    subprocess.run(cmd)


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
