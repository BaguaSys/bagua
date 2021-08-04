import socket
import subprocess
import time
from bagua.torch_api.env import get_rank, get_local_rank, get_world_size, get_local_size
from redis import Redis
from typing import List, Dict, Optional
from .store import Store, ClusterStore
import torch.distributed.distributed_c10d as c10d
import json
import logging


_host_ip = None


class RedisStore(ClusterStore):
    def __init__(
        self,
        hosts: List[Dict[str, str]] = None,
        cluster_mode: bool = False,
        capacity_per_node: int = 100_000_000_000,
    ):
        """ """

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
            self._bootstrap_redis_server()

        stores = []
        for h in self.hosts:
            store = _RedisStore(
                host=h["host"], port=h["port"], bootstrap=self.bootstrap
            )
            stores.append(store)

        super(RedisStore, self).__init__(stores)

    def _bootstrap_redis_server(self):
        ip, port = get_host_ip(), find_free_port()
        hostinfo = {"host": ip, "port": port}
        if get_local_rank() == 0:
            start_redis_server_cli(port, self.capacity_per_node)

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

        if self.cluster_mode:
            self.hosts.extend(hosts)
        else:
            self.hosts.append(hosts[nrank])


class _RedisStore(Store):
    def __init__(self, host, port, bootstrap):
        self.client = create_redis_client(host=host, port=port)
        self.bootstrap = bootstrap

        assert self._connect_with_retry(
            retry_times=3
        ), "Could not connect to redis server {}:{}".format(host, port)

    def _connect_with_retry(self, retry_times=3):
        for i in range(retry_times):
            try:
                connected = self.client.ping()
            except Exception as e:
                if i == retry_times - 1:
                    return False

                time.sleep(10)
            else:
                return connected

        return False

    def set(self, key: str, value: str):
        self.client.set(key, value)

    def get(self, key: str) -> Optional[str]:
        return self.client.get(key)

    def num_keys(self) -> int:
        return self.client.dbsize()

    def clear(self) -> bool:
        self.client.flushdb()

    def mset(self, mapping: Dict[str, str]):
        self.client.mset(mapping)

    def mget(self, keys: List[str]) -> List[Optional[str]]:
        return self.client.mget(keys)

    def status(self) -> bool:
        return self.client.ping()

    def shutdown(self):
        if self.bootstrap:
            self.client.shutdown(nosave=True)


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
        "--protected-mode no"
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
