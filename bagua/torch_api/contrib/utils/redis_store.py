import socket
import subprocess
import time
from bagua.torch_api.env import get_rank, get_local_rank, get_world_size, get_local_size

# from rediscluster import RedisCluster
from redis import Redis
from typing import List, Dict, Optional
from .store import Store
import torch.distributed.distributed_c10d as c10d
import torch
import json
import logging

_host_ip = None


class RedisStore(Store):
    def __init__(
        self,
        hosts: List[Dict[str, str]] = None,
        cluster_mode: bool = False,
        capacity_per_node: int = 1_000_000_000,
    ):
        """ """

        self.hosts = []
        if hosts is None:
            logging.info("Ready to bootstrap redis server locally")
            self.bootstrap = True
        else:
            logging.info("Ready to connect redis servers: {}".format(hosts))
            self.bootstrap = False
            self.hosts.extends(hosts)

        self.cluster_mode = cluster_mode
        self.capacity_per_node = capacity_per_node

        if self.bootstrap:
            self._bootstrap_redis_server()

        if self.cluster_mode:
            raise ValueError("RedisStore does not support cluster mode at present")
            # self.client = RedisCluster(startup_nodes=self.hosts, decode_responses=True)
        else:
            self.client = create_redis_client(
                host=self.hosts[0]["host"], port=self.hosts[0]["port"]
            )

        assert self.client.ping()

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

    def _bootstrap_redis_server(self):
        ip, port = get_host_ip(), find_free_port()
        hostinfo = {"host": ip, "port": port}
        if get_local_rank() == 0:
            start_redis_server_cli(port, self.cluster_mode, self.capacity_per_node)
        else:
            wait_for_start_redis_server_cli()

        if get_world_size() > 1:
            nrank = get_rank() // get_local_size()
            nnodes = get_world_size() // get_local_size()
            default_store = c10d._get_default_store()
            key_pattern = "redis-node{}"

            if get_local_rank() == 0:
                default_store.set(key_pattern.format(nrank), json.dumps(hostinfo))

            for i in range(nnodes):
                ret = json.loads(default_store.get(key_pattern.format(i)))
                self.hosts.append(ret)
        else:
            self.hosts.append(hostinfo)

        if not self.cluster_mode:
            return

        # create_redis_cluster_cli(self.hosts)
        # wait_for_create_redis_cluster_cli()


def create_redis_cluster_cli(hosts: List[Dict[str, str]]):
    cmd = ["redis-cli", "--cluster", "create"]

    for h in hosts:
        cmd.append("{}:{}".format(h["host"], h["port"]))

    logging.debug(f"create redis cluster, command: {cmd}")

    subprocess.run(cmd, capture_output=True, text=True, input="yes")
    time.sleep(5)


def wait_for_create_redis_cluster_cli():
    time.sleep(5)


def start_redis_server_cli(port, cluster_mode, capacity, *args):
    cmd = [
        "redis-server",
        "--daemonize yes",
        "--port {}".format(port),
        "--maxmemory {}".format(capacity),
        "--maxmemory-policy allkeys-random",  # use random eviction by default
        "--appendonly no",  # disable persistence by default
        '--save ""',
    ]

    if cluster_mode:
        cluster_config = [
            "--cluster-enabled yes",
            "--cluster-config-file nodes.conf",
            "--cluster-node-timeout 5000",
        ]
        cmd.extend(cluster_config)

    cmd.extend(list(args))
    logging.debug(f"start redis server, command: {cmd}")
    subprocess.run(cmd)
    time.sleep(10)


def wait_for_start_redis_server_cli():
    time.sleep(10)


def create_redis_client(host, port):
    return Redis(port=port) if host == get_host_ip() else Redis(host=host, port=port)


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
