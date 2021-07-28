import socket
import subprocess
import time
from bagua.torch_api.env import get_rank, get_local_rank, get_world_size, get_local_size
from rediscluster import RedisCluster
from redis import Redis
from typing import List, Dict, Optional
from .store import Store
import torch.distributed.distributed_c10d as c10d
import torch
import pickle
import logging
import redis


class RedisStore(Store):
    def __init__(self, bootstrap=True, hosts: List[Dict[str, str]] = None):
        if not bootstrap and (hosts is None or len(hosts) == 0):
            raise ValueError("Must provide `hosts` when bootstrap is `False`")

        if bootstrap:
            if hosts is not None and len(hosts) > 0:
                logging.warn("Ignore input `hosts` when bootstrap is `True`")
            hosts = []

        self.cluster_mode = True
        self.hosts = hosts

        if bootstrap:
            self._start_redis_cluster()

        if self.cluster_mode:
            self.client = RedisCluster(startup_nodes=hosts, decode_responses=True)
        else:
            self.client = Redis(host=self.hosts[0]["host"], port=self.hosts[0]["port"])

        assert self.client.ping()

    def set(self, key: str, value: str):
        self.client.set(key, value)

    def get(self, key: str) -> Optional[str]:
        return self.client.get(key)

    def num_keys(self) -> int:
        return sum(self.client.dbsize().values())

    def clear(self):
        self.client.flushdb()

    def mset(self, mapping: Dict[str, str]):
        self.client.mset(mapping)

    def mget(self, keys: List[str]) -> List[Optional[str]]:
        return self.client.mget(keys)

    def status(self) -> bool:
        return self.client.ping()

    def _start_redis_cluster(self):
        nrank = get_rank() // get_local_size()
        nnodes = get_world_size() // get_local_size()

        ip, port = get_host_ip(), find_free_port()
        if not torch.distributed.is_initialized() or nnodes == 1:
            start_redis_server_cli(port, False)
            self.hosts.append({"host": "127.0.0.1", "port": port})
            self.cluster_mode = False
            return

        default_store = c10d._get_default_store()

        key_pattern = "redis-node{}"
        if get_local_rank() == 0:
            start_redis_server_cli(port, True)
            content = {"host": ip, "port": port}
            default_store.set(key_pattern.format(nrank), pickle.dumps(content))

        for i in range(nnodes):
            ret = default_store.get(key_pattern.format(i))
            print(ret)
            self.hosts.append(ret)

        create_redis_cluster_cli(self.hosts)


def create_redis_cluster_cli(hosts: List[Dict[str, str]]):
    cmd = ["redis-cli", "--cluster", "create"]

    for h in hosts:
        cmd.append("{}:{}".format(h["host"], h["port"]))

    logging.debug(f"create redis cluster, command: {cmd}")

    subprocess.run(cmd, capture_output=True, text=True, input="yes")
    time.sleep(5)


def start_redis_server_cli(port, cluster_mode, *args):
    cmd = ["redis-server", "--daemonize yes", "--port {}".format(port)]

    if cluster_mode:
        cluster_config = [
            "--cluster-enabled yes",
            "--cluster-config-file nodes.conf",
            "--cluster-node-timeout 5000",
            "--appendonly yes",
        ]
        cmd.extend(cluster_config)

    cmd.extend(list(args))
    logging.debug(f"start redis server, command: {cmd}")
    subprocess.run(cmd)
    time.sleep(10)


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("localhost", 0))
    sockname = sock.getsockname()
    sock.close()
    return sockname[1]


def get_host_ip():
    try:
        host_name = socket.gethostname()
        return socket.gethostbyname(host_name)
    except:
        print("Unable to get host IP")
