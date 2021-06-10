import time
import json
import logging
import requests
import socket
import threading
from flask import request, Flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app
from werkzeug.serving import make_server


def wait_for_port(port, host="localhost", timeout=5.0):
    """Wait until a port starts accepting TCP connections.
    Args:
        port (int): Port number.
        host (str): Host address on which the port should exist.
        timeout (float): In seconds. How long to wait before raising errors.
    Raises:
        TimeoutError: The port isn't accepting connection after time specified in `timeout`.
    """
    start_time = time.perf_counter()
    while True:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                break
        except OSError as ex:
            time.sleep(0.01)
            if time.perf_counter() - start_time >= timeout:
                raise TimeoutError(
                    "Waited too long for the port {} on host {} to start accepting "
                    "connections.".format(port, host)
                ) from ex


class ServerThread(threading.Thread):
    def __init__(self, master_addr, master_port, app):
        threading.Thread.__init__(self)
        self.srv = make_server(master_addr, master_port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.srv.serve_forever()

    def shutdown(self):
        self.srv.shutdown()


def pick_n_free_ports(n: int):
    socks = []
    for i in range(n):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("localhost", 0))
        socks.append(sock)

    n_free_ports = [sock.getsockname()[1] for sock in socks]
    for sock in socks:
        sock.close()

    return n_free_ports


def setup_app(slots: list, server_addr: str, server_port: int):
    # run http server
    app = Flask(__name__)
    app.use_reloader = False  # type: ignore

    @app.route("/get_fuselib_server_addr", methods=["POST"])
    def get_fuselib_server_addr():
        req: dict = request.get_json(force=True)
        hostname: str = req["hostname"]
        rank: int = req["rank"]

        slots[rank] = hostname

        return json.dumps({"server_addr": server_addr, "server_port": server_port})

    # Add prometheus wsgi middleware to route /metrics requests
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/metrics": make_wsgi_app()})  # type: ignore

    return app


def generate_and_broadcast_server_addr(
    master_addr,
    master_port,
    world_size: int,
    my_rank: int,
    proxies={
        "http": None,
        "https": None,
    },
):
    # import torch
    # torch.distributed.init_process_group(backend="gloo", init_method="tcp://{}:{}".format(master_addr, master_port), world_size=world_size, rank=my_rank)
    # if my_rank == 0:
    #     objects = [pick_n_free_ports(1)[0]]
    # else:
    #     objects = [None]
    # dist.broadcast_object_list(objects, src=0)
    # dist.destroy_process_group()

    # server_port = broadcast_objects[0]

    # return (master_addr, server_port)

    slots = [None] * world_size
    if my_rank == 0:
        server_addr = master_addr
        server_port = pick_n_free_ports(1)[0]
        slots[my_rank] = socket.gethostname()  # type: ignore

        app = setup_app(slots, server_addr, server_port)
        server = ServerThread("0.0.0.0", master_port, app)
        server.start()

        while True:
            n_empty_slot = len([x for x in slots if x is None])
            if n_empty_slot == 0:
                break

            time.sleep(1)

        server.shutdown()
    else:
        # Wait service discovery service ready
        timeout = time.time() + 60  # 60s timeout
        rsp = None
        wait_for_port(host=master_addr, port=master_port)
        while time.time() < timeout:
            try:
                with requests.session() as sess:
                    rsp = sess.post(
                        "http://{}:{}/get_fuselib_server_addr".format(
                            master_addr, master_port
                        ),
                        json={"rank": my_rank, "hostname": socket.gethostname()},
                        proxies=proxies,
                        timeout=timeout,
                    )
            except requests.exceptions.ConnectionError as ex:
                logging.info(ex)

            if rsp and rsp.status_code == 200:
                break
            time.sleep(1)
        if rsp is None or rsp.status_code != 200:
            raise RuntimeError("Waiting for service discovery service start timeout")

        server_addr = json.loads(rsp.content)["server_addr"]
        server_port = json.loads(rsp.content)["server_port"]

    return (master_addr, server_port, slots)
