import os
import socket


def get_world_size() -> int:
    """
    Get the number of processes in the default process group.

    Returns:
        The world size of the default process group.
    """
    return int(os.environ.get("WORLD_SIZE", 1))


def get_rank() -> int:
    """
    Get the rank of the default process group.

    Rank is a unique identifier assigned to each process within the default
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Returns:
        The rank of the default process group.
    """
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    """
    Get the rank of current node.

    Local rank is a unique identifier assigned to each process within a node.
    They are always consecutive integers ranging from 0 to ``local_size``.

    Returns:
        The local rank of the node.
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def get_local_size() -> int:
    """
    Get the number of processes in the node.

    Returns:
        The local size of the node.
    """
    return int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def get_node_rank() -> int:
    """
    Get the rank among all nodes.

    Returns:
        The node rank of the node.
    """
    if _is_elastic_launched():
        return int(os.environ.get("GROUP_RANK", 0))
    else:
        return int(os.environ.get("NODE_RANK", 0))


def _is_elastic_launched():
    """Returns ``True`` if the current process was launched using the bagua.distributed.run command."""
    required_env_vars = {"RANK", "GROUP_RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE"}
    return required_env_vars.issubset(os.environ.keys())


def get_default_bucket_size() -> int:
    """Get default communication bucket byte size.

    Returns:
        The default bucket size.
    """
    return int(os.environ.get("BAGUA_DEFAULT_BUCKET_SIZE", 10 * 1024 ** 2))


def get_master_addr() -> str:
    return os.environ.get("MASTER_ADDR", "127.0.0.1")


def get_bagua_service_port() -> int:
    return int(os.environ.get("BAGUA_SERVICE_PORT", -1))


def is_report_metrics_switch_on() -> bool:
    """
    Whether bagua report switch is on or not.
    """
    return int(os.environ.get("BAGUA_REPORT_METRICS", 0)) == 1


# ** Autotune Environment Variable **


def get_autotune_level() -> int:
    """Get the autotune level.

    Returns:
        The autotune level.
    """
    return int(os.environ.get("BAGUA_AUTOTUNE", 0))


def get_autotune_max_samples() -> int:
    return int(os.environ.get("BAGUA_AUTOTUNE_MAX_SAMPLES", 60))


def get_autotune_sampling_confidence_time_s() -> float:
    return float(os.environ.get("BAGUA_AUTOTUNE_SAMPLING_CONFIDENCE_TIME_S", 5.0))


def get_autotune_warmup_time_s() -> float:
    return float(os.environ.get("BAGUA_AUTOTUNE_WARMUP_TIME_S", 30.0))


def get_is_output_autotune_log() -> bool:
    return bool(os.environ.get("BAGUA_IS_OUTPUT_AUTOTUNE_LOG", 0))


def get_autotune_server_wait_time() -> int:
    return int(os.environ.get("BAGUA_AUTOTUNE_SERVER_WAIT_TIME", 300))


def find_free_network_port() -> int:
    """Finds a free port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port
