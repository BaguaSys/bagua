import os


def get_world_size():
    """
    Get the number of processes in the current process group.

    Returns:
        The world size of the process group.
    """
    return int(os.environ.get("WORLD_SIZE", 1))


def get_rank():
    """
    Get the rank of current process group.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Returns:
        The rank of the process group.
    """
    return int(os.environ.get("RANK", 0))


def get_local_rank():
    """
    Get the rank of current node.

    Local rank is a unique identifier assigned to each process within a node.
    They are always consecutive integers ranging from 0 to ``local_size``.

    Returns:
        The local rank of the node.
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def get_local_size():
    """
    Get the number of processes in the node.

    Returns:
        The local size of the node.
    """
    return int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def get_default_bucket_size() -> int:
    """Get default communication bucket byte size.

    Returns:
        int: default bucket size
    """
    return int(os.environ.get("BAGUA_DEFAULT_BUCKET_SIZE", 10 * 1024 ** 2))


def get_master_addr():
    return os.environ.get("MASTER_ADDR", "127.0.0.1")


def get_bagua_service_port():
    return int(os.environ.get("BAGUA_SERVICE_PORT", -1))


def is_report_metrics_switch_on():
    """
    Whether bagua report switch is on or not.
    """
    return int(os.environ.get("BAGUA_REPORT_METRICS", 0)) == 1


# ** Autotune Environment Variable **


def get_autotune_level() -> int:
    """Get the autotune level.
    Returns:
        int: The autotune level.
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
