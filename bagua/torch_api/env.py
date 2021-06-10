import os


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def get_rank():
    return int(os.environ.get("RANK", 0))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def get_local_size():
    return int(os.environ.get("LOCAL_SIZE", 1))


def get_autotune_server_addr():
    return os.environ.get("AUTO_TUNE_SERVER_ADDR")


def is_report_metrics_switch_on():
    return int(os.environ.get("BAGUA_REPORT_METRICS", 0)) == 1


def get_autotune_level():
    return int(os.environ.get("BAGUA_AUTOTUNE", 0))


def _horovod_0_21_1_compat_mode():
    return int(os.environ.get("BAGUA_HOROVOD_0_21_1_COMPAT_MODE", 0)) == 1


def _horovod_0_21_3_compat_mode():
    return int(os.environ.get("BAGUA_HOROVOD_0_21_3_COMPAT_MODE", 0)) == 1
