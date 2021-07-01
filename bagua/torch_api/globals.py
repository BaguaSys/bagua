_global_state = None


def _set_global_state(global_state):
    global _global_state
    _global_state = global_state


def _get_global_state():
    global _global_state
    return _global_state


def is_initialized():
    """
    Checking if bagua global communication state has been initialized
    """
    global _global_state
    return _global_state is not None
