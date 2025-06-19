# register_signal.py

signal_registry = {}

def register_signal(signal_name: str, function_code: str):
    """
    Dynamically register a signal-generating function from source code.
    The function must be named `signal`.
    """
    local_env = {}
    try:
        exec(function_code, {}, local_env)
        signal_func = local_env["signal"]
        signal_registry[signal_name] = signal_func
        return True
    except Exception as e:
        print(f"Signal registration error: {e}")
        return False

def get_signal(signal_name: str):
    return signal_registry.get(signal_name)
