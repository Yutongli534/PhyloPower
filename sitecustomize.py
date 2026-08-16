"""Local runtime patch for QIIME subprocesses in restricted process sandboxes."""
try:
    import psutil

    if psutil.cpu_count() is None:
        psutil.cpu_count = lambda logical=True: 8
except Exception:
    pass
