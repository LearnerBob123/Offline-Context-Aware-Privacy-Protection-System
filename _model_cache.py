"""
Singleton model store for the Streamlit app.

Because Streamlit re-executes app.py via exec() on every rerun, module-level
variables defined there are reset.  Variables defined in an *imported* module
are held by Python's import cache (sys.modules) and survive across reruns.
"""

import threading

store: dict = {}                   # populated by loader thread
lock  = threading.Lock()
ready = threading.Event()          # set() when loading is done
error: list = []                   # [traceback_str] on failure
started = threading.Event()        # set() as soon as loader thread is spawned
