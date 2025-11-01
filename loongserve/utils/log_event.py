import os
import json
import time
import threading
from datetime import datetime

import numpy as np
from loongserve.longserve_server.io_struct import Req

LOG_DIR = "/workspace/LoongServe/logs/events"
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"events_{timestamp}.jsonl")


_write_lock = threading.Lock()


def log_event(event: str, modify_func = None, **kwargs):
    return
    if modify_func is not None:
        for key, value in kwargs.items():
            try:
                kwargs[key] = modify_func(value)
            except Exception as e:
                print(f"[log_event] modify_func failed for key '{key}': {e}")

    record = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "event": event,
        **kwargs,
    }
    def convert(o):
        if isinstance(o, (np.integer, np.int64)):
            return int(o)
        elif isinstance(o, (np.floating, np.float64)):
            return float(o)
        elif isinstance(o, (np.bool_)):
            return bool(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return str(o)
        
    with _write_lock, open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(
        json.dumps(record, ensure_ascii=False, default=convert, indent=2) + "\n"
    )


def find_and_modify(self, obj, target_type=None, modify_fn=None):
    if isinstance(obj, target_type):
        if modify_fn:
            return modify_fn(obj)
        return obj
    elif isinstance(obj, Req):
        return obj.request_id
    elif isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_dict[k] = self.find_and_modify(v, target_type, modify_fn)
        return new_dict
    elif isinstance(obj, list):
        new_list = []
        for item in obj:
            new_list.append(self.find_and_modify(item, target_type, modify_fn))
        return new_list
    elif hasattr(obj, "__dict__"):
        for attr, value in vars(obj).items():
            new_value = self.find_and_modify(value, target_type, modify_fn)
            setattr(obj, attr, new_value)
        return obj
    else:
        return obj
def get_instance_info(batch):
    instance_batch_mapping = {sp_rank: batch for batch in batch for sp_rank in batch.occupied_instances}

log_event("LOGGER_INIT", pid=os.getpid(), file=LOG_FILE)