import json
import os

class CaseLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        # Clear file if it exists so we don't append to old logs
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def log(self, data_dict):
        """Appends a dictionary as a JSON line to the log file."""
        with open(self.filepath, "a") as f:
            f.write(json.dumps(data_dict) + "\n")