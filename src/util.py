import json, time


json_cache = {}

def load_json(file_name):
    if file_name in json_cache:
        return json_cache[file_name]
    
    with open(file_name, 'r') as file:
        data = json.load(file)
    json_cache[file_name] = data
    return data


class Timer:
    def __init__(self):
        self.start = 0

    def reset(self):
        self.start = time.perf_counter()
    
    def get(self, start = None):
        return time.perf_counter() - (start or self.start)
    
    def now(self, start = None):
        return f"{self.get(start):.02f}s"
    

class Progress:
    def __init__(self, timer):
        self.message = ""
        self.tick = 0
        self.total = 0
        self.time_start = 0
        self.section_start = 0
        self.flush = 0
        self.entries = {}

        self.timer = timer

    def update(self, tick, forced = False):
        if self.timer.get(self.flush) > 0.5 or forced:
            for _ in range(len(self.entries) + 1):
                print(f"\33[2K\r\033[F", end = "")
            print(f"{self.message} ({tick}/{self.total}) [time = {self.timer.now(self.section_start)}, elapsed = {self.timer.now(self.time_start)}]")
            for k, v in self.entries.items():
                print(f"{k} = {v}")
            self.flush = time.perf_counter()
            self.tick = tick
            self.section_start = time.perf_counter()

    def log(self, message, total):
        self.message = message
        self.total = total
        self.time_start = time.perf_counter()
        self.section_start = self.time_start
        self.entries.clear()
        print(f"{self.message} (0/{self.total}) [time = N/A, elapsed = 0.00s]")

    def add_entry(self, name):
        self.entries[name] = "N/A"

    def update_entry(self, name, value):
        self.entries[name] = value

    def done(self):
        print(f"\033[F{self.message} ({self.total}/{self.total}) [elapsed: {self.timer.now(self.time_start)}]")

    def terminate(self):
        print(f"\033[F{self.message} ({self.tick}/{self.total}) [elapsed: {self.timer.now(self.time_start)}]")
        print("    terminated because time is dwindling")