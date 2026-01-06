import time

class RequestMetrics:
    def __init__(self):
        self.start = time.time()
        
    def mark(self, name):
        setattr(self, name, time.time())
        
    def duration(self, start, end):
        return getattr(self, end) - getattr(self, start)
        
    def total_time(self):
        return time.time() - self.start