import time
import datetime
import numpy as np

class Logger():
    def __init__(self, max_iter, current_iter=0, log_freq=100):
        self.start_time = time.time()
        self.max_iter = max_iter
        self.current_iter = current_iter
        self.log_freq = log_freq
        self.last_is = 0
        self.last_fid = np.inf
        
        
    def __iter__(self):
        return self
    
    
    def __next__(self):
        if self.current_iter >= self.max_iter:
            raise StopIteration
        if self.current_iter > 0 and self.current_iter % self.log_freq == 0:
            self.log_progress()
        self.current_iter += 1
        return self.current_iter - 1
 
    def pritn_metrics(self, last_fid, last_is):
        self.last_fid = last_fid
        self.last_is = last_is[0]
        print(f'Step: {self.current_iter + 1}, FID: {last_fid:.4f}, IS: {last_is[0]:.4f} +- {last_is[1]:.4f}\n')
        
    
    def log_progress(self):
        current_time = time.time()
        delta = int(current_time - self.start_time)
        hours_passed = delta // 3600
        mimuts_passed = delta % 3600 // 60
        seconds_passed = delta % 3600 % 60
        
        time_left = int(delta / (self.current_iter + 1) * self.max_iter - delta)
        hours_left = time_left // 3600
        mimuts_left = time_left % 3600 // 60
        seconds_left = time_left % 3600 % 60
        
        print(f'Current iter {self.current_iter + 1} \ {self.max_iter}, {(self.current_iter + 1) / self.max_iter * 100 :.1f}% complete, last FID: {self.last_fid:.4f}, last IS: {self.last_is:.4f}, time passed {hours_passed}h:{mimuts_passed}m:{seconds_passed}s, approx time left {hours_left}h:{mimuts_left}m:{seconds_left}s')