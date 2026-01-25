import numpy as np

class dct_obj:
    def __init__(self, points: np.array, lig: list):
        self.points = points
        self.lig = lig
        
class circle:
    def __init__(self, cnt, r):
        self.center = cnt
        self.ray = r
        
cnt = np.array([0.3, 0.4])
r = 0.21
c1 = circle(cnt, r)