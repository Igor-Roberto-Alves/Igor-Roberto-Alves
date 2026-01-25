import numpy as np
from pixelquad import *

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

def pixel_in(p1, c, r):

    if euclid_dist(p1,c) <= r:
        return True
    
    return False

def euclid_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def in_out(p: pixel, center=c1.center, ray=c1.ray):
    p1, p2, p3, p4 = p.coordinates
    # p1: min_x, min_y | p4: max_x, max_y
    x_min, y_min = p1[0], p1[1]
    x_max, y_max = p4[0], p4[1]

    # 1. Verificar se todos os cantos estão dentro (Totalmente IN)
    cantos = [p1, p2, p3, p4]
    if all(euclid_dist(canto, center) <= ray for canto in cantos):
        p.color = "red"
        return "in"

    # 2. Encontrar o ponto (P) dentro do pixel mais próximo do centro do círculo
    # Isso resolve o problema da raiz e do círculo "flutuante"
    closest_x = np.clip(center[0], x_min, x_max)
    closest_y = np.clip(center[1], y_min, y_max)
    ponto_proximo = np.array([closest_x, closest_y])

    # 3. Se a distância ao ponto mais próximo for menor ou igual ao raio, 
    # há intersecção (WHITE)
    if euclid_dist(ponto_proximo, center) <= ray:
        p.color = "white"
        return "inter"
    
    # 4. Caso contrário, está totalmente fora (OUT)
    else:
        p.color = "blue"
        return "out"




        

    
    
