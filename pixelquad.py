import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class pixel:

    def __init__(self, ras:np.array, rt = None):
        
        self.coordinates = ras
        self.root = rt
        if rt == None:
            self.level = 0
        else:
            self.level = self.level()
            

        self.sons = []
        self.color = "white"

    def level(self):
        return self.root.level + 1

    
    def down_pixel(self):

        p1, p2, p3, p4 = self.coordinates
        mid_p = ((p1+p2)/2 + (p3+p4)/2)/2
        mid_low = (p1 + p2)/2
        mid_up = (p3 + p4)/2
        mid_lf = (p1 + p3)/2
        mid_rt =  (p2 + p4)/2
        s1 = [p1, mid_low, mid_lf, mid_p]
        s2 = [mid_low, p2, mid_p, mid_rt]
        s3 = [mid_lf, mid_p, p3, mid_up]
        s4 = [mid_p, mid_rt, mid_up, p4]
        self.sons = [pixel(s1, rt = self),pixel(s2, rt = self),pixel(s3, rt = self),pixel(s4, rt = self)]


class pixelquad:

    def __init__(self, root:pixel):
        
        if type(root) != pixel:
            return KeyError
        
        self.root = root
        self.level = 1

    def down(self, pxl):
        pxl.down_pixel()
        self.level += 1
    


def plot_quadtree(pxl, ax):
    """
    Função recursiva que desenha os pixels da árvore.
    """
    # As coordenadas estão no formato: [p1, p2, p3, p4]
    # Onde p1=(x,y) inferior esquerdo, p2=(x,y) inferior direito, etc.
    p1 = pxl.coordinates[0]
    p4 = pxl.coordinates[3]
    
    # Calculamos largura e altura para o retângulo
    width = p4[0] - p1[0]
    height = p4[1] - p1[1]
    
    # Criamos o desenho do quadrado (borda)
    rect = patches.Rectangle((p1[0], p1[1]), width, height, 
                             linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    
    # Se o pixel tiver filhos, desenhamos os filhos recursivamente
    if pxl.sons:
        for son in pxl.sons:
            plot_quadtree(son, ax)


    


    

