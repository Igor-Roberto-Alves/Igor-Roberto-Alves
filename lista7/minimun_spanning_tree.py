import copy
points = [[0,1],[0,2],[9,8],[7,6]]
def oo(points):
    points.sort()
    pontos_y = copy.deepcopy(points)
    pontos_y.sort(key=lambda x: x[1])

    for ponto in points:
        vert_min = None
        for ponto2 in points:
            if ponto != ponto2:
                if not vert_min:
                    vert_min = ponto2
            
                

