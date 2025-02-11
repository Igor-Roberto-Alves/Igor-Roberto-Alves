#Grafo por lista de adjascência
class vert():
    def __init__(self,name,valor=None):
        self.valor = valor
        self.name = name
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    

class grafo():
    def __init__(self,list_vert):
        self.lv = list_vert
        self.edge = {vert : [] for vert in self.lv}
    
    def __str__(self):
        stri = ""
        for vert in self.edge.keys():
            stri += vert.name + ":"
            stri += " " + "".join(str([conex for conex in self.edge[vert]]))
            stri += "\n"
        return stri
        
    def adjascent(self,x,y):
        if y in self.edge[x]:
            return True
        if x in self.edge[y]:
            return True
        return False

    def neighbors(self,x):
        lista_v = []
        for i in self.edge[x]:
            lista_v.append(i)
        return lista_v

    def add_vertex(self,x):
        if x in self.lv:
            return False
        else:
            self.edge[x] = []

            return True

    
    def remove_vertex(self,x):
        if x in self.edge:
            del self.edge[x]

            for vert in self.edge.keys():
                if x in self.edge[vert]:
                    self.edge[vert].remove(x)
            return True
        return False
    
    def add_edge(self,x,y):
        if y in self.edge[x]:
            return False
        else:
            self.edge[x].append(y)
            return True
    
    def remove_edge(self,x,y):
        if y in self.edge[x]:
            self.edge[x].remove(y)
            return True
        else:
            return False
    
    @staticmethod
    def get_vertex_value(x):
        return x.valor
    
    @staticmethod
    def set_vertex_value(x, new_valor):
        x.valor = new_valor


def bfs(grafo, vert_ini):
    percorridos = []
    fila = [vert_ini]
    while fila:
        vert_atual = fila.pop(0)
        if vert_atual not in percorridos:
            percorridos.append(vert_atual)
            for i in grafo.neighbors(vert_atual):
                if i not in percorridos:
                    fila.append(i)
    return percorridos

A = vert("A",1)
B = vert("B",1)
C = vert("C",0)
D = vert("D",0)
E = vert("E",1)
grafo1  = grafo([A,B,C,D,E])
grafo1.add_edge(A,B)
grafo1.add_edge(B,C)
grafo1.add_edge(C,A)
grafo1.add_edge(C,E)
print(grafo1)
print("Busca Em Largura partindo do vértice A:\n ", bfs(grafo1,A))

        
