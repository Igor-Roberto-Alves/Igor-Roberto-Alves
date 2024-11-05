import random
from ler_txt import txt_to_matrix

def island_min_max(matrix):
    def neighbors(index):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1),(1,1),(1,-1),(-1,1),(-1,-1)]
        neighbors = []
        for dx, dy in directions:
            x, y = index[0] + dx, index[1] + dy
            if 0 <= x < n and 0 <= y < m:
                neighbors.append((x, y))
        return neighbors

    n = len(matrix)
    m = len(matrix[0])
    visitados = set()  
    count = 0
    ilhas = []
    for i in range(n):
        for j in range(m):
            if matrix[i][j] == '1' and (i, j) not in visitados:
                count += 1
                fila = [(i, j)]
                ilha = set()
                ilha.add((i,j))
                while fila:
                    atual = fila.pop(0)
                    if atual not in visitados:
                        visitados.add(atual)
                        for vizinho in neighbors(atual):
                            if matrix[vizinho[0]][vizinho[1]] == '1' and vizinho not in visitados:
                                ilha.add((vizinho[0],vizinho[1]))
                                fila.append(vizinho)

                lista_ilha = [v for v in ilha]
                ilhas.append(lista_ilha)
                
                
    ilha_min = ilhas[0]
    ilha_max = ilhas[0]
    for ilha in ilhas:
        if len(ilha) > len(ilha_max):
            ilha_max = ilha
        if len(ilha) < len(ilha_min):
            ilha_min = ilha

    ilha_min.sort()
    ilha_max.sort()
    return ilha_min, ilha_max

if __name__ == "__main__":
    # Testando com a matriz
    matriz = [
        ['1', '1', '1', '0', '0'],
        ['0', '1', '0', '0', '1'],
        ['1', '0', '0', '1', '1'],
        ['0', '0', '0', '1', '0'],
        ['1', '0', '1', '1', '0']
    ]

    matriz = txt_to_matrix("/home/al.igor.alves/Downloads/test_map.txt")

    print(island_min_max(matriz))
   
