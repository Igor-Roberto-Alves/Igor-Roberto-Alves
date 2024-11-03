def number_of_islands(matrix):
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

    for i in range(n):
        for j in range(m):
            if matrix[i][j] == '1' and (i, j) not in visitados:
                count += 1
                fila = [(i, j)]
                while fila:
                    atual = fila.pop(0)
                    if atual not in visitados:
                        visitados.add(atual)
                        for vizinho in neighbors(atual):
                            if matrix[vizinho[0]][vizinho[1]] == '1' and vizinho not in visitados:
                                fila.append(vizinho)

    return count

if __name__ == "__main__":
    # Testando com a matriz
    matriz = [
        ['1', '1', '1', '0', '0'],
        ['0', '1', '0', '0', '1'],
        ['1', '0', '0', '1', '1'],
        ['0', '0', '0', '0', '0'],
        ['1', '0', '1', '1', '0']
    ]

    print(number_of_islands(matriz))
