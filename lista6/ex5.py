def find_lake(matrix):
    def neighbors(index):
        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ]  # Agora só consideraremos vizinho em quatro direções, cima, baixo, esquerda e direita
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
    water = []
    for i in range(n):
        for j in range(m):
            if matrix[i][j] == "0" and (i, j) not in visitados:
                count += 1
                fila = [(i, j)]
                ilha = set()
                ilha.add((i, j))
                while fila:
                    atual = fila.pop(0)
                    visitados.add(atual)
                    for vizinho in neighbors(atual):
                        if (
                            matrix[vizinho[0]][vizinho[1]] == "0"
                            and vizinho not in visitados
                        ):
                            ilha.add((vizinho[0], vizinho[1]))
                            fila.append(vizinho)

                lista_ilha = [v for v in ilha]
                lista_ilha.sort()
                water.append(lista_ilha)

    lakes = []
    for conjunto in water:
        is_border_touching = False
        for zero in conjunto:
            # Verifica se o elemento zero está na borda
            if zero[0] == 0 or zero[0] == n - 1 or zero[1] == 0 or zero[1] == m - 1:
                is_border_touching = True
                break  # Interrompe o loop interno se encontrar um zero na borda

        # Apenas adiciona o conjunto se ele não tocar as bordas
        if not is_border_touching:
            lakes.append(conjunto)

    return lakes  # Water carregará apenas os lagos que a matriz tiver, então se estiver vazia não há lagos
    #Se a lista estiver vazia a matriz não contém nenhum lago

if __name__ == "__main__":
    matrix = [
        ["1", "1", "0", "1", "1"],
        ["1", "1", "1", "0", "1"],
        ["1", "1", "1", "1", "1"],
        ["0", "0", "1", "0", "1"],
        ["1", "1", "1", "1", "1"],
        ["1", "0", "0", "0", "0"],
    ]

    # Chamando a função para encontrar os lagos
    lakes = find_lake(matrix)
    print(lakes)
