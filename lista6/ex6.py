from ex5 import find_lake
from ex4 import number_of_islands
import colorama

colorama.init(autoreset=True)

# Definindo a matriz para teste
matrix = [
    ['1', '1', '1', '1', '1'],
    ['1', '1', '0', '1', '1'],
    ['1', '1', '1', '1', '0'],
    ['1', '1', '1', '1', '1'],
    ['1', '1', '1', '1', '1'],
    ['1', '0', '0', '0', '0']
]

# Chamando a função para encontrar os lagos
lakes = find_lake(matrix)
print(lakes)

# Função para imprimir a matriz colorida
def print_arquip(matrix, lakes):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == "1":
                # Quadrado marrom (substituído por amarelo)
                print(colorama.Back.YELLOW + ' ', end='')
            elif matrix[i][j] == "0":
                if any((i, j) in sublist for sublist in lakes):
                    # Quadrado azul claro
                    print(colorama.Back.CYAN + ' ', end='')
                else:
                    # Quadrado azul escuro
                    print(colorama.Back.BLUE + ' ', end='')
        print()  # Move para a próxima linha após cada linha da matriz

# Exibindo a matriz colorida
print_arquip(matrix, lakes)
