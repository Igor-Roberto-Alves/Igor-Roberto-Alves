import random
import pygame
import time

# Inicialização do Pygame
pygame.init()

# Carregar as imagens PNG para cada estado
TREE_ALIVE_IMG = pygame.image.load("av2prog\Tree_Small.png")
TREE_BURNING_IMG = pygame.image.load("av2prog\Fire_Small.png")
#LAKE_IMG = pygame.image.load("av2prog\lake.png")

# Ajuste o tamanho das imagens (caso necessário)
cell_size = 20
TREE_ALIVE_IMG = pygame.transform.scale(TREE_ALIVE_IMG, (cell_size, cell_size))
TREE_BURNING_IMG = pygame.transform.scale(TREE_BURNING_IMG, (cell_size, cell_size))

# Definições de cores
COLOR_BURNED = (0, 0, 0)    # Preto (árvores queimadas)
COLOR_WATER = (0, 0, 255)   # Azul (barreiras)

class Tree:
    def __init__(self, coord):
        self.condition = "alive"
        self.density = random.randint(50, 80)
        self.next_condition = None
        self.x = coord[0]
        self.y = coord[1]

    def neighbors(self, matriz):
        lista = []
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < len(matriz) and 0 <= ny < len(matriz[0]) and isinstance(matriz[nx][ny], Tree):
                lista.append(matriz[nx][ny])
        return lista

    def attempt_to_burn(self, matriz):
        if self.condition == "alive":
            self.next_condition = "burning"
            for neighbor in self.neighbors(matriz):
                if neighbor.condition == "alive" and neighbor.next_condition != "burning":
                    probability = 100 - neighbor.density
                    if random.random() < probability / 100:
                        neighbor.next_condition = "burning"
                        

    def update_condition(self,matriz):
        if self.next_condition == "burned":
            matriz[self.x][self.y] = "v"

        if self.next_condition == "burning":
            self.attempt_to_burn(matriz)
            self.condition = self.next_condition
            self.next_condition = "burned"
    
        
        

    def extinguish_fire(self):
        if self.condition == "burning":
            self.condition = "alive"

    def __repr__(self):
        if self.condition == "alive":
            return "1"
        if self.condition == "burning":
            return "b"
        if self.condition == "burned":
            return "0"

class Barrier:
    def __init__(self, coord):
        self.x = coord[0]
        self.y = coord[1]
    
    def __repr__(self):
        return "a"
    

class Forest:
    def __init__(self, matriz):
        self.matriz = matriz
        self.n = len(matriz)
        self.m = len(matriz[0])
    
    def incendio(self):
        while True:
            k = random.randint(0, self.n - 1)
            l = random.randint(0, self.m - 1)
            if isinstance(self.matriz[k][l], Tree) and self.matriz[k][l].condition == "alive":
                self.matriz[k][l].attempt_to_burn(self.matriz)
                break

    def update_forest(self):
        for row in self.matriz:
            for cell in row:
                if isinstance(cell, Tree):
                    cell.update_condition(self.matriz)

    def extinguish_tree_at(self, x, y):
        col, row = x // cell_size, y // cell_size
        if 0 <= row < self.n and 0 <= col < self.m:
            cell = self.matriz[row][col]
            if isinstance(cell, Tree):
                cell.extinguish_fire()

    def surge_trees(self):
        for i in range(self.n):
            for j in range(self.m):
                if self.matriz[i][j] == "v":
                    a = random.randint(1,2)
                    if a == 1:
                        self.matriz[i][j] = Tree((i,j))

    def tree_f_pos(self,coord):
        self.matriz[coord[0]][coord[1]] == Tree((coord[0],coord[1]))

def draw_forest(screen, forest):
    for i in range(forest.n):
        for j in range(forest.m):
            cell = forest.matriz[i][j]
            if isinstance(cell, Tree):
                if cell.condition == "alive":
                    screen.blit(TREE_ALIVE_IMG, (j * cell_size, i * cell_size))
                elif cell.condition == "burning":
                    screen.blit(TREE_BURNING_IMG, (j * cell_size, i * cell_size))
            elif isinstance(cell, Barrier):
                pygame.draw.rect(screen, (173, 216, 230), (j * cell_size, i * cell_size, cell_size, cell_size))
                #screen.blit(LAKE_IMG, (j * cell_size, i * cell_size))

def main():
    screen_width, screen_height = 400, 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Forest Fire Simulation")
    screen.fill((0, 0, 255))
    
    matriz = [[Tree((i, j)) for j in range(20)] for i in range(20)]
    matriz[4][4] = Barrier((4,4))
    matriz[4][5] = Barrier((4,5))
    matriz[5][5] = Barrier((5,5))

    forest = Forest(matriz)
    forest.incendio()

    running = True
    while running:
        screen.fill((85, 107, 47))
        incendio = True
        for row in matriz:
            for cell in row:
                if isinstance(cell, Tree):
                    if cell.condition == "burning":
                        incendio = False
            print(row)
        if incendio:
            forest.incendio()

        draw_forest(screen, forest)
        pygame.display.flip()
        forest.surge_trees()


        forest.update_forest()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                forest.tree_f_pos((x, y))

        time.sleep(0.5)

    pygame.quit()

if __name__ == "__main__":
    main()
