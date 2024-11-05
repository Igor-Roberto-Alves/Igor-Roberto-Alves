import random


def generateMap(m: int, n: int, ground_water_ration = .2, water = '0', ground = '1' ):
    r = int(m*n*ground_water_ration+.5)
    newMap = [[water]*m for _ in range(n)]
    coord = [(i, j) for i in range(n) for j in range(m)]
    random.shuffle(coord)
    for i, j in coord[:r]:
        newMap[i][j] = ground
    return newMap

def save_map(map_s, path = 'new_map.txt'):
    with open(path, 'wt') as f:
        for row in map_s:
            f.write("".join(map(str, row)))
            f.write('\n')

def print_map(map_p):
    for row in map_p:
        print("".join(map(str, row)))

if __name__ == '__main__':
    random.seed(10012)
    m1 = generateMap(50, 10, 0.1, '0', '1')
    print_map(m1)

    m2 = generateMap(100, 120)
    save_map(m2, 'test_map.txt')
