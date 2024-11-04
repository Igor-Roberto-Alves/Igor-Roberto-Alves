def txt_to_matrix(path):
    with open(path) as f:
        linhas = f.read().splitlines()
    matrix = []
    for linha in linhas:
        a  = []
        for char in linha:
            a.append(char)
        matrix.append(a)
    return matrix

                    
           
