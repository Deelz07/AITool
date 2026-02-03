path = 'prealgebra_exponents.txt'

def txtfile_to_string(path):
    with open(path,'r') as f:
        questions = f.read()
        print('questions were read in from the txt file')
    return questions