import os
import numpy as np

from utils import ParseToQSystem
from ops import Fidelity

def ReadSystem(path, quantum_count):
    with open(path, 'r') as f:
        s = f.read()
    return ParseToQSystem(s, quantum_count)

def ReadU(path):
    '''
    read a U matrix, split by blank
    '''
    # with open(path, 'r') as f:
    #     for l in f:
    #         print([v.strip() for v in l.strip().split(' ', )])
    with open(path, 'r') as f:
        x = [[float(v) for v in l.strip().split(' ')]for l in f]
        return np.matrix(x)


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    x = ReadU('Questions/Question_2_Unitary.txt')
    print('shape:', x.shape)
    print(x)
    y = ReadSystem('../Answer/Question_2_Answer.txt', 2).matrix
    print(y)
    print('Similar: ', Fidelity(x, y))
    print()

    x = ReadU('Questions/Question_3_Unitary.txt')
    print('shape:', x.shape)
    print(x)
    y = ReadSystem('../Answer/Question_3_Answer.txt', 3).matrix
    print(y)
    print('Similar: ', Fidelity(x, y))
    print()

    x = ReadU('Questions/Question_4_Unitary.txt')
    print('shape:', x.shape)
    print(x)
    y = ReadSystem('../Answer/Question_4_Answer.txt', 3).matrix
    print(y)
    m=x*y.transpose()
    print(m)
    print('trace:', np.trace(m))
    print('sum', sum([m[i,i] for i in range(8)]))
    print('Similar: ', Fidelity(x, y))
    print()

    x = ReadU('Questions/Question_5_Unitary.txt')
    print('shape:', x.shape)
    # print(x)
    y = ReadSystem('../Answer/Question_5_Answer.txt', 4).matrix
    print('Similar: ', Fidelity(x, y))
    print()

    x = ReadU('Questions/Question_6_Unitary.txt')
    print('shape:', x.shape)
    # print(x)
    y = ReadSystem('../Answer/Question_6_Answer.txt', 8).matrix
    print('Similar: ', Fidelity(x, y))
    print()

