import os
import numpy as np
from Reader import ReadU
from ops import MakeR, Fidelity
from utils import Minimize

INPUT_TXT = 'Questions/Question_1_Unitary.txt'
ANSWER_TXT = '../Answer/Question_1_Answer.txt'

def main(in_txt=INPUT_TXT, out_txt=ANSWER_TXT):
    U = ReadU(in_txt)
    func = lambda t:Fidelity(MakeR(t), U)
    theta = Minimize(func, 0, 2*np.pi)
    with open(out_txt, 'w') as f:
        f.write('R 0 %.16lf'%theta)
    # print(x)

if __name__ == '__main__':
    if not os.path.exists('../Answer'):
        os.makedirs('../Answer')
    main()
