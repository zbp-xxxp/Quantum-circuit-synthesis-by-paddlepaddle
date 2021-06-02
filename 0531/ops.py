import numpy as np
from math import sin, cos

EPS = 1e-8

def NormalizeAngle(theta):
    '''make theta in [0, 2*np.pi)'''
    while theta<0:
        theta+=2*np.pi
    while theta>=2*np.pi:
        theta-=2*np.pi
    return theta

def quantum_zero():
    return np.matrix([[1],[0]], dtype=float)

def quantum_one():
    return np.matrix([[0],[1]], dtype=float)

def tensor_product(x, y=None):
    if y is None and isinstance(x, list):
        v = np.matrix([1], dtype=float)
        for xi in x:
            v = np.kron(v, xi)
    else:
        v = np.kron(x, y)
    return v

def MakeR(theta):
    theta /= 2
    return np.matrix([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

def quantum_eye(quantum_count=1):
    return np.matrix(np.eye(2**quantum_count, dtype=float))

def quantum_flip(i=0, j=1, c=1):
    A = quantum_eye(c)
    A[[i,j], :] = A[[j, i], :]
    return A
    # return np.matrix([[0,1],[1,0]], dtype=float)

def Fidelity(u1, u2):
    if not isinstance(u1, np.matrix):
        u1=np.matrix(u1)
    if not isinstance(u2, np.matrix):
        u2=np.matrix(u2)
    tr = np.trace(u1*u2.transpose())
    return abs(tr)/u1.shape[0]

if __name__ == '__main__':
    q0 = quantum_zero()
    q1 = quantum_one()
    import random
    r0 = MakeR(random.uniform(0, np.pi*2))
    r1 = MakeR(random.uniform(0, np.pi*2))
    q0 = r0*q0
    q1 = r1*q1
    print('Init status:')
    print(q0.transpose())
    print(q1.transpose())
    
    r0 = MakeR(random.uniform(0, np.pi*2))
    r1 = MakeR(random.uniform(0, np.pi*2))

    q = tensor_product(q0, q1)
    r = tensor_product(r0, r1)
    print('Two different compute sequence.')
    print((r*q).transpose())
    print(tensor_product(r0*q0, r1*q1).transpose())
    print()

    print('product of eye and swap:')
    print(tensor_product([quantum_eye(), quantum_flip()]))
    print('product of swap and eye:')
    print(tensor_product([quantum_flip(), quantum_eye()]))
    
    t1 = -0.5
    t2 = NormalizeAngle(t1)
    print('NormalizeAngle Fidelity:', Fidelity(MakeR(t1), MakeR(t2)))



