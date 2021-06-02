import numpy as np
from tqdm import tqdm

from ops import tensor_product, quantum_one, quantum_zero, quantum_eye, quantum_flip, MakeR, EPS, Fidelity, NormalizeAngle

class EnumAllCNOT(object):
    def __init__(self, quantum_count, layer_count):
        self.quantum_count = quantum_count
        self.layer_count = layer_count
        self.size = (quantum_count*(quantum_count-1))**layer_count
        self.gen = self.Generator()
        pass

    def Generator(self):
        cnot = np.zeros([self.layer_count, 2], dtype=int)
        code = np.zeros([self.layer_count], dtype=int)
        sel = []
        for i in range(self.quantum_count):
            for j in range(i+1, self.quantum_count):
                sel.append((i, j))
                sel.append((j, i))
        D = len(sel)
        while True:
            for i in range(self.layer_count):
                cnot[i]=sel[code[i]]
            yield cnot
            code[0]+=1
            for i in range(self.layer_count):
                if code[i]==D:
                    code[i]=0
                    if i+1<self.layer_count:
                        code[i+1]+=1
                else:
                    break

    def __len__(self):
        return self.size

    def __call__(self):
        return self.gen.__next__()

def RandomCNOT(quantum_count):
    i=j=0
    while j==i:
        i, j = np.random.randint(quantum_count, size=2)
    return i, j

def RandomCNOTs(quantum_count, layer_count):
    cnot = np.zeros([layer_count, 2], dtype=int)
    for c in range(layer_count):
        cnot[c, :] = RandomCNOT(quantum_count)
    return cnot

class QSystem(object):
    def __init__(self, layer_list, normalize=True):
        self.layer_list = layer_list
        self.quantum_count = layer_list[0].quantum_count
        if normalize:
            self.Normalize()

    def __call__(self, quantum_status):
        '''quantum_status: list or vector(np.matrix)'''
        if isinstance(quantum_status, list):
            quantum_status = tensor_product(quantum_status)
        for layer in self.layer_list:
            quantum_status = layer(quantum_status)
        return quantum_status

    def __len__(self):
        return len(self.layer_list)

    def Normalize(self):
        for layer in self.layer_list:
            if isinstance(layer, RLayer):
                layer.Normalize()

    @property
    def matrix(self):
        I = quantum_eye(self.quantum_count)
        for layer in self.layer_list:
            I = layer.matrix*I
        return I

    @property
    def string(self):
        return ''.join([layer.string for layer in self.layer_list])

class QLayer(object):
    '''base layer'''
    def __init__(self, quantum_count):
        self.quantum_count = quantum_count

    @property
    def matrix(self):
        '''get the matrix of the layer'''
        raise NotImplementedError('matix must realize.')

    @property
    def string(self):
        raise NotImplementedError('string must realize.')

    def __call__(self, quantum_status):
        if isinstance(quantum_status, list):
            quantum_status = tensor_product(quantum_status)
        # print(type(self.matrix), type(quantum_status))
        return self.matrix * quantum_status

class RLayer(QLayer):
    '''旋转门'''
    def __init__(self, thetas, **kwargs):
        super(RLayer, self).__init__(**kwargs)
        if len(thetas) != self.quantum_count:
            raise ValueError('theta count must be equal to quantum_count.')
        self.thetas = thetas

    def Normalize(self):
        self.thetas = [NormalizeAngle(t) for t in self.thetas]

    @property
    def matrix(self):
        return tensor_product([MakeR(t) for t in self.thetas])
    
    @property
    def string(self):
        return ''.join(['R %d %.16f\n'%(i, t) for i,t in enumerate(self.thetas) if abs(t)>1e-8])
        # return ''.join(['R %d %.16f\n'%(i, t) for i,t in enumerate(self.thetas)])

class CLayer(QLayer):
    '''控制翻转门'''
    def __init__(self, control_quantum, target_quantum, **kwargs):
        super(CLayer, self).__init__(**kwargs)
        if control_quantum>=self.quantum_count or target_quantum>=self.quantum_count or control_quantum==target_quantum:
            raise ValueError('control=%d and target=%d is wrong in CLayer with quantum_count=%d'%(control_quantum, target_quantum, self.quantum_count))
        self.control_quantum = control_quantum
        self.target_quantum = target_quantum

    @property
    def matrix(self):
        bit_c = 1<<(self.quantum_count-self.control_quantum-1)
        bit_t = 1<<(self.quantum_count-self.target_quantum-1)
        A = quantum_eye(self.quantum_count)
        for i in range(1<<self.quantum_count):
            if (i&bit_c)==bit_c and (i&bit_t)==0:
                j = i|bit_t
                A[[i,j],:]=A[[j,i],:]
        return A

    @property
    def string(self):
        return 'C %d %d\n'%(self.control_quantum, self.target_quantum)

class ILayer(QLayer):
    '''恒等门'''
    @property
    def matrix(self):
        return quantum_eye(self.quantum_count)

    @property
    def string(self):
        return 'I\n'

def ParseToQSystem(s, quantum_count):
    '''不能用于多重R的情况'''
    layers = []
    thetas = [0]*quantum_count
    for line in s.split('\n'):
        line = line.strip()
        if len(line)!=0:
            mark, p1, p2 = line.split(' ')
            if mark == 'R':
                # thetas = [0]*quantum_count
                thetas[int(p1)]=float(p2)
            elif mark == 'C':
                layers.append(RLayer(thetas, quantum_count=quantum_count))
                layers.append(CLayer(int(p1), int(p2), quantum_count=quantum_count))
                thetas = [0]*quantum_count
    if sum([abs(t) for t in thetas])>0:
        layers.append(RLayer(thetas, quantum_count=quantum_count))
    return QSystem(layers)

def CostCompute(s):
    sc=0
    for line in s.split('\n'):
        if len(line)>0:
            if line[0]=='R':
                sc+=1
            elif line[0]=='C':
                sc+=8
    return sc

def SearchParams(U, quantum_count, cnot_layers, EPOCH_STAGE_1=10000, EPOCH_STAGE_2=1000):

    def MakeSystem(params, quantum_count=quantum_count, cnot_layers=cnot_layers):
        layers = [RLayer(list(params[0]), quantum_count=quantum_count)]
        for i, (c, t) in enumerate(cnot_layers):
            layers.append(CLayer(c, t, quantum_count=quantum_count))
            layers.append(RLayer(list(params[i+1]), quantum_count=quantum_count))
        return QSystem(layers)

    best_score = 0
    param_size = (len(cnot_layers)+1, quantum_count)
    for epoch in tqdm(range(EPOCH_STAGE_1)):
        params = np.random.uniform(0, 2*np.pi, size=param_size)
        model = MakeSystem(params)
        M = model.matrix
        score = Fidelity(U, M)
        if score>best_score:
            # print('epoch_%d, score = %g'%(epoch, score))
            best_score = score
            best_param = params
            best_model = model.string
    for eps in tqdm([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]):
        for epoch in range(EPOCH_STAGE_2):
            dt = np.random.uniform(-eps, eps, size=param_size)
            params = best_param+dt
            model = MakeSystem(params)
            score = Fidelity(U, model.matrix)
            if score>best_score:
                # print('eps=%g, epoch=%d, score = %g'%(eps, epoch, score))
                best_score = score
                best_param = params
                best_model = model.string
    return best_score, best_model

def Minimize(func, x_min, x_max, EPS=1e-8):
    alpha = (np.sqrt(5)-1)/2 # 0.618
    x0=x_min
    x3=x_max
    x1=x3-alpha*(x3-x0)
    x2=x0+alpha*(x3-x0)
    y1=func(x1)
    y2=func(x2)
    while x2-x0>EPS:
        if y1<y2:
            x0=x1
            x1=x2
            y1=y2
            x2=x0+alpha*(x3-x0)
            y2=func(x2)
        else:
            x3=x2
            x2=x1
            y2=y1
            x1=x3-alpha*(x3-x0)
            y1=func(x1)
    x = (x1+x2)/2
    return x

if __name__ == '__main__':
    layer1 = RLayer([0, 1.5], quantum_count=2)
    layer2 = ILayer(quantum_count=2)
    layer3 = RLayer([0, 1.5], quantum_count=2)
    model = QSystem([layer1, layer2, layer3])
    print('Model matrix:')
    print(model.matrix)
    q0 = quantum_zero()
    q1 = quantum_one()
    q = tensor_product([q0, q1])
    print('Quantum status:')
    print(model(q).transpose())
    print('Model string:')
    print(model.string)

    print('Test C2Layer')
    C2Layer = lambda:CLayer(0,1,quantum_count=2)
    model = C2Layer()
    q0 = quantum_zero()
    q1 = quantum_one()
    q = tensor_product([q0, q1])
    print('Init status:')
    print(q.transpose())
    print('After C2Layer:')
    print(model(q).transpose())
    print()

    q = tensor_product([q1, q0])
    print('Init status:')
    print(q.transpose())
    print('After C2Layer:')
    print(model(q).transpose())
    print()
    
    q = tensor_product([q0, q0])
    print('Init status:')
    print(q.transpose())
    print('After C2Layer:')
    print(model(q).transpose())
    print()
    
    q = tensor_product([q1, q1])
    print('Init status:')
    print(q.transpose())
    print('After C2Layer:')
    print(model(q).transpose())

    print('CNOT(0,1):')
    layer = CLayer(0,1,quantum_count=2)
    print(layer.matrix, '\n')

    print('CNOT(1,0):')
    layer = CLayer(1,0,quantum_count=2)
    print(layer.matrix, '\n')

    print('CNOT(1,2):')
    layer = CLayer(1,2,quantum_count=3)
    print(layer.matrix, '\n')

    print('CNOT(2,0):')
    layer = CLayer(2,0,quantum_count=3)
    print(layer.matrix, '\n')

    print('Test enum:')
    creator = EnumAllCNOT(3, 2)
    print('len = ', len(creator))
    for _ in range(5):
        print(creator())

    from Reader import ReadSystem
    S1 = QSystem([
        RLayer([-0.5,0,8], quantum_count=3),
        CLayer(1, 2, quantum_count=3),
        RLayer([-1.5,1,3], quantum_count=3),
    ], normalize=False)
    U1 = S1.matrix
    print('Before:')
    print(S1.string)
    S1.Normalize()
    print('After:')
    print(S1.string)
    U2 = S1.matrix
    print('Fidelity:', Fidelity(U1, U2))

