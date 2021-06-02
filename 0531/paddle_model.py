import time

import numpy as np

import paddle

from utils import CLayer, RLayer, QSystem, CostCompute
from ops import Fidelity, NormalizeAngle, EPS

class RLayer_d1(paddle.nn.Layer):

    def __init__(self):
        super(RLayer_d1, self).__init__()

        self.theta = self.create_parameter(shape=[1,1], default_initializer=paddle.nn.initializer.Uniform(low=0, high=2*np.pi))

        self.create_matrix()

    def set_theta(self, theta):
        '''theta must be a scalar'''
        self.theta.set_value(np.array([[theta]], dtype=np.float32))

    def create_matrix(self):
        cs = paddle.cos(self.theta/2)
        sn = paddle.sin(self.theta/2)
        self.matrix = paddle.concat([
            paddle.concat([cs, -sn], axis=1),
            paddle.concat([sn, cs], axis=1),
        ], axis=0)

    def forward(self, x):
        # self.create_matrix()
        tmp = self._helper.create_variable_for_type_inference('float32')
        self._helper.append_op(
                type="mul",
                inputs={"X": self.matrix,
                        "Y": x},
                outputs={"Out": tmp},
                attrs={"x_num_col_dims": 1,
                        "y_num_col_dims": 1})
        x = tmp
        return x

class RLayer_paddle(paddle.nn.Layer):

    def __init__(self, quantum_count):
        super(RLayer_paddle, self).__init__()

        # self.thetas = self.create_parameter(shape=[quantum_count,1], default_initializer=paddle.nn.initializer.Uniform(low=0, high=2*np.pi))
        # self.thetas = self.create_parameter(shape=[quantum_count,1], default_initializer=paddle.nn.initializer.Constant(0))
        self.thetas = paddle.nn.ParameterList([paddle.create_parameter(shape=[1,1], dtype='float32', default_initializer=paddle.nn.initializer.Constant(0)) for _ in range(quantum_count)])
        
        self.quantum_count = quantum_count

    def lock_theta(self, theta_id):
        if theta_id>=self.quantum_count:
            raise ValueError('Can not lock a theta outside.')
        self.thetas[theta_id].set_value(np.zeros([1,1], dtype=np.float32))
        self.thetas[theta_id].stop_gradient = True

    def unlock_theta(self,theta_id, pre_theta=0):
        if theta_id>=self.quantum_count:
            raise ValueError('Can not unlock a theta outside.')
        self.thetas[theta_id].set_value(np.array([[pre_theta,],], dtype=np.float32))
        self.thetas[theta_id].stop_gradient = False

    def islocked_theta(self, theta_id):
        return self.thetas[theta_id].stop_gradient

    def set_thetas(self, thetas):
        '''thetas is a numpy, has the same number of scalars with quantum_count'''
        if not isinstance(thetas, np.ndarray):
            thetas = np.array(thetas)
        thetas.reshape([-1,1])
        if len(thetas) != self.quantum_count:
            raise ValueError('len of thetas must be equal to quantum_count.')
        for st, t in zip(self.thetas, thetas):
            st.set_value(np.array([[t]], dtype=np.float32))
    
    def forward(self, x):
        #TODO:
        matrix = paddle.ones([1,1], dtype=np.float32)
        for t in self.thetas:
            cs = paddle.cos(t/2)
            sn = paddle.sin(t/2)
            m = paddle.concat([
                paddle.concat([cs, -sn], axis=1),
                paddle.concat([sn, cs], axis=1),
            ], axis=0)
            matrix = paddle.kron(matrix, m)

        x = paddle.matmul(matrix, x)
        return x

class CLayer_paddle(paddle.nn.Layer):
    def __init__(self, quantum_count, cnot_params, matrix=None):
        '''cnot_params和matrix不能同时为None'''
        super(CLayer_paddle, self).__init__()

        if cnot_params is None:
            raise ValueError('cnot_params不能为None')
        if matrix is None:
            c, t = cnot_params
            matrix = CLayer(c, t, quantum_count=quantum_count).matrix
        self.matrix = paddle.to_tensor(matrix, dtype='float32')
        self.quantum_count=quantum_count
        self.cnot_params = cnot_params
    
    def forward(self, x):
        return paddle.matmul(self.matrix, x)

def AutoLockRotate(model):
    first = True
    for layer in model.sublayers():
        if not first:
            if isinstance(layer, RLayer_paddle):
                for i,m in enumerate(mask):
                    if m:
                        layer.lock_theta(i)
            elif isinstance(layer, CLayer_paddle):
                mask = np.ones([layer.quantum_count])
                for i in layer.cnot_params:
                    mask[i]=0
        first = False

def GetQSystem(model):
    # TODO:
    layers_numpy = []
    for layer in model.sublayers():
        if isinstance(layer, RLayer_paddle):
            layers_numpy.append(RLayer([t.numpy()[0,0] for t in layer.thetas], quantum_count=layer.quantum_count))
        elif isinstance(layer, CLayer_paddle):
            layers_numpy.append(CLayer(layer.cnot_params[0], layer.cnot_params[1], quantum_count=layer.quantum_count))
    return QSystem(layers_numpy)

def ParseFromQSystem(qsystem, auto_lock_zero=True):
    quantum_count = qsystem.quantum_count
    layers = []
    for qlayer in qsystem.layer_list:
        if isinstance(qlayer, RLayer):
            layer = RLayer_paddle(quantum_count)
            # print('need to set:')
            # print(qlayer.thetas)
            layer.set_thetas(qlayer.thetas)
            if auto_lock_zero:
                for ti, t in enumerate(qlayer.thetas):
                    if abs(t)<EPS:
                        layer.lock_theta(ti)
        elif isinstance(qlayer, CLayer):
            c, t = qlayer.control_quantum, qlayer.target_quantum
            layer = CLayer_paddle(quantum_count=quantum_count, cnot_params=(c, t))
        layers.append(layer)
    return paddle.nn.Sequential(*layers)

def OptimizeModel(U, model, quantum_count, learning_rate=0.5, iterations=100, verbose=False):
    if verbose == True:
        verbose = 100
    size = 2**quantum_count
    I = paddle.to_tensor(np.eye(size, dtype=np.float32))
    U = paddle.to_tensor(U, dtype='float32')
    lr = paddle.optimizer.lr.PolynomialDecay(learning_rate, power=0.9, decay_steps=iterations, end_lr=learning_rate/2)
    # lr = paddle.optimizer.lr.LinearWarmup(learning_rate, iterations, 0, learning_rate)
    opt = paddle.optimizer.Adam(lr, 
        parameters=model.parameters(), 
        # weight_decay=paddle.regularizer.L1Decay(0.01),
        # grad_clip = paddle.nn.ClipGradByNorm(clip_norm=1.0),
        grad_clip = paddle.nn.ClipGradByValue(1.0),
    )
    for iter in range(iterations):
        U2 = model(I)
        #TODO: try paddle.diag
        loss = 1-paddle.sum(U*U2)/size
        fidelity = 1-loss.numpy()[0]
        loss.backward()
        opt.step()
        if isinstance(opt._learning_rate,
                        paddle.optimizer.lr.LRScheduler):
            opt._learning_rate.step()
        opt.clear_grad()
        if verbose and (iter+1)%verbose==0:
            print('iter_%d: fidelity=%g'%(iter, fidelity))
    return model, fidelity

def TryRemoveZeros(U, model, quantum_count, eval_func, try_epochs=100, learning_rate=0.1, iterations=150, iter_break=10):
    best_qsystem = qsystem = GetQSystem(model=model)
    cost = CostCompute(qsystem.string)
    fidelity = Fidelity(U, qsystem.matrix)
    best_score = eval_func(fidelity, cost)
    print('base score = %g(F=%g, cost=%d)'%(best_score, fidelity, cost))
    iter_no_change=0
    for epoch in range(try_epochs):
        # Find smallest R and lock it
        ids = []
        v = []
        for i, layer in enumerate(model.sublayers()):
            if isinstance(layer, RLayer_paddle):
                for j, t in enumerate(layer.thetas):
                    if t.stop_gradient:
                        continue
                    t = NormalizeAngle(t.numpy()[0,0])
                    t = min(t, 2*np.pi-t)
                    v.append(t)
                    ids.append((i, j))
        idx = np.argsort(v)[:5]
        sel_i = np.random.choice(idx)
        layer_id, theta_id = ids[sel_i]
        pre_theta = v[sel_i]
        model.sublayers()[layer_id].lock_theta(theta_id)
        # print('Model layer count:', len(model))
        # qsystem = GetQSystem(model=model)
        # print('QSystem layer count:', len(qsystem))
        # print('before cut:')
        # print(qsystem.string)
        model, fidelity = OptimizeModel(
            U=U,
            model=model, 
            quantum_count=quantum_count,
            learning_rate=learning_rate,
            iterations=iterations,
            verbose=False)
        # print('Model layer count:', len(model))
        qsystem = GetQSystem(model=model)
        # print('QSystem layer count:', len(qsystem))
        # print('after cut:')
        # print(qsystem.string)
        cost = CostCompute(qsystem.string)
        fidelity = Fidelity(U, qsystem.matrix)
        score = eval_func(fidelity, cost)
        print('try remove layer_%d, theta_%d, pre_theta=%g, score = %g(F=%g, cost=%d)'%(layer_id, theta_id, pre_theta, score, fidelity, cost))
        if best_score<score:
            best_score = score
            best_qsystem = qsystem
            iter_no_change = 0
        else:
            model.sublayers()[layer_id].unlock_theta(theta_id, pre_theta)
            iter_no_change += 1
            if iter_no_change > iter_break:
                break
    return ParseFromQSystem(best_qsystem), best_score

def BackwardParams(U, quantum_count, cnot_layers, learning_rate=0.5, iterations=100, verbose=False, auto_lock=True):
    layers = [RLayer_paddle(quantum_count)]
    for c, t in cnot_layers:
        layers.append(CLayer_paddle(quantum_count=quantum_count, cnot_params=(c, t)))
        layers.append(RLayer_paddle(quantum_count))
    model = paddle.nn.Sequential(*layers)
    AutoLockRotate(model)
    model, best_score = OptimizeModel(U, model, quantum_count, learning_rate=learning_rate, iterations=iterations, verbose=verbose)
    best_model = GetQSystem(model).string
    return best_score, best_model

def test_backward_params():
    from Reader import ReadU
    from utils import RandomCNOTs
    # U = np.array([[1, -1],[1, 1]], dtype=np.float32)/np.sqrt(2)
    U = ReadU('Questions/Question_4_Unitary.txt')
    quantum_count = 3
    best_score = 0
    np.random.seed(2021)
    # paddle.set_device('cpu')
    start_time = time.time()
    for epoch in range(1):
        # cnot_layers = RandomCNOTs(quantum_count, 6)
        cnot_layers = [(2,1),(2,1),(0,1),(1,2),(1,0),(1,2)]
        sc, model = BackwardParams(U, quantum_count, cnot_layers, iterations=100, verbose=10, learning_rate=0.5)
        if sc>best_score:
            best_score = sc
            best_model = model
        print('No_%d score: %g, best_score: %g, time: %gs'%(epoch, sc, best_score, time.time()-start_time))
    print(best_model)

def test_paddle_param():
    U = paddle.to_tensor(np.array([[1, -1],[1, 1]], dtype=np.float32)/np.sqrt(2))
    U.stop_gradient = True
    theta = paddle.static.create_parameter(shape=[1, 1], dtype='float32')
    cs = paddle.cos(theta/2)
    sn = paddle.sin(theta/2)
    matrix = paddle.concat([
        paddle.concat([cs, -sn], axis=1),
        paddle.concat([sn, cs], axis=1),
    ], axis=0)
    loss = 1-paddle.trace(paddle.matmul(U, paddle.transpose(matrix, [1,0])))/2
    # I = paddle.to_tensor(np.eye(2, dtype=np.float32))
    dt = paddle.grad(
            outputs=[loss],
            inputs=[theta],
            create_graph=False,
            retain_graph=True)[0]
    print(dt)
    # opt = paddle.optimizer.Adam(0.1, parameters=[theta])
    # for epoch in range(200):
    #     loss.backward(retain_graph=True)
    #     opt.step()
    #     theta.clear_grad()
    #     if (epoch+1)%20==0:
    #         print('epoch_%d: score=%g, theta=%g'%(epoch, 1-loss.numpy()[0], theta.numpy()[0]))
    #         print(matrix.numpy())

def test_paddle_backward():
    ## 必须每次重新构造矩阵
    I = paddle.to_tensor(np.eye(2, dtype=np.float32))
    U = paddle.to_tensor(np.array([[1, -1],[1, 1]], dtype=np.float32)/np.sqrt(2))
    layer = RLayer_d1()
    layer.set_theta(1.)
    U2 = layer(I)
    loss = 1-paddle.trace(paddle.matmul(U, paddle.transpose(U2, [1,0])))/2
    opt = paddle.optimizer.Adam(0.1, parameters=layer.parameters())
    for epoch in range(110):
        loss.backward(retain_graph=True)
        opt.step()
        opt.clear_grad()
        if (epoch+1)%10==0:
            print('epoch_%d: score=%g, theta=%g'%(epoch, 1-loss.numpy()[0], layer.theta.numpy()[0]))
            print(layer.matrix.numpy())

if __name__ == '__main__':
    # test_paddle_param()
    # test_paddle_backward()
    test_backward_params()
    pass
    

