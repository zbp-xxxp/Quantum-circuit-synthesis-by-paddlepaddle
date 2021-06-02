import time
import numpy as np
from tqdm import tqdm

from utils import RandomCNOT, RandomCNOTs

def SimulatedAnnealing(quantum_count, layer_count, solver, epochs=100, save_path=None, global_best_score=0):
    #TODO:
    best_score = 0
    cnot = RandomCNOTs(quantum_count, layer_count)
    sc, model = solver(cnot)
    if sc>best_score:
        best_score = sc
        cnot_seed = cnot
        best_model = model
        best_cnot = cnot
    if save_path is not None and best_score>global_best_score:
        with open(save_path, 'w') as f:
            f.write(best_model)
    start_time = time.time()
    for epoch in range(epochs):
        for i in range(layer_count):
            cnot_layers = cnot_seed.copy()
            cnot_layers[i] = RandomCNOT(quantum_count)
            sc, model = solver(cnot_layers)
            if sc>best_score or np.random.randint(epochs)>epoch:
                cnot_seed = cnot_layers
                if sc>best_score:
                    best_score = sc
                    best_model = model
                    best_cnot = cnot_layers
                    if save_path is not None and best_score>global_best_score:
                        with open(save_path, 'w') as f:
                            f.write(best_model)
            print('epoch %d, iter %d, Score = %g, best_score = %g, global_best_score = %g, time = %gs'%(epoch, i, sc, best_score, global_best_score, time.time()-start_time))
    # print(best_model)
    return best_score, best_model, best_cnot


def SequenceJitter(quantum_count, layer_count, solver, init_epochs=10, epochs=100, save_path=None, global_best_score=0):
    #TODO:
    best_score = 0
    print('Init cnot seed.')
    for _ in tqdm(range(init_epochs)):
        cnot = RandomCNOTs(quantum_count, layer_count)
        sc, model = solver(cnot)
        if sc>best_score:
            best_score = sc
            cnot_seed = cnot
            best_model = model
            best_cnot = cnot
        if save_path is not None and best_score>global_best_score:
            with open(save_path, 'w') as f:
                f.write(best_model)
    start_time = time.time()
    for epoch in range(epochs):
        for i in range(layer_count):
            cnot_layers = cnot_seed.copy()
            cnot_layers[i] = RandomCNOT(quantum_count)
            sc, model = solver(cnot_layers)
            if sc>best_score:
                best_score = sc
                cnot_seed = cnot_layers
                best_model = model
                best_cnot = cnot_layers
                if save_path is not None and best_score>global_best_score:
                    with open(save_path, 'w') as f:
                        f.write(best_model)
            print('Score = %g, best_score = %g, global_best_score = %g, time = %gs'%(sc, best_score, global_best_score, time.time()-start_time))
    # print(best_model)
    return best_score, best_model, best_cnot

def RandomSearch(cnot_creater, solver, epochs=100, save_path=None):
    '''
    随机搜索
    Parameters:
        cnot_creater: 生成CNOT层的可执行对象
        solver: 一个可执行对象，给定网络结构后，求解网络参数的求解器
        epochs: 随机搜索的轮数
        save_path: 保存最佳结果的路径
    '''
    best_score = 0
    start_time = time.time()
    for epoch in range(epochs):
        cnot_layers = cnot_creater()
        sc, model = solver(cnot_layers)
        if sc>best_score:
            best_score = sc
            best_model = model
            if save_path is not None:
                with open(save_path, 'w') as f:
                    f.write(best_model)
        print('No_%d: score = %g, best_score = %g, time = %gs'%(epoch, sc, best_score, time.time()-start_time))
    # print(best_model)
    return best_score, best_model
