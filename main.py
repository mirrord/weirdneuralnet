
#import numpy as np
import cupy as np
from numpy import frombuffer, setdiff1d

from network import WeirdNetwork
import node_utils

import os
import hashlib, requests, gzip

import matplotlib.pyplot as plt

def get_dataset(path):
    def fetch(url):
        fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                data = f.read()
        else:
            with open(fp, "wb") as f:
                data = requests.get(url).content
                f.write(data)
        #return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy())
        return np.array(frombuffer(gzip.decompress(data), dtype=np.uint8).copy())

    X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).T
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    #Validation split
    rand=np.arange(60000)
    np.random.shuffle(rand)
    train_no=rand[:50000]

    #val_no=np.setdiff1d(rand,train_no)
    val_no=np.array(setdiff1d(np.asnumpy(rand),np.asnumpy(train_no)))

    X_train,X_val=X[train_no,:,:],X[val_no,:,:]
    Y_train,Y_val=Y[train_no],Y[val_no]
    #reshape
    X_train = X_train.reshape((-1,28*28)).T
    X_val = X_val.reshape((-1,28*28)).T
    # def binarize(y):
    #     targets = np.zeros((len(y),10), np.float32)
    #     targets[range(targets.shape[0]),y] = 1
    #     return targets
    def binarize(y):
        targets = np.zeros((len(y),10), np.float32)
        for i in range(targets.shape[0]):
            targets[i][y[i]] = 1
        return targets
    Y_train, Y_val, Y_test = binarize(Y_train).T, binarize(Y_val).T, binarize(Y_test).T
    return X_train, Y_train, X_test, Y_test, X_val, Y_val


def run_test(epochs):
    #fetch data
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset('./datasets/')

    #build network
    node_params =[
        {
            'x':28*28,
            'y':128,
            'activation': 'sigmoid',
            'input':True
        },
        {
            'x':128,
            'y':128,
            'activation': 'sigmoid',
        },
        {
            'x':128,
            'y':10,
            'activation': 'sigmoid',
            'output':True
        }
    ]
    edges = [
        (0,1),
        (1,2)
    ]
    model = WeirdNetwork(node_params, edges)
    #print(f"calculating eval({X_test.shape}, {Y_test.shape})")
    #cost = model.evaluate(list(zip(X_test, Y_test)))
    #print(f"{cost.shape} = eval({X_test.shape}, {Y_test.shape})")

    costs = []
    epoch_vals = []

    for i in range(epochs):
        print(f"epoch: {i}...")
        costs.append(model.train(X_train, Y_train))

        # if i%5==0:
        #     costs.append(model.evaluate(X_test, Y_test))
        #     epoch_vals.append(i)

    #costs.append(model.evaluate(X_test, Y_test))
    #epoch_vals.append(i)
    print(f"test error: {model.evaluate(X_test, Y_test)}")
    # final_error = model.evaluate(X_val, Y_val)
    # print(f"validation: {final_error}")
    #plt.plot(epoch_vals, costs)
    plt.plot(list(range(epochs)), costs)
    plt.show()
    model.save("my_model.wn")
    return model
        

def load_test(fname):
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset('./datasets/')
    model = WeirdNetwork.load(fname)
    final_error = model.evaluate(X_test, Y_test)
    print(f"cost: {final_error}")
    return model

def equivalence_test(fname):
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset('./datasets/')
    model = WeirdNetwork.load(fname)
    weights, biases = model.compile()
    model_prediction = model.predict(X_train)
    classic_prediction = node_utils.classic_net_predict(weights, biases, X_train)
    print('####################################')
    assert(np.array_equal(model_prediction, classic_prediction))
    #print(f"model output ({model_prediction.shape}): {model_prediction}")
    #print(f"classic output ({classic_prediction.shape}): {classic_prediction}")

    model_error = model.backpropagate(X_train, Y_train)
    classic_error = node_utils.classic_net_backprop(weights, biases, X_train, Y_train)
    print(f"model error: {[(i, m.shape) for i,m in model_error[0].items()]}, {[(i,m.shape) for i,m in model_error[1].items()]}")
    print(f"class error: {[m.shape for m in classic_error[0]]}, {[m.shape for m in classic_error[1]]}")

if __name__=="__main__":
    model = run_test(100)
    #load_model = load_test("my_model.wn")
    #equivalence_test("my_model.wn")