
import numpy as np
from network import WeirdNetwork

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
        return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

    X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).T
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    #Validation split
    rand=np.arange(60000)
    np.random.shuffle(rand)
    train_no=rand[:50000]

    val_no=np.setdiff1d(rand,train_no)

    X_train,X_val=X[train_no,:,:],X[val_no,:,:]
    Y_train,Y_val=Y[train_no],Y[val_no]
    #reshape
    X_train = X_train.reshape((-1,28*28)).T
    X_val = X_val.reshape((-1,28*28)).T
    def binarize(y):
        targets = np.zeros((len(y),10), np.float32)
        targets[range(targets.shape[0]),y] = 1
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
        model.train(X_train, Y_train)

        if i%5==0:
            costs.append(model.evaluate(X_test, Y_test))
            epoch_vals.append(i)

    costs.append(model.evaluate(X_test, Y_test))
    epoch_vals.append(i)
    print(f"final cost: {costs[-1]}")
    final_error = model.evaluate(X_val, Y_val)
    print(f"validation: {final_error}")
    plt.plot(epoch_vals, costs)
    plt.show()
    model.save("my_model.wn")
    return model
        

def load_test(fname):
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset('./datasets/')
    model = WeirdNetwork.load(fname)
    final_error = model.evaluate(X_test, Y_test)
    print(f"cost: {final_error}")
    return model


if __name__=="__main__":
    model = run_test(10)
    load_model = load_test("my_model.wn")
    