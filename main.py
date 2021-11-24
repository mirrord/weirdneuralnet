
import numpy as np
from network import WeirdNetwork

import os
import hashlib, requests, gzip

def run_test(epochs):
    #fetch data
    path='./datasets/'
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
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    #Validation split
    rand=np.arange(60000)
    np.random.shuffle(rand)
    train_no=rand[:50000]

    val_no=np.setdiff1d(rand,train_no)

    X_train,X_val=X[train_no,:,:],X[val_no,:,:]
    Y_train,Y_val=Y[train_no],Y[val_no]
    #reshape
    X_train = X_train.reshape((-1,28*28))
    X_val = X_val.reshape((-1,28*28))
    def binarize(y):
        targets = np.zeros((len(y),10), np.float32)
        targets[range(targets.shape[0]),y] = 1
        return targets
    Y_train, Y_val, Y_test = binarize(Y_train), binarize(Y_val), binarize(Y_test)

    training_data = list(zip(X_train, Y_train))

    #print(X_train.shape)

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
    cost = model._evaluate(list(zip(X_test, Y_test)))
    #print(f"{cost.shape} = eval({X_test.shape}, {Y_test.shape})")

    for i in range(epochs):
        print(f"epoch: {i}...")
        model.train(training_data)

        if i%5==0:
            cost = model._evaluate(list(zip(X_test, Y_test)))
            print(f"\tcost: {cost}")

    final_error = model._evaluate(list(zip(X_val, Y_val)))
    print(f"validation: {final_error}")
        


if __name__=="__main__":
    run_test(10)