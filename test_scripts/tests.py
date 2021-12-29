
import matplotlib.pyplot as plt

#import numpy as np
import cupy as np

from weirdneuralnet.network import WeirdNetwork
from weirdneuralnet.node_utils import *
from weirdneuralnet.datasets import *

def get_accuracy(model, X, Y):
    prediction_classes = model.predict(X, debinarize=True)
    num_correct = np.where(np.equal(prediction_classes, Y.argmax(axis=0)))[0].shape[0]
    return num_correct, len(prediction_classes)

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
        (0,2),
        (1,2)
    ]
    model = WeirdNetwork(node_params, edges)
    #print(f"calculating eval({X_test.shape}, {Y_test.shape})")
    #cost = model.evaluate(list(zip(X_test, Y_test)))
    #print(f"{cost.shape} = eval({X_test.shape}, {Y_test.shape})")

    cost_history = model.train(X_train, Y_train, epochs)

    #costs.append(model.evaluate(X_test, Y_test))
    #epoch_vals.append(i)
    print(f"test error: {model.evaluate(X_test, Y_test)}")
    correct, total = get_accuracy(model, X_test, Y_test)
    print(f"total samples: {total}\nnumber correct: {correct}\naccuracy: {correct/total}")
    # final_error = model.evaluate(X_val, Y_val)
    # print(f"validation: {final_error}")
    plt.plot(list(range(epochs)), cost_history)
    plt.show()
    # model.save("my_model.wn")
    return model
        

def load_test(fname):
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset('./datasets/')
    model = WeirdNetwork.load(fname)
    final_error = model.evaluate(X_test, Y_test)
    print(f"cost: {final_error}")
    correct, total = get_accuracy(model, X_test, Y_test)
    print(f"total samples: {total}\nnumber correct: {correct}\naccuracy: {correct/total}")
    return model

def equivalence_test(fname):
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset('./datasets/')
    model = WeirdNetwork.load(fname)
    weights, biases = model.compile()
    model_prediction = model.predict(X_train)
    classic_prediction = classic_net_predict(weights, biases, X_train)
    print('####################################')
    assert(np.array_equal(model_prediction, classic_prediction))
    #print(f"model output ({model_prediction.shape}): {model_prediction}")
    #print(f"classic output ({classic_prediction.shape}): {classic_prediction}")

    model_error = model.backpropagate(X_train, Y_train)
    classic_error = classic_net_backprop(weights, biases, X_train, Y_train)
    print(f"model error: {[(i, m.shape) for i,m in model_error[0].items()]}, {[(i,m.shape) for i,m in model_error[1].items()]}")
    print(f"class error: {[m.shape for m in classic_error[0]]}, {[m.shape for m in classic_error[1]]}")