

from .node import Node
from .node_utils import *
#import numpy as np
import cupy as np

from pickle import Pickler, Unpickler


class WeirdNetwork():
    '''A gradient descent NN of arbitrary topology.
    Methods
    --------
    load (classmethod)
        load a network instance from a saved model file.
    save
        save a network to a file.
    compile
        compile this network into a raw network with less overhead.
    predict
        compute forward propagation on some input.
    get_last_output
        retrieve the last-calculated output of this network.
    backpropagate
        calculate weight & bias updates based on input & expected output.
    evaluate
        compute the error cost of the network's prediction on some input & expected output.
    train
        train the network on some set of inputs and expected outputs for some number of epochs.'''
    def __init__(self,
                node_params,
                edges,
                error_cost_func="diff_squares",
                learning_rate=0.01,
                regularize=(None, 0)):
        '''Construct a WeirdNetwork NN.
        Inputs
            node_params
                A list containing a dict for each node in the network. Each dict
                must include at minimum: "x" (input dimension), "y" (output dimension),
                and "activation". 
            edges
                a list of (int,int) tuples that describe the topology of the network
                as a directed graph.
            error_cost_func
                a string label for the error cost function to be used.
            learning_rate
                the learning coefficient to be used during training.
            regularize
                a (str, int...) tuple that contains the string label for the
                regularization function to be used during training and its parameters.
        '''
        # node_params should be a list of dicts of node parameters
        # edges should be a list of node index tuples denoting connections between nodes
        self.nodes = []
        self.input_indices = []
        self.feed_indices = {}
        self.backfeed_indices = {}
        self.output_node = 0
        self.learning_rate = learning_rate
        self.input, self.output = None, None
        self.regularize_params, self.regularize = regularize, REGULARIZATIONS.get(regularize[0], noreg)(regularize[1])
        self.cost, self.cost_deriv = COSTS.get(error_cost_func, (diff_squares, ddiff_squares))
        for idx, param in enumerate(node_params):
            self.feed_indices[idx] = [edge[1] for edge in edges if edge[0]==idx]
            self.backfeed_indices[idx] = [edge[0] for edge in edges if edge[1]==idx]
            #print(f"backfeed idxs {idx}: {self.backfeed_indices[idx]}")
            self.nodes.append(Node(param['x'], param['y'], param['activation']))
            if param.get('output', False):
                self.output_node = idx
            if param.get('input', False):
                self.input_indices.append(idx)
        self.edges = edges


    def __str__(self):
        return f"<WeirdNetwork size={len(self.nodes)} nodes>"

    @classmethod
    def load(cls, fname:str):
        '''Load a WeirdNetwork model instance from a file.'''
        with open(fname, 'rb') as f:
            u = Unpickler(f)
            model = u.load()
        for node in model.nodes:
            node.reload()
        model.regularize = REGULARIZATIONS.get(model.regularize_params[0], noreg)(model.regularize_params[1])
        return model

    @classmethod
    def create_from_config(cls, json_config):
        return cls(
            json_config["node_params"],
            json_config["edges"],
            json_config.get("cost function", "diff_squares"),
            json_config.get("learning rate", 0.01),
            (json_config.get("regularization function", None),
                json_config.get("regularization parameter", 0))
        )

    def save(self, fname:str):
        '''Save a WeirdNetowrk model instance to a file.'''
        self.regularize = None
        for node in self.nodes:
            node.clear_history()
        with open(fname, 'wb') as f:
            p = Pickler(f)
            return p.dump(self)

    def compile(self):
        '''Build a raw NN from this object to remove the OO overhead.'''
        # only works for linear networks
        to_traverse = []
        to_traverse.extend(self.input_indices)
        for idx in self.input_indices:
            to_traverse.extend(self.feed_indices[idx])
        for idx in to_traverse:
            to_traverse.extend([fidx for fidx in self.feed_indices[idx] if fidx not in to_traverse])
        if self.output_node in to_traverse:
            weights = [self.nodes[nidx].weight for nidx in to_traverse]
            biases = [self.nodes[nidx].bias for nidx in to_traverse]
            return weights, biases
        raise Exception("Output node is not fed")

    def predict(self, input:np.array, debinarize=False):
        '''Calculate this model's prediction for some input.
        Inputs
            input
                the input vector. It must be of shape (features, samples).
            debinarize
                if set to True, return the output as a classifier integer
                instead of a vector.
        '''
        outputs = {}
        to_traverse = []
        #print(f"feed idxs: {self.feed_indices}")
        #print(f"backfeed idxs: {self.backfeed_indices}")
        # print(f"predicting... {input.shape}")
        for idx in self.input_indices:
            # print(f"feeding input node {idx}...")
            outputs[idx] = self.nodes[idx].feed(input)
            to_traverse.extend(self.feed_indices[idx])
        # print(f"initial propogation targets: {to_traverse}")
        for idx in to_traverse:
            if idx not in outputs:
                #find outputs this node wants as input
                inputs = [self.nodes[i].get_output() for i in self.backfeed_indices[idx]]
                #print(f"feeding node {idx} with outputs from: {[i for i in self.backfeed_indices[idx] if i in outputs]}")
                #"synapse" them together
                this_input = sum(inputs)
                outputs[idx] = self.nodes[idx].feed(this_input)
                to_traverse.extend([fidx for fidx in self.feed_indices[idx] if fidx not in outputs])
        if self.output_node in outputs:
            #print(f"taking output from output node {self.output_node}")
            #print(f"prediction output: {outputs[self.output_node].shape}")
            if debinarize:
                return outputs[self.output_node].argmax(axis=1)
            return outputs[self.output_node]
        raise Exception("Output node is not fed")

    def get_last_output(self):
        '''Retrieve the last-calculated model output.'''
        return self.nodes[self.output_node].output

    def backpropagate(self, input:np.array, exp_output:np.array):
        '''Calculate weight & bias updates using gradient descent.
        Inputs
            input
                vector on which to calculate error singal, shape (features, samples).
            exp_output
                vector to compare with model prediction, shape (binarized classes, samples).
        '''
        num_sample = input.shape[0]
        predicted_output = self.predict(input)
        # print(f"output: {predicted_output.shape}")
        backfeed = {}
        error_delta = self.cost_deriv(predicted_output, exp_output)
        # print(f"dE {error_delta.shape} = dC( {predicted_output.shape}, {exp_output.shape} )")
        # print(f"(backprop) before backfeed: {self.backfeed_indices}")
        backfeed[self.output_node] = self.nodes[self.output_node].backfeed(error_delta)
        # print(f"(backprop) after backfeed: {self.backfeed_indices}")
        to_traverse = self.backfeed_indices.get(self.output_node, []).copy()
        # print(f"(backprop) after to_traverse: {self.backfeed_indices}")
        for idx in to_traverse:
            error_signal_components = [backfeed.get(oidx, (0,0,0)) for oidx in self.feed_indices[idx]]
            de = sum([i[2] for i in error_signal_components]) #I think this actually requires a derivative of the synapse func
            db, dw, de = self.nodes[idx].backfeed(de)
            if idx not in backfeed:
                backfeed[idx] = db, dw, de
            else:
                backfeed[idx][0]+=db
                backfeed[idx][1]+=dw
                backfeed[idx][2]+=de

            for jdx in self.backfeed_indices[idx]:
                if jdx not in to_traverse:
                    to_traverse.append(jdx)

        bup = {i:np.sum(b[0],0,keepdims=True)/num_sample for i,b in backfeed.items()}
        wup = {i:w[1]/num_sample for i,w in backfeed.items()}
        return bup, wup

    def evaluate(self, input:np.array, exp_out:np.array):
        '''Calculate cost for some input & expected output.'''
        return self.cost(self.predict(input),exp_out)

    def train(self, input:np.array, exp_output:np.array, epochs:int):
        '''train the network on some set of inputs and expected outputs for some number of epochs.'''
        #TODO: adapt to use a batch size
        cost_history = []
        for i in range(epochs):
            print(f"epoch {i}...") #TODO: use progress bar instead
            shuffle_in_unison(input, exp_output)
            bup, wup = self.backpropagate(input, exp_output)
            cost_history.append(self.cost(self.get_last_output(), exp_output))
            #update
            for idx, node in enumerate(self.nodes):
                if idx in wup:
                    update_weights = self.learning_rate*wup[idx]
                    if self.regularize_params[0] is not None:
                        update_weights+=self.regularize(wup)
                    node.update(self.learning_rate*bup[idx], update_weights)

        return cost_history

#TODO still need to test this
# if it turns out this implementation is equivalent, I'll probably remove the OO
# implementation. 
class CompiledNetwork():
    '''a topology-agnostic gradient descent NN with lower overhead.'''
    def __init__(self,
                weights,
                biases,
                edges,
                activations,
                normalizations,
                input_idxs=[0],
                output_idx=-1):
        self.weights = weights
        self.biases = biases
        self.edges = edges
        self.activation_lables = activations
        self.activations = [ACTIVATIONS.get(act,(sigmoid, dsigmoid)) for act in activations]
        self.normalization_params = normalizations
        self.normalizations = [NORMALIZATIONS.get(norm, nonorm) for norm in normalizations]
        self.inputs = {}
        self.outputs = {}
        self.zs = {}
        self.feed_indices = [[edge[1] for edge in edges if edge[0]==idx] for idx in range(len(weights))]
        self.backfeed_indices = [[edge[0] for edge in edges if edge[1]==idx] for idx in range(len(weights))]
        self.input_nodes = input_idxs
        self.output_node = output_idx if output_idx is not -1 else len()

    def predict(self, input:np.array):
        '''Calculate this model's prediction for some input.
        Inputs
            input
                the input vector. It must be of shape (features, samples).
        '''
        outputs = {}
        to_traverse = []
        for idx in self.input_indices:
            self.inputs[idx] = input
            if self.normalize_label:
                self.inputs[idx] = self.normalize(input)
            self.zs[idx] = np.dot(self.weights[idx], self.inputs[idx]) + self.bias[idx]
            self.outputs[idx] = self.activations[idx][0](self.zs[idx])
            to_traverse.extend(self.feed_indices[idx])
        for idx in to_traverse:
            if idx not in outputs:
                #find outputs this node wants as input
                inputs = [self.nodes[i].get_output() for i in self.backfeed_indices[idx]]
                #"synapse" them together
                self.inputs[idx] = sum(inputs)
                if self.normalize_label:
                    self.inputs[idx] = self.normalize(self.inputs[idx])
                self.zs[idx] = np.dot(self.weights[idx], self.inputs[idx]) + self.bias[idx]
                self.outputs[idx] = self.activations[idx][0](self.zs[idx])
                to_traverse.extend([fidx for fidx in self.feed_indices[idx] if fidx not in outputs])
        if self.output_node in self.outputs:
            return self.outputs[self.output_node]
        raise Exception("Output node is not fed")
        
    def backprop(self, 
                input, 
                expout,
                derror_func=ddiff_squares,
                ):
        '''Calculate weight & bias updates using gradient descent.
        Inputs
            input
                vector on which to calculate error singal, shape (features, samples).
            exp_output
                vector to compare with model prediction, shape (binarized classes, samples).
        '''
        num_sample = input.shape[1]
        backfeed = {}
        de_dz = derror_func(self.predict(input), expout)
        backfeed[self.output_node][0] = de_dz * self.activations[self.output_node][1](self.zs[self.output_node])
        backfeed[self.output_node][1] = np.dot(backfeed[self.output_node][0], self.inputs[self.output_node].T)
        backfeed[self.output_node][2] = np.dot(self.weights[self.output_node].T, backfeed[self.output_node][0])
        to_traverse = self.backfeed_indices.get(self.output_node, []).copy()
        for idx in to_traverse:
            error_signal_components = [backfeed.get(oidx, (0,0,0)) for oidx in self.feed_indices[idx]]
            de = sum([i[2] for i in error_signal_components])
            db = de_dz * self.activations[idx][1](self.zs[idx])
            dw = np.dot(backfeed[idx][0], self.inputs[idx].T)
            de = np.dot(self.weights[idx].T, backfeed[idx][0])
            db, dw, de = self.nodes[idx].backfeed(de)
            if idx not in backfeed:
                backfeed[idx] = db, dw, de
            else:
                backfeed[idx][0]+=db
                backfeed[idx][1]+=dw
                backfeed[idx][2]+=de

            for jdx in self.backfeed_indices[idx]:
                if jdx not in to_traverse:
                    to_traverse.append(jdx)

        bup = {i:np.sum(b[0],1,keepdims=True)/num_sample for i,b in backfeed.items()}
        wup = {i:w[1]/num_sample for i,w in backfeed.items()}
        return bup, wup