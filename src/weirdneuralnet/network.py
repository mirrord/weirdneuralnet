

from .node import Node
from .node_utils import *
#import numpy as np
import cupy as np

from pickle import Pickler, Unpickler


class WeirdNetwork():
    '''A gradient-descent NN of arbitrary topology.'''
    def __init__(self,
                node_params,
                edges,
                error_cost_func="diff_squares",
                learning_rate=0.01,
                regularize=(None, 0)):
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

    #TODO: this is insecure and doesn't really work.
    #   implement a weirdnet-specific load & save.
    @classmethod
    def load(cls, fname):
        with open(fname, 'rb') as f:
            u = Unpickler(f)
            model = u.load()
        for node in model.nodes:
            node.reload()
        model.regularize = REGULARIZATIONS.get(model.regularize_params[0], noreg)(model.regularize_params[1])
        return model

    def save(self, fname, keep_history=False):
        if not keep_history:
            self.regularize = None
            for node in self.nodes:
                node.clear_history()
        with open(fname, 'wb') as f:
            p = Pickler(f)
            return p.dump(self)

    def compile(self):
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

    def predict(self, input, debinarize=False):
        '''input shape must be (features, samples)'''
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
                return outputs[self.output_node].argmax(axis=0)
            return outputs[self.output_node]
        raise Exception("Output node is not fed")

    def get_last_output(self):
        return self.nodes[self.output_node].output

    ##TODO: backprop, train, minibatch train, etc
    def backpropagate(self, input, exp_output):
        num_sample = input.shape[1]
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

        bup = {i:np.sum(b[0],1,keepdims=True)/num_sample for i,b in backfeed.items()}
        wup = {i:w[1]/num_sample for i,w in backfeed.items()}
        return bup, wup

    def evaluate(self, input, exp_out):
        return self.cost(self.predict(input),exp_out)

    def train(self, input, exp_output, epochs):
        #TODO: adapt to use a batch size
        cost_history = []
        for i in range(epochs):
            print(f"epoch {i}...") #TODO: use progress bar instead
            shuffle_in_unison(input.T, exp_output.T)
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
