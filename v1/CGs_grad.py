from queue import Queue

import numpy as np
import CGs

_gradient_registry={}

class RegGrad:
    def __init__(self,op_name) -> None:
        self._op_type=eval(op_name)
    def __call__(self,f):
        _gradient_registry[self._op_type]=f
        # return f

def compute_gradients(loss):
    grad_table={loss:1}

    # visited={loss}
    queue=Queue()
    queue.put(loss)

    while not queue.empty():
        node=queue.get()

        if node!=loss:
            grad_table[node]=0
            for consumer in node.consumers:
                lossgrad_wrt_consumer_output=grad_table.get(consumer,0)
                bprop=_gradient_registry[consumer.__class__]
                lossgrad_wrt_consumer_input=bprop(consumer,lossgrad_wrt_consumer_output)
                if len(consumer.input_nodes)==1:
                    grad_table[node]+=lossgrad_wrt_consumer_input
                else:
                    lossgrad_node=lossgrad_wrt_consumer_input[consumer.input_nodes.index(node)]
                    grad_table[node]+=lossgrad_node

        
        if hasattr(node,'input_nodes'):
            for input_node in node.input_nodes:
                # if not input_node in visited:
                    # visited.add(input_node)
                    queue.put(input_node)
    return grad_table

@RegGrad('CGs.negative')
def _negative_gradient(op,grad):
    return -grad

@RegGrad('CGs.log')
def _log_gradient(op,grad):
    x=op.inputs[0]
    return grad/x

@RegGrad('CGs.sigmoid')
def _sigmoid_gradient(op,grad):
    sigmoid=op.output
    return grad*sigmoid*(1-sigmoid)

@RegGrad('CGs.mul')
def _mul_gradient(op,grad):
    A,B=op.inputs
    return [grad*B,grad*A]

@RegGrad('CGs.truediv')
def _div_gradient(op,grad):
    a,b=op.inputs
    return [grad/b,grad/a]

@RegGrad('CGs.pow')
def _pow_gradient(op,grad):
    x,y=op.inputs
    return [grad*y*x**(y-1),grad*op.output*np.log(x)]

@RegGrad('CGs.matmul')
def _matmul_gradient(op,grad):
    A,B=op.inputs
    return [grad@B.T,A.T@grad]

@RegGrad('CGs.add')
def _add_gradient(op,grad):
    # a,b=op.output
    return [grad,grad]

@RegGrad('CGs.reduce_sum')
def reduce_sum(op,grad):
    pass

@RegGrad('CGs.softmax')
def softmax_gradient(op,grad):
    pass
