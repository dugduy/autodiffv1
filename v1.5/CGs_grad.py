from queue import Queue
import CGs

_gradient_registry={}

class RegGrad:
    def __init__(self,op_type) -> None:
        self._op_type=eval(op_type)
    def __call__(self,f):
        _gradient_registry[self._op_type]=f
        # return f

@RegGrad('CGs.negative')
def _negative_gradient(op,grad):
    return -grad

@RegGrad('CGs.add')
def _add_gradient(op,grad):
    return grad,grad

@RegGrad('CGs.sub')
def _sub_gradient(op,grad):
    return grad,-grad

@RegGrad('CGs.mul')
def _mul_gradient(op,grad):
    a,b=op.inputs
    return [grad*b,grad*a]

@RegGrad('CGs.div')
def _div_gradient(op,grad):
    a,b=op.inputs
    return [grad/b,grad*-a/b**2]

@RegGrad('CGs.pow')
def _pow_gradient(op,grad):
    a,b=op.inputs
    # return [grad*b*a**(b-1),grad*op.output*CGs.np.log(a)]
    return [grad*b*a**(b-1),1]

def compute_gradient(op):
    grad_table={op:1}

    # visited={op}
    q=Queue()
    q.put(op)
    node_postorder=CGs.traverse_postorder(op)

    while not q.empty():
        node = q.get()
        # print(node.name)
        if node != op:
            grad_table[node]=0

            for consumer in node.consumers:
                # print(consumer.name)
                if not consumer in node_postorder:
                    continue
                lossgrad_wrt_consumer_input=_gradient_registry[consumer.__class__](consumer,grad_table.get(consumer,0))
                if len(consumer.input_nodes)==1:
                    grad_table[node]+=lossgrad_wrt_consumer_input
                else:
                    grad_table[node]+=lossgrad_wrt_consumer_input[consumer.input_nodes.index(node)]
        
        if hasattr(node,'input_nodes'):
            for input_node in node.input_nodes:
                # if not input_node in visited:
                    # visited.add(input_node)
                    q.put(input_node)
    
    return grad_table