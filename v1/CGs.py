import numpy as np

class Graph:
    def __init__(self,name='MyCGs') -> None:
        self.name=name
        self.operations=[]
        self.placeholders=[]
        self.variables=[]
    def as_default(self):
        global _default_graph
        _default_graph=self

class Operation:
    def __init__(self,input_nodes=[],name='') -> None:
        self.input_nodes=input_nodes
        self.consumers=[]
        for input_node in input_nodes:
            input_node.consumers.append(self)
        self.name=name
        _default_graph.operations.append(self)
    def compute(self):
        pass
    def __add__(self,other):
        return add(self,other,'after plus '+other.name)
    def __sub__(self,other):
        return add(self,-other,'after subtract '+other.name)
    def __mul__(self,other):
        return mul(self,other,'after multiply '+other.name)
    def __neg__(self):
        return negative(self)
    def __truediv__(self,other):
        return truediv(self,other,'after divide '+other.name)
    def __pow__(self,other):
        return pow(self,other,'after power '+other.name)

class add(Operation):
    def __init__(self, x,y, name='') -> None:
        super().__init__([x,y], name)
    def compute(self,x_val,y_val):
        return x_val+y_val

class mul(add):
    def compute(self, x_val, y_val):
        return x_val*y_val

class truediv(add):
    def compute(self, x_val, y_val):
        return x_val/y_val

class matmul(add):
    def compute(self, x_val, y_val):
        return x_val@y_val
    
class pow(add):
    def compute(self, x_val, y_val):
        return x_val**y_val
    
class sigmoid(Operation):
    def __init__(self, a, name='') -> None:
        super().__init__([a], name)
    def compute(self,a_val):
        return 1/(1+np.exp(-a_val))

class softmax(sigmoid):
    def compute(self, a_val):
        return np.exp(a_val)/np.sum(np.exp(a_val),1)[:,None]

class log(sigmoid):
    def compute(self, x_val):
        return np.log(x_val)

class reduce_sum(Operation):
    def __init__(self, A,axis=None, name='') -> None:
        super().__init__([A], name)
        self.axis=axis
    def compute(self,A_val):
        return np.sum(A_val,self.axis)

class negative(sigmoid):
    def compute(self, x_val):
        return -x_val

class PlaceHolder:
    def __init__(self,name='x') -> None:
        self.name=name
        self.consumers=[]
        _default_graph.placeholders.append(self)

class Variable:
    def __init__(self,init_val=None,name='') -> None:
        self.name=name
        self.value=init_val
        self.consumers=[]
        _default_graph.variables.append(self)
    
    def __add__(self,other):
        return add(self,other,'after plus '+other.name)
    def __sub__(self,other):
        return add(self,-other,'after subtract '+other.name)
    def __neg__(self):
        return negative(self)
    def __mul__(self,other):
        return mul(self,other,'after multiply '+other.name)
    def __truediv__(self,other):
        return truediv(self,other,'after divide '+other.name)
    def __pow__(self,other):
        return pow(self,other,'after power '+other.name)

def traverse_postoder(operation):
    nodes_postorder=[]
    def recurse(node):
        if isinstance(node,Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)
    recurse(operation)
    return nodes_postorder

class Session:
    def run(self,operation,feed_dict={}):
        self.nodes_postorder=traverse_postoder(operation)

        for node in self.nodes_postorder:
            if type(node)==PlaceHolder:
                node.output=feed_dict[node]
            elif type(node)==Variable:
                node.output=node.value
            else: #Operation
                node.inputs=[input_node.output for input_node in node.input_nodes]
                node.output=node.compute(*node.inputs)
            
            if type(node.output)==list:
                node.output=np.array(node.output)
        return operation.output

# Graph().as_default()

# A=Variable([
#     [1,0],
#     [0,-1]
# ],'A')
# x=PlaceHolder()
# b=Variable([1,1],'b')

# y=matmul(A,x,'y')
# z=add(b,y,'z')

# print([i.name for i in traverse_postoder(z)])

# sess=Session()
# print(sess.run(z,{x:[1,2]}))