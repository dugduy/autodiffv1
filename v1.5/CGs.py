import numpy as np

class Graph:
    def __init__(self,name='MyCGs') -> None:
        self.name=name
        self.ops=[]
        self.phs=[]
        self.vars=[]
    def as_default(self):
        global _default_graph
        _default_graph=self


class Operation:
    def __init__(self,input_nodes=[],name='') -> None:
        self.input_nodes=input_nodes
        self.consumers=[]
        for input_node in self.input_nodes:
            input_node.consumers.append(self)
        self.name=name
        _default_graph.ops.append(self)
    
    def compute(self):
        pass

    def __add__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):
            other=Variable(other,'constant')
        return add(self,other)
    def __radd__(self,other):
        return self+other
    def __sub__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):
            other=Variable(other,'constant')
        return sub(self,other)
    def __rsub__(self,other):
        return -self+other
    def __mul__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):
            other=Variable(other,'constant')
        return mul(self,other)
    def __rmul__(self,other):
        return self*other
    def __truediv__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):
            other=Variable(other,'constant')
        return div(self,other)
    def __rtruediv__(self,other):
        return div(Variable(other),self)
    def __pow__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):
            other=Variable(other,'constant')
        return pow(self,other)
    def __neg__(self):
        return negative(self)
    
    def __str__(self) -> str:
        return str(self.output)

class negative(Operation):
    def __init__(self, a, name='') -> None:
        super().__init__([a], name)
    def compute(self,a_val):
        return -a_val

class add(Operation):
    def __init__(self, a,b, name='') -> None:
        super().__init__([a,b], name)
    def compute(self,x_val,y_val):
        return x_val+y_val

class sub(add):
    def compute(self, x_val, y_val):
        return x_val-y_val

class mul(add):
    def compute(self, x_val, y_val):
        return x_val*y_val

class div(add):
    def compute(self, x_val, y_val):
        return x_val/y_val

class pow(add):
    def compute(self, x_val, y_val):
        return x_val**y_val


class PlaceHolder:
    def __init__(self,name='') -> None:
        self.consumers=[]
        self.name=name
        _default_graph.phs.append(self)
    
    def __add__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):
            other=Variable(other,'constant')
        return add(self,other)
    def __radd__(self,other):
        return self+other
    def __sub__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):
            other=Variable(other,'constant')
        return sub(self,other)
    def __rsub__(self,other):
        return -self+other
    def __mul__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):
            other=Variable(other,'constant')
        return mul(self,other)
    def __rmul__(self,other):
        return self*other
    def __truediv__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):
            other=Variable(other,'constant')
        return div(self,other)
    def __rtruediv__(self,other):
        return div(Variable(other),self)
    def __pow__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):
            other=Variable(other,'constant')
        return pow(self,other)
    def __neg__(self):
        return negative(self)
    
    def __str__(self) -> str:
        return str(self.output)

class Variable:
    def __init__(self,init_val=None,name='') -> None:
        self.value=init_val
        self.name=name
        self.consumers=[]
        _default_graph.vars.append(self)

    def __add__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):
            other=Variable(other,'constant')
        return add(self,other)
    def __radd__(self,other):
        return self+other
    def __sub__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):
            other=Variable(other,'constant')
        return sub(self,other)
    def __rsub__(self,other):
        return -self+other
    def __mul__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):
            other=Variable(other,'constant')
        return mul(self,other)
    def __rmul__(self,other):
        return self*other
    def __truediv__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):
            other=Variable(other,'constant')
        return div(self,other)
    def __rtruediv__(self,other):
        return div(Variable(other),self)
    def __pow__(self,other):
        if not (type(other) in [PlaceHolder,Variable] or isinstance(other,Operation)):
            other=Variable(other,'constant')
        return pow(self,other)
    def __neg__(self):
        return negative(self)
    
    def __str__(self) -> str:
        return str(self.output)


def traverse_postorder(op):
    nodes_postorder=[]
    def recurse(node):
        if isinstance(node,Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)
    recurse(op)
    return nodes_postorder
class Session:
    def run(self,op,feed_dict={}):
        self.nodes_postorder=traverse_postorder(op)
        for node in self.nodes_postorder:
            if type(node)==PlaceHolder:
                node.output=feed_dict[node]
            elif type(node)==Variable:
                node.output=node.value
            else:
                node.inputs=[i.output for i in node.input_nodes]
                node.output=node.compute(*node.inputs)
        return op.output


# Graph().as_default()

# x=Variable(5,'x')
# y=add(x,Variable(4,'constant0'),'y')
# z=mul(y,x,'z')

# sess=Session()
# print(sess.run(z))
# print([i.name for i in sess.nodes_postorder])