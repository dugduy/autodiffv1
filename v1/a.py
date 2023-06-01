from CGs import *
from CGs_grad import *

Graph().as_default()

x=Variable(6,'a')
y=x+x*Variable(2,'constant1')

sess=Session()
print(sess.run(y))
# print([i.name for i in sess.nodes_postorder])

graded=compute_gradients(y)
# print([(i.name,graded[i]) for i in graded])
print(graded.get(x))