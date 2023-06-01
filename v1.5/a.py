from CGs import *
from CGs_grad import *

Graph().as_default()

x=Variable(4,'x')
y=x*3
z=x+Variable(2,'asd')


# print([i.name for i in traverse_postorder(z)])
sess=Session()
print(sess.run(z))

graded=compute_gradient(z)
print(graded.get(x))