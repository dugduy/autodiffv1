from CGs import *
from CGs_grad import *


xs=[-4,-3,-2,-1,0,1,2,3,4,5]
ys=[-9,-7,-5,-3,-1,1,3,5,7,9]



Graph().as_default()


w=Variable(0,'w')
b=Variable(0,'b')



for i in range(500):
    for x_batch, y_batch in zip(xs,ys):
        x=PlaceHolder('x')
        y_pred=x*w+b
        loss=(y_pred-y_batch)**2
        sess=Session()
        print(i,'Loss:',sess.run(loss,{x:x_batch}))
        graded=compute_gradient(loss)
        w=Variable(w.value-graded.get(w)*0.1,'w')
        b=Variable(b.value-graded.get(b)*0.1,'b')

print(Session().run(x*w+b,{x:234}))