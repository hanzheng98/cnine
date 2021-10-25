import numpy
import torch
import cnine

a=cnine.cscalar()
print(a.str())

dims=cnine.gdims([4,4])
print(dims.str())

#T=cnine.ctensor(dims,cnine.fill_zero())
T=cnine.ctensor.sequential(dims,-1,0)
print(T)
print(T.get_k())
print(T.get_dims())
print(T.get_dim(0))
print(T.get(1,2))

T.set_value(1,2,99)
print(T)

print(T.transp())

#print(T[1,2])

