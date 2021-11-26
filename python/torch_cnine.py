import torch
import cnine


class rtensor(torch.Tensor):

    @staticmethod
    def zeros(i0,i1):
        return rtensor(torch.zeros(i0,i1))
    
    #def __init__(self,x):
     #   super().__assign__(self,x) #.__init__(x)
    #def __init__(self,i0,i1):
     #   print('kljj2')
      #  self=torch.zeros(i0,i1)
