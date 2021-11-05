*******
Tensors
*******


The core data type in cnine are real and complex multidimensional arrays. 
The corresponding classes are ``rtensor`` and ``ctensor``. 
Currently, to facilitate GP-Gpu operations only single precision arithmetic is supported. 

We describe basic tensor functionality for the ``rtensor`` class.  
The ``ctensor`` class's interface is analogous. 


================
Creating tensors
================


The following example shows how to create and print out a :math:`4\times 4` 
dimensional tensor filled with zeros.

.. code-block:: python

  >>> A=rtensor.zero([4,4])
  >>> print(A)
  [ 0 0 0 0 ]
  [ 0 0 0 0 ]
  [ 0 0 0 0 ]
  [ 0 0 0 0 ]

Alternative fill patters are ``ones`` (a tensor filled with 1s), ``sequential`` (a tensor filled 
with the numbers 0,1,2,... in sequence, ``identity`` (used to construct an identity matrix), 
and ``gaussian`` (a tensor whose elements are drawn i.i.d. from a standard normal distribution). 

The number of tensor dimensions and the dimensions themselves can be read out as follows.

.. code-block:: python

  >>> A=rtensor.zero([3,4,4])
  >>> G.get_ndims() 
  3
  >>> print(A.get_dims())
  (3,4,4)
  >>> print(A.get_dim(2))
  4


=========================
Accessing tensor elements
=========================



Tensor indexing in `cnine` always starts with 0 along each dimension. 
The following example shows how tensor elements can be set and read. 

.. code-block:: python

  >>> from cnine import *
  >>> A=rtensor.sequential([4,4])
  >>> print(A)
  [ 0 1 2 3 ]
  [ 4 5 6 7 ]
  [ 8 9 10 11 ]
  [ 12 13 14 15 ]

  >>> A([1,2])
  6.0
  >>> A[[1,2]]=99
  >>> print(A)
  [ 0 1 2 3 ]
  [ 4 5 99 7 ]
  [ 8 9 10 11 ]
  [ 12 13 14 15 ]


=====================
Arithmetic operations
=====================

``rtensor`` supports the usual arithmetic tensor and matrix operations.

.. code-block:: python

  >>> A=rtensor.sequential([4,4])
  >>> B=rtensor.ones([4,4])
  >>> print(A+B)
  [ 1 2 3 4 ]
  [ 5 6 7 8 ]
  [ 9 10 11 12 ]
  [ 13 14 15 16 ]
  >>> print(A*5)
  [ 0 5 10 15 ]
  [ 20 25 30 35 ]
  [ 40 45 50 55 ]
  [ 60 65 70 75 ]
  >>> print(A*A)
  [ 56 62 68 74 ]
  [ 152 174 196 218 ]
  [ 248 286 324 362 ]
  [ 344 398 452 506 ]

``rtensor`` also offers in-place operators.

.. code-block:: python

  >>> B=rtensor.ones([4,4])
  >>> A=rtensor.sequential([4,4])
  >>> A+=B
  >>> print(A)
  [ 1 2 3 4 ]
  [ 5 6 7 8 ]
  [ 9 10 11 12 ]
  [ 13 14 15 16 ]

  >>> A-=B
  >>> print(A)
  [ 0 1 2 3 ]
  [ 4 5 6 7 ]
  [ 8 9 10 11 ]
  [ 12 13 14 15 ]


====================
Functions of tensors
====================

The following shows how to compute the inner product 
:math:`\langle A, B\rangle=\sum_{i_1,\ldots,i_k} A_{i_1,\ldots,i_k} B_{i_1,\ldots,i_k}` 
between two tensors and the squared Frobenius norm 
:math:`\vert A\vert^2=\sum_{i_1,\ldots,i_k} \vert A_{i_1,\ldots,i_k}\vert^2`.

.. code-block:: python

  >>> A=rtensor.gaussian([4,4])
  >>> print(A)
  [ -1.23974 -0.407472 1.61201 0.399771 ]
  [ 1.3828 0.0523187 -0.904146 1.87065 ]
  [ -1.66043 -0.688081 0.0757219 1.47339 ]
  [ 0.097221 -0.89237 -0.228782 1.16493 ]
  >>> B=rtensor.ones([4,4])
  >>> inp(A,B)
  2.107801675796509
  >>> norm2(A)
  18.315340042114258


The ``ReLU`` function applies the function :math:`\textrm{ReLU}(x)=\textrm{max}(0,x)` to 
each element of the tensor.

.. code-block:: python

  >>> print(ReLU(A))
  [ 0 0 1.61201 0.399771 ]
  [ 1.3828 0.0523187 0 1.87065 ]
  [ 0 0 0.0757219 1.47339 ]
  [ 0.097221 0 0 1.16493 ]


====================
Slices and reshaping
====================

The ``slice(i,c)`` method returns the slice of the tensor corresponding to setting the i'th index 
equal to c. ``reshape`` reinterprets the tensor as a tensor of a different shape.

.. code-block:: python

  >>> A=rtensor.sequential([4,4])
  >>> print(A.slice(1,2))
  [ 2 6 10 14 ]

  >>> A.reshape([2,8])
  >>> print(A)
  [ 0 1 2 3 4 5 6 7 ]
  [ 8 9 10 11 12 13 14 15 ]


==========
Transposes
==========

The ``transp`` method returns the transpose of a matrix.

.. code-block:: python

  >>> A=rtensor.sequential([4,4])
  >>> print(A.transp())
  [ 0 4 8 12 ]
  [ 1 5 9 13 ]
  [ 2 6 10 14 ]
  [ 3 7 11 15 ]


================
gdims and gindex
================

In the previous examples tensors dimensions and tensor indices were given simply as lists.  
As an alternative, tensor dimensions and indices can also be specified using the specialized 
classes `gdims` and `gindex`. 

.. code-block:: python

   >>> dims=gdims([3,3,5])
   >>> print(dims)
   (3,3,5)
   >>> print(len(dims))
   >>> print(dims[2])
   5
   >>> dims[2]=7
   >>> print(dims)
   (3,3,7)
   >>> 

===============
Complex tensors
===============


The ``ctensor`` complex valued tensor class supports all the above operations. 
In addition, it also has ``conj`` and ``herm`` methods to take the conjugate 
and conjugate transpose (Hermitian conjugate) of the tensor.

.. code-block:: python

  >>> A=ctensor.gaussian([4,4])
  >>> print(A)
  [ (-1.23974,0.584898) (-0.407472,-0.660558) (1.61201,0.534755) (0.399771,-0.607787) ]
  [ (1.3828,0.74589) (0.0523187,-1.75177) (-0.904146,-0.965146) (1.87065,-0.474282) ]
  [ (-1.66043,-0.546571) (-0.688081,-0.0384917) (0.0757219,0.194947) (1.47339,-0.485144) ]
  [ (0.097221,-0.370271) (-0.89237,-1.12408) (-0.228782,1.73664) (1.16493,0.882195) ]

  >>> print(A.conj())
  [ (-1.23974,-0.584898) (-0.407472,0.660558) (1.61201,-0.534755) (0.399771,0.607787) ]
  [ (1.3828,-0.74589) (0.0523187,1.75177) (-0.904146,0.965146) (1.87065,0.474282) ]
  [ (-1.66043,0.546571) (-0.688081,0.0384917) (0.0757219,-0.194947) (1.47339,0.485144) ]
  [ (0.097221,0.370271) (-0.89237,1.12408) (-0.228782,-1.73664) (1.16493,-0.882195) ]

