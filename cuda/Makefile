ROOTDIR=..
include $(ROOTDIR)/common.txt

INCLUDE= -I$(INCLUDEDIR) -I$(INCLUDEDIR)/cmaps -I$(SCALARDIR) -I$(TENSORDIR) -I$(TENSORARRAYDIR) -I$(TENSORARRAYDIR)/cell_ops \
-I$(MANAGEDDIR) -I$(MSCALARDIR) -I$(MTENSORDIR) \
$(CENGINE_INCLUDES) 

TESTS=

DEPS= $(TENSORDIR)/*.hpp $(INCLUDEDIR)/*.hpp $(INCLUDEDIR)/cmaps/*.hpp $(SCALARDIR)/*.hpp $(MSCALARDIR)/*.hpp $(MTENSORDIR)/*.hpp

EXECS= 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_OBJECTS+=Flock.o CtensorA_add.o CtensorA_add_plus.o CtensorA_copy.o CtensorA_plus.o 
CUDA_EXTERNS+=$(INCLUDEDIR)/Cnine_base.cu 
NVCCFLAGS+= -D_WITH_CUDA -D_WITH_CUBLAS
endif 


# -------------------------------------------------------------------------------------------------------------



Flock.o: Flock.cu Makefile  
	$(NVCC) $(NVCCFLAGS) -c Flock.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

CtensorA_add.o: CtensorA_add.cu Makefile  
	$(NVCC) $(NVCCFLAGS) -c CtensorA_add.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

CtensorA_add_plus.o: CtensorA_add_plus.cu Makefile  
	$(NVCC) $(NVCCFLAGS) -c CtensorA_add_plus.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

CtensorA_copy.o: CtensorA_copy.cu Makefile  
	$(NVCC) $(NVCCFLAGS) -c CtensorA_copy.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

CtensorA_plus.o: CtensorA_plus.cu Makefile  
	$(NVCC) $(NVCCFLAGS) -c CtensorA_plus.cu $(CFLAGS) $(MACROS) $(INCLUDE) 



objects: $(OBJECTS) $(CUDA_OBJECTS)
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects execs derived tests 

clean: 
	rm -f $(TESTS) $(EXECS) $(OBJECTS) $(CUDA_OBJECTS) 

anew: clean all


