ROOTDIR=../../..
include $(ROOTDIR)/common.txt

INCLUDE= -I$(INCLUDEDIR) -I$(SCALARDIR) -I$(TENSORDIR) 

TESTS= testRscalarObj testCscalarObj 

DEPS=$(SCALARDIR)/*.hpp $(INCLUDEDIR)/*.hpp 

EXECS= 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_OBJECTS+=
CUDA_EXTERNS+=$(INCLUDEDIR)/Cnine_base.cu 
endif 




# -------------------------------------------------------------------------------------------------------------


testRscalarObj: testRscalarObj.cpp  $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testRscalarObj testRscalarObj.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testRscalarObj testRscalarObj.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCscalarObj: testCscalarObj.cpp  $(DEPS) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCscalarObj testCscalarObj.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testCscalarObj testCscalarObj.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif



# -------------------------------------------------------------------------------------------------------------


objects: $(OBJECTS) $(CUDA_OBJECTS)
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects execs derived tests 

clean: 
	rm -f $(TESTS) $(EXECS) $(OBJECTS) $(CUDA_OBJECTS) 

anew: clean all







