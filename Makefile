TARGET = pbas

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3

NVCC = nvcc
NVCCFLAGS =  -Xptxas -O3 --default-stream per-thread -std=c++14 -Xcompiler -pipe,-fPIC,-ffast-math,-fomit-frame-pointer
NVCCFLAGS += -gencode arch=compute_75,code=compute_75 # Change this line with your CUDA Compute Capability !

SRC = $(wildcard *.cpp)
OBJ = $(SRC:%.cpp=%.o)
CU_FILES = $(wildcard *.cu)
CU_OBJ = $(CU_FILES:%.cu=%.o)

# You should have OpenCV library installed!
INCS = -I ./ -I/usr/local/include/opencv -I/usr/local/include
LIBS = -L/usr/local/lib -lopencv_core -lopencv_videoio -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lpthread\

# Cuda Paths
INCS += -I /usr/local/cuda/include
LIBS += -L/usr/local/cuda/lib64 -lcudart -lcurand

# Uncomment following line to enable Showing Output Video feature
CXXFLAGS += -D SHOW_OUTPUT_VIDEO

# Uncomment following line to enable Time Recorder feature
#CXXFLAGS += -D TIME_RECORDER

# Uncomment following line to enable Writing Output Video feature
#CXXFLAGS += -D OUTPUT_VIDEO_WRITER

# Uncomment following line to enable Writing Output Frames feature
#CXXFLAGS += -D OUTPUT_FRAME_WRITER

default: $(TARGET)

$(TARGET): $(CU_OBJ) $(OBJ)
	$(CXX) -o $(TARGET) $(OBJ) $(CU_OBJ) $(LIBS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCS) -c -o $*.o $<

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCS) -c -o $*.o $<

clean:
	$(RM) core* $(TARGET) $(OBJ) $(CU_OBJ)