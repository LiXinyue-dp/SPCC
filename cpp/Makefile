CXXFLAGS	= -march=native -O3
CXXFLAGS	+= -std=c++11
CXXFLAGS	+= -isystem ./include
CXXFLAGS	+= -MMD
CC	= g++

TARGET	=  LFCK

all: $(TARGET)

LFCK: LFCK.o

clean:
	rm -f *~ $(TARGET) *.d *.o

-include *.d
