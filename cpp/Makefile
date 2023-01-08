CFLAGS = -std=c++17 -O3 -Wall -pedantic
CFLAGS += `pkg-config --cflags opencv4`
C = g++

OPENCV_LIBS = -lopencv_core

FILES = $(wildcard src/*.cpp)
NPFILES = $(patsubst src/%,%,$(FILES))
OBJS = $(addprefix bin/,$(patsubst %.cpp,%.o,$(NPFILES)))
EXENAME = main

HEADERS = $(wildcard src/*.hpp)

main: lib/libslam.a 
	@mkdir -p bin
	$(C) -o bin/$(EXENAME) $(CFLAGS) main/main.cpp -Llib -lslam $(OPENCV_LIBS) -Isrc

lib: lib/libslam.a

lib/libslam.a: $(OBJS)
	@mkdir -p lib
	ar rv $@ $^
	ranlib $@

$(OBJS): bin/%.o: src/%.cpp $(HEADERS)
	@mkdir -p bin
	$(C) -c $(patsubst bin/%,src/%,$(patsubst %.o,%.cpp,$@)) -o $@ $(CFLAGS)

clean:
	rm -f bin/$(EXENAME)
	rm -f bin/*.o
	rm -f lib/*.a