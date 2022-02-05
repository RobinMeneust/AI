HEAD = $(wildcard ./include/*.h)
SRC = $(wildcard src/*.cpp)
OBJ = $(patsubst src/%.cpp, obj/%.o, $(SRC))
CC = g++
PROG=./bin/IA
LINKSFLAGSOLD=-I"E:\install\opencv\OpenCV-MinGW-Build-OpenCV-4.5.2-x64\include" -L"E:\install\opencv\OpenCV-MinGW-Build-OpenCV-4.5.2-x64\x64\mingw\bin" -L"E:\install\opencv\OpenCV-MinGW-Build-OpenCV-4.5.2-x64\x64\mingw\lib" -llibopencv_calib3d452 -llibopencv_core452 -llibopencv_dnn452 -llibopencv_features2d452 -llibopencv_flann452 -llibopencv_highgui452 -llibopencv_imgcodecs452 -llibopencv_imgproc452 -llibopencv_ml452 -llibopencv_objdetect452 -llibopencv_photo452 -llibopencv_stitching452 -llibopencv_video452 -llibopencv_videoio452
LINKSFLAGS=-I E:\install\opencv\OpenCV-MinGW-Build-OpenCV-4.5.2-x64\include -L E:\install\opencv\OpenCV-MinGW-Build-OpenCV-4.5.2-x64\x64\mingw\bin -L E:\install\opencv\OpenCV-MinGW-Build-OpenCV-4.5.2-x64\x64\mingw\lib -lopencv_core452 -lopencv_highgui452

all: $(PROG) 

$(PROG) : $(OBJ)
	$(CC) $(LINKSFLAGSOLD) $^ -o $@ 

#obj/%.o: src/%.cpp $(HEAD)
#	$(CC) $(LINKSFLAGS) -c $< -o $@ 

obj/main.o: src/main.cpp $(HEAD)
	$(CC) -I"E:\install\opencv\OpenCV-MinGW-Build-OpenCV-4.5.2-x64\include" -c $< -o $@ 

obj/classes.o: src/classes.cpp $(HEAD)
	$(CC) -c $< -o $@ 
	
.PHONY : cleanlinux cleanwin doc run

cleanlinux:
	rm obj/*.o

cleanwin:
	del obj\*.o

cleandoclinux:
	del doxygen\html

cleandocwin:
	del doxygen\html

doc:
	doxygen doxygen/Doxyfile

run:
	$(PROG)