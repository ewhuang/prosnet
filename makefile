CC = g++
CFLAGS = -lm -pthread -O2 -march=native -Wall -funroll-loops -Wno-unused-result -lm
INCLUDES = -I/usr/local/include -I../eigen-3.2.4
LIBS = -L/usr/local/lib


embed : ransampl.o linelib.o main.o
	$(CC) $(CFLAGS) -o embed ransampl.o linelib.o main.o $(INCLUDES) $(LIBS)

ransampl.o : ransampl.c
	$(CC) $(CFLAGS) -c ransampl.c $(INCLUDES) $(LIBS)

linelib.o : linelib.cpp ransampl.h
	$(CC) $(CFLAGS) -c linelib.cpp $(INCLUDES) $(LIBS)

main.o : main.cpp linelib.o
	$(CC) $(CFLAGS) -c main.cpp $(INCLUDES) $(LIBS)

clean :
	rm -rf *.o embed
