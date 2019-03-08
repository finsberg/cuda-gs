LIBNAME = libgs
LIBFILE = $(LIBNAME).so
LIB_SOURCES = gs.c
LIB_OBJECTS = $(LIB_SOURCES:.c=.o)

CFLAGS = -Wall -O3 -fopenmp -fpic -march=native

$(LIBFILE): $(LIB_OBJECTS)
	$(CC) -shared -o $@ $^

clean:
	$(RM) $(LIB_OBJECTS) $(LIBFILE)
