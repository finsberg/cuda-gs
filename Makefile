LIBNAME = libgs
LIBFILE = $(LIBNAME).so
LIB_SOURCES = gs.c
LIB_OBJECTS = $(LIB_SOURCES:.c=.o)

CFLAGS = -Wall -O3 -fPIC -march=native

#$(info $(LIB_OBJECTS))
$(LIBFILE): $(LIB_OBJECTS)
	$(CC) -shared -o $@ $^
