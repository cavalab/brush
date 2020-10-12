CC=g++
CFLAGS=-I /usr/include/eigen3/
LFLAGS=-I /usr/include/eigen3/
DEPS = state.h init.h logger.h nodevector.h state.h tree.hh utils.h
OBJ = data.o ex_tree.o logger.o utils.o

%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $< 

ex_tree: $(OBJ)
	$(CC) -o $@ $^ $(LFLAGS)
