SEQ= seq
MPI= mpi
OMP= omp




SRC_MPI = OpenMpi.cpp misc.cpp
SRC_OMP = OpenMp.cpp misc.cpp
SRC_SEQ = seq.cpp misc.cpp


all: $(SEQ) $(MPI) $(OMP)


$(SEQ): $(SRC_SEQ)
	c++ -O3 $(SRC_SEQ) -o $@


$(MPI): $(SRC_MPI)
	mpicxx -O3 $(SRC_MPI) -o $@

$(OMP): $(SRC_OMP)
	c++ -fopenmp -O3 $(SRC_OMP) -o $@

clean:
	rm $(SEQ) $(MPI) $(OMP)
