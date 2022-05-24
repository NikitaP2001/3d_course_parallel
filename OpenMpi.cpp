#include <math.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>

#include "misc.hpp"

const double eps =  0.0001; // accuracy

MPI_Status status;     
int num_procs, my_rank, Root = 0;

/*
 * Solves A*X = F, X - init approximation and res 
 * of ith iteration
 */
void Jacobi(double *A, double *F, double *X, int N)
{
  	int nb;

	double *X_New, *X_Old, *BlockX, tmp;

	nb = N / num_procs;
	int *FBlockSz = new int[num_procs];
	int *FBlockDisp = new int[num_procs];
	int *ABlSz = new int[num_procs];
	int *ABlDisp = new int[num_procs];
	FBlockSz[0] = (N % num_procs > 0) ? nb + 1 : nb;
	FBlockDisp[0] = 0;
	ABlSz[0] = FBlockSz[0] * N;
	ABlDisp[0] = 0;
	for (int i = 1; i < num_procs; i++) {
		FBlockSz[i] = (i < N % num_procs) ? nb + 1 : nb;
		FBlockDisp[i] = FBlockDisp[i - 1] + FBlockSz[i - 1];
		ABlSz[i] = FBlockSz[i] * N;
		ABlDisp[i] = ABlDisp[i - 1] + ABlSz[i - 1];
	}


	double *BlockA = new double[FBlockSz[my_rank] * N];
	double *BlockF = new double[FBlockSz[my_rank]];

	MPI_Scatterv(A, ABlSz, ABlDisp, MPI_DOUBLE, BlockA, ABlSz[my_rank],
	    MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Scatterv(F, FBlockSz, FBlockDisp, MPI_DOUBLE, BlockF, FBlockSz[my_rank],
	    MPI_DOUBLE, 0, MPI_COMM_WORLD);

	X_New  = new double[N];
	X_Old  = new double[N];

	BlockX = new double[FBlockSz[my_rank]];
	std::fill_n(BlockX, FBlockSz[my_rank], 0);

	MPI_Allgatherv(BlockX, FBlockSz[my_rank], MPI_DOUBLE, X_New, FBlockSz,
		       	FBlockDisp, MPI_DOUBLE, MPI_COMM_WORLD);


	double norm;
  	do {
  		norm = 0;

		for (int i = 0; i < N; i++) {
			X_Old[i] = X_New[i];
		}

		for (int i = 0; i < FBlockSz[my_rank]; i++) {
			BlockX[i] = BlockF[i];	
			int index = i * N;

			for (int g = 0; g < N; g++) {
				if (g != FBlockDisp[my_rank] + i)
					BlockX[i] -= BlockA[index + g] * X_New[g];
			}

			BlockX[i] /= BlockA[index + FBlockDisp[my_rank] + i];
		}

		MPI_Allgatherv(BlockX, FBlockSz[my_rank], MPI_DOUBLE, X_New, FBlockSz,
		       	FBlockDisp, MPI_DOUBLE, MPI_COMM_WORLD);

		for (int i = 0; i < N; i++) {
			if (fabs(X_New[i] - X_Old[i]) > norm)
				norm = fabs(X_New[i] - X_Old[i]);
		}

	} while(norm > eps); 

	if (my_rank == Root) {
		
		for (int i = 0; i < N; i++)
			X[i] = X_New[i];
	  
	}

	delete[] FBlockSz;
	delete[] FBlockDisp;
	delete[] ABlSz;
	delete[] ABlDisp;

	delete[] BlockA;
	delete[] BlockF;
	delete[] X_New;
	delete[] X_Old;
	delete[] BlockX;
}

/* run Jacobi with random data of dimension - N,
 * count execution time and print 
 */
void test(long N)
{	
	struct timeval tstart;
	struct timeval tend;
	vector<vector<double>> A(N, vector<double>(N));
	vector<double> F(N);
	vector<double> X(N);
	double *A_m;
	double *F_arr, *X_arr;

	if (my_rank == 0) {
		random_diag_dorminant_matrix(A);
		random_vector(F);

		gettimeofday(&tstart, NULL);
		F_arr = F.data();
		X_arr = X.data();

		A_m  = new double[N*N];
		int index = 0;
		for(int i = 0; i < N; i++)
			for(int j = 0; j < N; j++)
				A_m[index++] = A[i][j];
		
	}

	Jacobi(A_m, F_arr, X_arr, N);

	if (my_rank == 0) {
		
		gettimeofday(&tend, NULL);
		time_t sec = tend.tv_sec - tstart.tv_sec;
		suseconds_t msec = tend.tv_usec - tstart.tv_usec;
		double trun = sec + (double)msec / 10e5;

		cout << setw(23) << fixed << setprecision(6)
		<< trun;

		// error of iteration procces
		double err = 0;
		vector<double> F_err(N);
		matrix_mul(F_err, A, X);

		for (long i = 0; i < N; i++) {
			err += (F_err[i] - F[i]) * (F_err[i] - F[i]);
		}
		cout << setw(20) << err << endl;
		delete[] A_m;
	}
}

int main(int argc, char *argv[])
{
	long N;
	MPI_Init(&argc, &argv); 
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	N = 100;
	if (my_rank == 0) {
		cout << "Test Number" << setw(15) << "Matrix Size" << setw(25)
		<< "Execution time (sec)" << setw(10) << "Error" << endl;

		cout << setw(5) << 1 << setw(16) << N;
	}
	test(N);

	N = 1000;
	if (my_rank == 0) {

		cout << setw(5) << 2 << setw(16) << N;
	}
	test(N);

	N = 10000;
	if (my_rank == 0) {

		cout << setw(5) << 3 << setw(16) << N;
	}
	test(N);

	if (my_rank == 0) {
		cout << "Number of processors: " << num_procs << endl;
	}

	MPI_Finalize();

	exit(0);
}
