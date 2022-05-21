#include <math.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>

#include "misc.hpp"

#define  MAX_ITERATIONS 1000
const double diff =  0.001; // accuracy

MPI_Status status;     
int Numprocs, MyRank, Root = 0;

double Distance(double *X_Old, double *X_New, int n_size)
{
   int  index;
	double Sum;

   Sum = 0.0;
	for(index=0; index<n_size; index++)
		 Sum += (X_New[index] - X_Old[index])*(X_New[index]-X_Old[index]);

   return(Sum);
}

/*
 * Solves A*X = F, X - init approximation and res 
 * of ith iteration
 */
void Jacobi(double **Matrix_A, double *F, double *X, int N)
{
  	int n_size = N, NoofRows_Bloc;
  	int irow, jrow, icol, index, Iteration, GlobalRowNo;

	double *Input_B, *Input_A, *ARecv, *BRecv;
	double *X_New, *X_Old, *Bloc_X, tmp;

	if (MyRank == Root) {
		Input_B = new double[n_size];
		for (int i = 0; i < n_size; i++)
			Input_B[i] = F[i];

		Input_A  = new double[n_size*n_size];
	  	index = 0;
	  	for(irow=0; irow<n_size; irow++)
			for(icol=0; icol<n_size; icol++)
				Input_A[index++] = Matrix_A[irow][icol];
	}

	MPI_Bcast(&n_size, 1, MPI_INT, Root, MPI_COMM_WORLD); 

	NoofRows_Bloc = n_size/Numprocs;
	ARecv = new double[NoofRows_Bloc * n_size];
	BRecv = new double[NoofRows_Bloc];

	MPI_Scatter (Input_A, NoofRows_Bloc * n_size, MPI_DOUBLE, ARecv, NoofRows_Bloc * n_size, 
					MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Scatter (Input_B, NoofRows_Bloc, MPI_DOUBLE, BRecv, NoofRows_Bloc, MPI_DOUBLE, 0, 
					MPI_COMM_WORLD);
	X_New  = new double[n_size];
	X_Old  = new double[n_size];
	Bloc_X = new double[NoofRows_Bloc];

	for(irow=0; irow<NoofRows_Bloc; irow++)
		Bloc_X[irow] = BRecv[irow];

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allgather(Bloc_X, NoofRows_Bloc, MPI_DOUBLE, X_New, NoofRows_Bloc, 
					 MPI_DOUBLE, MPI_COMM_WORLD);

  	long norm;
  	do{
		for(irow=0; irow<n_size; irow++)
			X_Old[irow] = X_New[irow];

      		for(irow=0; irow<NoofRows_Bloc; irow++){

			GlobalRowNo = (MyRank * NoofRows_Bloc) + irow;
			Bloc_X[irow] = BRecv[irow];
			index = irow * n_size;

			for(icol=0; icol<GlobalRowNo; icol++)
				Bloc_X[irow] -= X_Old[icol] * ARecv[index + icol];

			for(icol=GlobalRowNo+1; icol<n_size; icol++)
				Bloc_X[irow] -= X_Old[icol] * ARecv[index + icol];

			Bloc_X[irow] = Bloc_X[irow] / ARecv[irow*n_size + GlobalRowNo];
		}

  		MPI_Allgather(Bloc_X, NoofRows_Bloc, MPI_DOUBLE, X_New, 
				NoofRows_Bloc, MPI_DOUBLE, MPI_COMM_WORLD);
	}while((Distance(X_Old, X_New, n_size) >= diff)); 

	if (MyRank == Root) {
		delete[] Input_A;
		delete[] Input_B;
		
		for (int i = 0; i < N; i++)
			X[i] = X_New[i];
	  
	}
	delete[] ARecv;
	delete[] BRecv;
	delete[] X_New;
	delete[] X_Old;
	delete[] Bloc_X;
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
	double **A_m;
	double *F_arr, *X_arr;

	if (MyRank == 0) {
		random_diag_dorminant_matrix(A);
		random_vector(F);
		A_m = new double*[N];
		for (long i = 0; i < N; i++) {
			A_m[i] = new double[N];
			for (long j = 0; j < N; j++)
				A_m[i][j] = A[i][j];
		}

		gettimeofday(&tstart, NULL);
		F_arr = F.data();
		X_arr = X.data();
	}

	Jacobi(A_m, F_arr, X_arr, N);

	if (MyRank == 0) {
		
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
		for (long i = 0; i < N; i++)
			delete[] A_m[i];
		delete[] A_m;
	}
}

int main(int argc, char *argv[])
{
	long N;
	MPI_Init(&argc, &argv); 
	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
	MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);

	N = 10;
	if (MyRank == 0) {
		cout << "Test Number" << setw(15) << "Matrix Size" << setw(25)
		<< "Execution time (sec)" << setw(10) << "Error" << endl;

		cout << setw(5) << 1 << setw(16) << N;
	}
	test(N);

	N = 100;
	if (MyRank == 0) {

		cout << setw(5) << 2 << setw(16) << N;
	}
	test(N);

	N = 1000;
	if (MyRank == 0) {

		cout << setw(5) << 3 << setw(16) << N;
	}
	test(N);

	N = 2000;
	if (MyRank == 0) {

		cout << setw(5) << 4 << setw(16) << N;
	}
	test(N);

	N = 3000;
	if (MyRank == 0) {

		cout << setw(5) << 5 << setw(16) << N;
	}
	test(N);

	N = 4000;
	if (MyRank == 0) {

		cout << setw(5) << 6 << setw(16) << N;
	}
	test(N);

	N = 5000;
	if (MyRank == 0) {

		cout << setw(5) << 7 << setw(16) << N;
	}
	test(N);

	N = 6000;
	if (MyRank == 0) {

		cout << setw(5) << 8 << setw(16) << N;
	}
	test(N);

	N = 7000;
	if (MyRank == 0) {

		cout << setw(5) << 9 << setw(16) << N;
	}
	test(N);

	N = 8000;
	if (MyRank == 0) {

		cout << setw(5) << 10 << setw(16) << N;
	}
	test(N);

	N = 9000;
	if (MyRank == 0) {

		cout << setw(5) << 11 << setw(16) << N;
	}
	test(N);

	N = 10000;
	if (MyRank == 0) {

		cout << setw(5) << 12 << setw(16) << N;
	}
	test(N);

	if (MyRank == 0) {
		cout << "Number of processors: " << Numprocs << endl;
	}

	MPI_Finalize();

}
