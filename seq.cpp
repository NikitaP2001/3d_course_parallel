#include <math.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <unistd.h>
#include <sys/time.h>

#include "misc.hpp"

const double eps = 0.001; // accuracy

/*
 * Solves A*X = F, X - init approximationi and res 
 * of ith iteration
 */
void Jacobi(vector<vector<double>> &A, vector<double> &F, vector<double> &X)
{
	long N = A.size();
	vector<double> TempX(N);
	double norm;

	do {
		for (long i = 0; i < N; i++) {
			TempX[i] = F[i];
			for (int g = 0; g < N; g++) {
				if (i != g)
					TempX[i] -= A[i][g] * X[g];
			}
			TempX[i] /= A[i][i];
		}
        	norm = fabs(X[0] - TempX[0]);
		for (long h = 0; h < N; h++) {
			if (fabs(X[h] - TempX[h]) > norm)
				norm = fabs(X[h] - TempX[h]);
			X[h] = TempX[h];
		}
	} while (norm > eps);
}

/* run Jacobi with random data of dimension - N,
 * count execution time and print 
 */
void test(long N)
{
	vector<vector<double>> A(N, vector<double>(N));
	vector<double> F(N);
	vector<double> X(N);

	random_diag_dorminant_matrix(A);
	random_vector(F);

	struct timeval tstart;
	struct timeval tend;
	gettimeofday(&tstart, NULL);

	Jacobi(A, F, X);

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
}

int main()
{
	cout << "Test Number" << setw(15) << "Matrix Size" << setw(25)
	<< "Execution time (sec)" << setw(10) << "Error" << endl;

	long N = 10;
	cout << setw(5) << 1 << setw(16) << N;
	test(N);

	N = 100;
	cout << setw(5) << 2 << setw(16) << N;
	test(N);

	N = 1000;
	cout << setw(5) << 3 << setw(16) << N;
	test(N);

	N = 2000;
	cout << setw(5) << 4 << setw(16) << N;
	test(N);

	N = 3000;
	cout << setw(5) << 5 << setw(16) << N;
	test(N);

	N = 4000;
	cout << setw(5) << 6 << setw(16) << N;
	test(N);

	N = 5000;
	cout << setw(5) << 7 << setw(16) << N;
	test(N);

	N = 6000;
	cout << setw(5) << 8 << setw(16) << N;
	test(N);

	N = 7000;
	cout << setw(5) << 9 << setw(16) << N;
	test(N);

	N = 8000;
	cout << setw(5) << 10 << setw(16) << N;
	test(N);

	N = 9000;
	cout << setw(5) << 11 << setw(16) << N;
	test(N);

	N = 10000;
	cout << setw(5) << 12 << setw(16) << N;
	test(N);
}
