#include <vector>
#include <math.h>

using namespace std;

/* init random diagonal dorminant matrix, to make sure
 * iteration process converge
 */
void random_diag_dorminant_matrix(vector<vector<double>> &A)
{
	double diagSum;
	long N = A.size();

	for (long i = 0; i < N; i++)
		for (long j = 0; j < N; j++)
			A[i][j] = (float) rand() / RAND_MAX;

	for (long i = 0; i < N; i++) {
		diagSum = 0;
		for (long j = 0; j < N; j++)
			if (j != i)
				diagSum += fabs(A[i][j]);

		if (A[i][i] > 0)
			A[i][i] += diagSum;
		else
			A[i][i] -= diagSum;

		if (fabs(A[i][i]) < 0.005) {
			if (A[i][i] >= 0)
				A[i][i] += 2;
			else
				A[i][i] -= 2;
		}
	}
}

// b = A*x;
void matrix_mul(vector<double> &b, vector<vector<double>> &A, vector<double> &x)
{
	for (long i = 0; i < A.size(); i++)
	{
		b[i] = 0;
		for (long j = 0; j < A.size(); j++)
			b[i] += A[i][j] * x[j];
	}
}

void random_vector(vector<double> &x)
{
	for (long i = 0; i < x.size(); i++) {
		x[i] = (float) rand() / RAND_MAX;
	}
}

