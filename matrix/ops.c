#include "ops.h"
#include <stdlib.h>
#include <stdio.h>

#define NUM_THREADS 24

int check_dimensions(Matrix *m1, Matrix *m2) {
	if (m1->rows == m2->rows && m1->cols == m2->cols) return 1;
	return 0;
}
Matrix* multiply(Matrix *m1, Matrix *m2) {
    if (check_dimensions(m1, m2)) {
        Matrix *m = matrix_create(m1->rows, m1->cols);

        // Temporary pointers for the data mapping
        double **m1_entries = m1->entries;
        double **m2_entries = m2->entries;
        double **m_entries = m->entries;

        // Map the data to the device (GPU)
        #pragma omp target enter data map(to: m1[:1], m2[:1])
        #pragma omp target enter data map(to: m1_entries[:m1->rows * m1->cols], m2_entries[:m2->rows * m2->cols])
        #pragma omp target enter data map(alloc: m_entries[:m->rows * m->cols])

        // Perform matrix multiplication on the GPU
        #pragma omp target teams distribute parallel for collapse(2) thread_limit(NUM_THREADS)
        for (int i = 0; i < m1->rows; i++) {
            for (int j = 0; j < m2->cols; j++) {
                m_entries[i][j] = m1_entries[i][j] * m2_entries[i][j];
            }
        }

        // Retrieve the results from the device (you can merge this codes to above map to as well)
        #pragma omp target exit data map(from: m_entries[:m->rows * m->cols])
        #pragma omp target exit data map(delete: m1_entries[:m1->rows * m1->cols], m2_entries[:m2->rows * m2->cols])

        return m;
    } else {
        printf("Dimension mismatch multiply: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(1);
    }
}

Matrix* add(Matrix *m1, Matrix *m2) {
    if (check_dimensions(m1, m2)) {
        Matrix *m = matrix_create(m1->rows, m1->cols);

        // Temporary pointers for the data mapping
        double **m1_entries = m1->entries;
        double **m2_entries = m2->entries;
        double **m_entries = m->entries;

        // Map the data to the device (GPU)
        #pragma omp target enter data map(to: m1[:1], m2[:1])
        #pragma omp target enter data map(to: m1_entries[:m1->rows * m1->cols], m2_entries[:m2->rows * m2->cols])
        #pragma omp target enter data map(alloc: m_entries[:m->rows * m->cols])

        // Perform matrix multiplication on the GPU
        #pragma omp target teams distribute parallel for collapse(2) thread_limit(NUM_THREADS)
        for (int i = 0; i < m1->rows; i++) {
            for (int j = 0; j < m2->cols; j++) {
                m_entries[i][j] = m1_entries[i][j] + m2_entries[i][j];
            }
        }

        // Retrieve the results from the device (you can merge this codes to above map to as well)
        #pragma omp target exit data map(from: m_entries[:m->rows * m->cols])
        #pragma omp target exit data map(delete: m1_entries[:m1->rows * m1->cols], m2_entries[:m2->rows * m2->cols])

        return m;
    } else {
        printf("Dimension mismatch multiply: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(1);
    }
}

Matrix* subtract(Matrix *m1, Matrix *m2) {
    if (check_dimensions(m1, m2)) {
        Matrix *m = matrix_create(m1->rows, m1->cols);

        // Temporary pointers for the data mapping
        double **m1_entries = m1->entries;
        double **m2_entries = m2->entries;
        double **m_entries = m->entries;

        // Map the data to the device (GPU)
        #pragma omp target enter data map(to: m1[:1], m2[:1])
        #pragma omp target enter data map(to: m1_entries[:m1->rows * m1->cols], m2_entries[:m2->rows * m2->cols])
        #pragma omp target enter data map(alloc: m_entries[:m->rows * m->cols])

        // Perform matrix multiplication on the GPU
        #pragma omp target teams distribute parallel for collapse(2) thread_limit(NUM_THREADS)
        for (int i = 0; i < m1->rows; i++) {
            for (int j = 0; j < m2->cols; j++) {
                m_entries[i][j] = m1_entries[i][j] - m2_entries[i][j];
            }
        }

        // Retrieve the results from the device (you can merge this codes to above map to as well)
        #pragma omp target exit data map(from: m_entries[:m->rows * m->cols])
        #pragma omp target exit data map(delete: m1_entries[:m1->rows * m1->cols], m2_entries[:m2->rows * m2->cols])

        return m;
    } else {
        printf("Dimension mismatch multiply: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(1);
    }
}

Matrix* apply(double (*func)(double), Matrix* m) {
	Matrix *mat = matrix_copy(m);

	double **mat_entries = mat->entries;
	#pragma omp target enter data map(alloc: mat_entries[:mat->rows * mat->cols])
	#pragma omp target teams distribute parallel for collapse(2) thread_limit(NUM_THREADS)
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] = (*func)(mat_entries[i][j]);
		}
	}
	return mat;
}



Matrix* dot(Matrix *m1, Matrix *m2) {
	if (m1->cols == m2->rows) {
		Matrix *m = matrix_create(m1->rows, m2->cols);

        // Temporary pointers for the data mapping
        double **m1_entries = m1->entries;
        double **m2_entries = m2->entries;
        double **m_entries = m->entries;

 // Map the data to the device (GPU)
        #pragma omp target enter data map(to: m1[:1], m2[:1])
        #pragma omp target enter data map(to: m1_entries[:m1->rows * m1->cols], m2_entries[:m2->rows * m2->cols])
        #pragma omp target enter data map(alloc: m_entries[:m->rows * m->cols])

        // Perform matrix multiplication on the GPU
        #pragma omp target teams distribute parallel for collapse(2) thread_limit(NUM_THREADS)
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				double sum = 0;
				for (int k = 0; k < m2->rows; k++) {
					sum += m1->entries[i][k] * m2->entries[k][j];
				}
				m->entries[i][j] = sum;
			}
		}

		        // Retrieve the results from the device (you can merge this codes to above map to as well)
        #pragma omp target exit data map(from: m_entries[:m->rows * m->cols])
        #pragma omp target exit data map(delete: m1_entries[:m1->rows * m1->cols], m2_entries[:m2->rows * m2->cols])
		
		return m;
	} else {
		printf("Dimension mistmatch dot: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* scale(double n, Matrix* m) {
	Matrix* mat = matrix_copy(m);
	double **mat_entries = mat->entries;
	double n_scale = n;
	#pragma omp target enter data map(to:n_scale)

	#pragma omp target enter data map(alloc: mat_entries[:mat->rows * mat->cols])

	#pragma omp target teams distribute parallel for collapse(2) thread_limit(NUM_THREADS)
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] = mat_entries[i][j] * n;
		}
	}
	return mat;
}

Matrix* addScalar(double n, Matrix* m) {
	Matrix* mat = matrix_copy(m);

	double **mat_entries = mat->entries;
	double n_scale = n;
	#pragma omp target enter data map(to:n_scale)
	#pragma omp target enter data map(alloc: mat_entries[:mat->rows * mat->cols])
	#pragma omp target teams distribute parallel for collapse(2) thread_limit(NUM_THREADS)
		for (int i = 0; i < m->rows; i++) {
			for (int j = 0; j < m->cols; j++) {
				mat->entries[i][j] = mat_entries[i][j] + n;
			}
		}
	return mat;
}

Matrix* transpose(Matrix* m) {
	Matrix* mat = matrix_create(m->cols, m->rows);
	double **mat_entries = mat->entries;

	#pragma omp target enter data map(alloc: mat_entries[:mat->rows * mat->cols])
	#pragma omp target teams distribute parallel for collapse(2) thread_limit(NUM_THREADS)
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[j][i] = mat_entries[i][j];
		}
	}
	return mat;
}