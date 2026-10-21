/*
 * Этап 4. MPI-программа. Вариант 5: блочное LU-разложение через GAXPY.
 *
 * Матрица хранится целиком на каждом процессе (после MPI_Bcast).
 * Для каждого блочного столбца K = [k, k+w):
 *   1) U(0:k,K) = L11^{-1} A(0:k,K): столбцы блока делятся между процессами,
 *      каждый решает треугольную систему для своих столбцов, затем MPI_Allgatherv;
 *   2) GAXPY A(k:n,K) -= L(k:n,0:k) U(0:k,K): строки делятся между процессами,
 *      затем MPI_Allgatherv собранной панели;
 *   3) неблочное LU панели (O(n r^2)) — избыточно на каждом процессе.
 *
 * Процесс 0 генерирует матрицу и рассылает её (MPI_Bcast), после разложения
 * проверяет решение A x = b (x* = (1,...,1)).
 *
 * Сборка (MS-MPI + MinGW):
 *   gcc -O2 -Iinclude -I"%MSMPI_INC%" -D"__int64=long long" stage4_mpi.c -L"%MSMPI_LIB32%" -lmsmpi
 * Запуск:  mpiexec -n P stage4_mpi.exe [n1 n2 ...]
 * Вывод (CSV, rank 0): procs,n,r,time,err,res
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define R    32
#define REPS 3
static int rank, size;

static void split(int total, int parts, int *cnt, int *off)
{
    for (int p = 0, o = 0; p < parts; p++) {
        cnt[p] = total / parts + (p < total % parts);
        off[p] = o;
        o += cnt[p];
    }
}

/* неблочное LU панели P (m x w, row-major) в jki (gaxpy) форме */
static void panel_lu(int m, int w, double *P)
{
    for (int c = 0; c < w; c++) {
        for (int j = 0; j < c; j++) {
            double u = P[j * w + c];
            for (int i = j + 1; i < m; i++)
                P[(size_t)i * w + c] -= P[(size_t)i * w + j] * u;
        }
        double p = 1.0 / P[c * w + c];
        for (int i = c + 1; i < m; i++) P[(size_t)i * w + c] *= p;
    }
}

void block_lu_gaxpy_mpi(int n, int r, double *A)
{
    int *cnt = malloc(size * sizeof(int)), *off = malloc(size * sizeof(int));
    int *rc = malloc(size * sizeof(int)), *ro = malloc(size * sizeof(int));
    int *cc = malloc(size * sizeof(int)), *co = malloc(size * sizeof(int));
    double *U   = malloc((size_t)n * r * sizeof(double));   /* k x w, row-major */
    double *P   = malloc((size_t)n * r * sizeof(double));   /* (n-k) x w       */
    double *buf = malloc((size_t)n * r * sizeof(double));

    for (int k = 0; k < n; k += r) {
        int w = (k + r <= n) ? r : n - k;

        /* ---- 1) треугольное решение, делим столбцы блока ---- */
        if (k > 0) {
            split(w, size, cnt, off);
            int c0 = k + off[rank], nc = cnt[rank];
            /* мои столбцы -> плотный буфер buf (k x nc), решение по строкам (gaxpy) */
            for (int i = 0; i < k; i++)
                for (int c = 0; c < nc; c++) buf[i * nc + c] = A[(size_t)i * n + c0 + c];
            for (int i = 1; i < k; i++) {
                const double *li = A + (size_t)i * n;
                double *ui = buf + i * nc;
                for (int j = 0; j < i; j++) {
                    double l = li[j];
                    const double *uj = buf + j * nc;
                    for (int c = 0; c < nc; c++) ui[c] -= l * uj[c];
                }
            }
            for (int p = 0; p < size; p++) { cc[p] = cnt[p] * k; co[p] = off[p] * k; }
            MPI_Allgatherv(buf, cc[rank], MPI_DOUBLE, U, cc, co, MPI_DOUBLE, MPI_COMM_WORLD);
            /* распаковка в A и в плотный блок U (k x w) */
            for (int p = 0; p < size; p++)
                for (int i = 0; i < k; i++)
                    for (int c = 0; c < cnt[p]; c++)
                        A[(size_t)i * n + k + off[p] + c] = U[co[p] + i * cnt[p] + c];
            for (int i = 0; i < k; i++)
                for (int c = 0; c < w; c++) U[i * w + c] = A[(size_t)i * n + k + c];

            /* ---- 2) GAXPY, делим строки k..n-1 ---- */
            int m = n - k;
            split(m, size, cnt, off);
            int i0 = off[rank], ni = cnt[rank];
            for (int i = i0; i < i0 + ni; i++)
                for (int c = 0; c < w; c++) P[(size_t)i * w + c] = A[(size_t)(k + i) * n + k + c];
            for (int i = i0; i < i0 + ni; i++) {
                const double *li = A + (size_t)(k + i) * n;
                double *pi = P + (size_t)i * w;
                for (int j = 0; j < k; j++) {
                    double l = li[j];
                    const double *uj = U + (size_t)j * w;
                    for (int c = 0; c < w; c++) pi[c] -= l * uj[c];
                }
            }
            for (int p = 0; p < size; p++) { rc[p] = cnt[p] * w; ro[p] = off[p] * w; }
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, P, rc, ro, MPI_DOUBLE, MPI_COMM_WORLD);
        } else {
            for (int i = 0; i < n; i++)
                for (int c = 0; c < w; c++) P[(size_t)i * w + c] = A[(size_t)i * n + c];
        }
        /* ---- 3) LU панели (в плотном буфере P) ---- */
        panel_lu(n - k, w, P);
        for (int i = 0; i < n - k; i++)
            for (int c = 0; c < w; c++) A[(size_t)(k + i) * n + k + c] = P[(size_t)i * w + c];
    }
    free(cnt); free(off); free(rc); free(ro); free(cc); free(co);
    free(U); free(P); free(buf);
}

static void run(int n)
{
    double *A = malloc((size_t)n * n * sizeof(double));
    double *A0 = NULL, *b = NULL;

    if (rank == 0) {
        A0 = malloc((size_t)n * n * sizeof(double));
        b = malloc(n * sizeof(double));
        srand(12345);
        for (size_t i = 0; i < (size_t)n * n; i++) A0[i] = (double)rand() / RAND_MAX - 0.5;
        for (int i = 0; i < n; i++) A0[(size_t)i * n + i] += n / 4.0;
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) s += A0[(size_t)i * n + j];
            b[i] = s;
        }
    }

    double tmin = 1e30;
    for (int rep = 0; rep < REPS; rep++) {
        if (rank == 0)
            for (size_t i = 0; i < (size_t)n * n; i++) A[i] = A0[i];
        MPI_Bcast(A, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double t = MPI_Wtime();
        block_lu_gaxpy_mpi(n, R, A);
        MPI_Barrier(MPI_COMM_WORLD);
        t = MPI_Wtime() - t;
        if (t < tmin) tmin = t;
    }

    if (rank == 0) {
        double *x = malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) x[i] = b[i];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < i; j++) x[i] -= A[(size_t)i * n + j] * x[j];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) x[i] -= A[(size_t)i * n + j] * x[j];
            x[i] /= A[(size_t)i * n + i];
        }
        double err = 0, res = 0, bn = 0;
        for (int i = 0; i < n; i++) {
            err = fmax(err, fabs(x[i] - 1.0));
            double s = -b[i];
            for (int j = 0; j < n; j++) s += A0[(size_t)i * n + j] * x[j];
            res = fmax(res, fabs(s));
            bn = fmax(bn, fabs(b[i]));
        }
        printf("%d,%d,%d,%.6f,%.3e,%.3e\n", size, n, R, tmin, err, res / bn);
        fflush(stdout);
        free(x); free(A0); free(b);
    }
    free(A);
}

int main(int argc, char **argv)
{
    static const int def[] = {500, 1000, 2000, 3000};
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) printf("procs,n,r,time,err,res\n");
    if (argc > 1)
        for (int i = 1; i < argc; i++) run(atoi(argv[i]));
    else
        for (size_t i = 0; i < sizeof def / sizeof *def; i++) run(def[i]);
    MPI_Finalize();
    return 0;
}
