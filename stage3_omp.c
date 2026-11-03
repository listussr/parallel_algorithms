/*
 * Этап 3. OpenMP: блочное LU через GAXPY.
 * Параллельные циклы помечены #pragma omp parallel for; итерации между
 * потоками распределяет сама OpenMP (schedule(static)), без ручного деления
 * по номеру потока. Число потоков задаёт OMP_NUM_THREADS.
 * Сборка и запуск — README.md; алгоритм и результаты — docs/03_openmp.md.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define R    32
#define REPS 3
#define CB   8    /* столбцов в порции TRSM: 8 double = одна кэш-линия */

/* omp_get_wtime в MinGW GCC 9.2 имеет разрешение около 1 мс. */
#ifdef _WIN32
#include <malloc.h>
#define alloc64(size) _aligned_malloc(size, 64)
#define free64(p)     _aligned_free(p)
#include <windows.h>
static double wtime(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
#include <time.h>
static double wtime(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
static void *alloc64(size_t size)
{
    void *p;
    return posix_memalign(&p, 64, size) ? NULL : p;
}
#define free64(p) free(p)
#endif

/* L11 * U = U_исх; L11 — нижняя часть A с единичной диагональю.
 * Строки U зависят друг от друга, а столбцы независимы, поэтому параллельный
 * цикл идёт по порциям из CB столбцов: потоки пишут в разные кэш-линии. */
static void trsm(int n, int k, int w, const double *restrict A, double *restrict U)
{
    #pragma omp parallel for schedule(static)
    for (int cb = 0; cb < w; cb += CB) {
        int ce = cb + CB < w ? cb + CB : w;
        for (int i = 1; i < k; i++) {
            const double *li = A + (size_t)i * n;
            double *ui = U + (size_t)i * w;
            for (int j = 0; j < i; j++) {
                double l = li[j];
                const double *uj = U + (size_t)j * w;
                for (int c = cb; c < ce; c++)
                    ui[c] -= l * uj[c];
            }
        }
    }
}

/* P(m x w) -= L(k:n, 0:k) * U(k x w): строки P независимы. */
static void gaxpy(int n, int k, int m, int w, const double *restrict A,
                  const double *restrict U, double *restrict P)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        const double *li = A + (size_t)(k + i) * n;
        double *pi = P + (size_t)i * w;
        for (int j = 0; j < k; j++) {
            double l = li[j];
            const double *uj = U + (size_t)j * w;
            for (int c = 0; c < w; c++)
                pi[c] -= l * uj[c];
        }
    }
}

/* LU панели P (m x w) по столбцам. Для столбца c элементы U (строки i <= c)
 * вычисляются последовательно, а строки L (i > c) независимы друг от друга. */
static void panel_lu(int m, int w, double *restrict P)
{
    for (int c = 0; c < w; c++) {
        for (int i = 1; i <= c; i++)
            for (int j = 0; j < i; j++)
                P[(size_t)i * w + c] -= P[(size_t)i * w + j] * P[(size_t)j * w + c];
        double p = 1.0 / P[(size_t)c * w + c];
        #pragma omp parallel for schedule(static)
        for (int i = c + 1; i < m; i++) {
            double s = P[(size_t)i * w + c];
            for (int j = 0; j < c; j++)
                s -= P[(size_t)i * w + j] * P[(size_t)j * w + c];
            P[(size_t)i * w + c] = s * p;
        }
    }
}

/* A (n x n, по строкам) заменяется на LU без выбора ведущего элемента.
 * Буферы вмещают n*r элементов, но текущий шаг строки в них равен w. */
void block_lu_gaxpy(int n, int r, double *A)
{
    /* U выровнен на 64 байта: порции CB столбцов TRSM не делят кэш-линии. */
    double *U = alloc64((size_t)n * r * sizeof(double));
    double *P = malloc((size_t)n * r * sizeof(double));
    for (int k = 0; k < n; k += r) {
        int w = (k + r <= n) ? r : n - k, m = n - k;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < k; i++)
            for (int c = 0; c < w; c++) U[(size_t)i * w + c] = A[(size_t)i * n + k + c];
        trsm(n, k, w, A, U);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < k; i++)
            for (int c = 0; c < w; c++) A[(size_t)i * n + k + c] = U[(size_t)i * w + c];

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++)
            for (int c = 0; c < w; c++) P[(size_t)i * w + c] = A[(size_t)(k + i) * n + k + c];
        gaxpy(n, k, m, w, A, U, P);
        panel_lu(m, w, P);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++)
            for (int c = 0; c < w; c++) A[(size_t)(k + i) * n + k + c] = P[(size_t)i * w + c];
    }
    free64(U); free(P);
}

static void lu_solve(int n, const double *A, double *x)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++) x[i] -= A[(size_t)i * n + j] * x[j];
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) x[i] -= A[(size_t)i * n + j] * x[j];
        x[i] /= A[(size_t)i * n + i];
    }
}

static void run(int n)
{
    double *A0 = malloc((size_t)n * n * sizeof(double));
    double *A  = malloc((size_t)n * n * sizeof(double));
    double *b  = malloc(n * sizeof(double));
    double *x  = malloc(n * sizeof(double));

    srand(12345);
    for (size_t i = 0; i < (size_t)n * n; i++) A0[i] = (double)rand() / RAND_MAX - 0.5;
    for (int i = 0; i < n; i++) A0[(size_t)i * n + i] += n / 4.0;
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = 0; j < n; j++) s += A0[(size_t)i * n + j];
        b[i] = s;
    }

    double tmin = 1e30;
    for (int rep = 0; rep < REPS; rep++) {
        for (size_t i = 0; i < (size_t)n * n; i++) A[i] = A0[i];
        double t = wtime();
        block_lu_gaxpy(n, R, A);
        t = wtime() - t;
        if (t < tmin) tmin = t;
    }

    for (int i = 0; i < n; i++) x[i] = b[i];
    lu_solve(n, A, x);
    double err = 0, res = 0, bn = 0;
    for (int i = 0; i < n; i++) {
        err = fmax(err, fabs(x[i] - 1.0));
        double s = -b[i];
        for (int j = 0; j < n; j++) s += A0[(size_t)i * n + j] * x[j];
        res = fmax(res, fabs(s));
        bn = fmax(bn, fabs(b[i]));
    }
    printf("%d,%d,%d,%.6f,%.3e,%.3e\n", n, omp_get_max_threads(), R, tmin, err, res / bn);
    fflush(stdout);
    free(A0); free(A); free(b); free(x);
}

int main(int argc, char **argv)
{
    static const int def[] = {100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000};
    printf("n,threads,r,time,err,res\n");
    if (argc > 1)
        for (int i = 1; i < argc; i++) run(atoi(argv[i]));
    else
        for (size_t i = 0; i < sizeof def / sizeof *def; i++) run(def[i]);
    return 0;
}
