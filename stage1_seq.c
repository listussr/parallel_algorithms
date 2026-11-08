/*
 * Этап 1. Последовательное блочное LU-разложение: Алгоритм 37 из пособия
 * Головашкина (параграф 2.4). Обозначения k, N, alpha, a, b, c — как в пособии.
 * Единичная диагональ L не записывается.
 * Выбора ведущего элемента и проверки нулевых пивотов нет.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ALPHA 32  /* размер малого блока */
#define REPS  3   /* число повторов, берётся минимальное время */

#ifdef _WIN32
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
#endif

/* A хранится по строкам: ниже диагонали — L, на диагонали и выше — U. */
#define A(i, j) A[(size_t)(i) * n + (j)]
/* Граница блока с номером i; последний блок может быть короче (задача 2.4.4). */
#define END(i) ((i) * alpha + alpha < n ? (i) * alpha + alpha : n)

void block_lu(int n, int alpha, double *A)
{
    int N = (n + alpha - 1) / alpha;

    for (int k = 0; k < N; k++) {
        int a0 = k * alpha, a1 = END(k);

        /* 1. A(a,a) = L(a,a) U(a,a): Алгоритм 30 со строчным saxpy */
        for (int l = a0; l < a1; l++)
            for (int s = l + 1; s < a1; s++) {
                A(s, l) /= A(l, l);
                for (int q = l + 1; q < a1; q++)
                    A(s, q) -= A(s, l) * A(l, q);
            }

        /* 2. L(b,a) U(a,a) = A(b,a): для каждой строки xU = b (задача 2.1.3) */
        for (int i = k + 1; i < N; i++)
            for (int s = i * alpha; s < END(i); s++)
                for (int l = a0; l < a1; l++) {
                    A(s, l) /= A(l, l);
                    for (int q = l + 1; q < a1; q++)
                        A(s, q) -= A(s, l) * A(l, q);
                }

        /* 3. L(a,a) U(a,c) = A(a,c): Алгоритм 27 для каждого столбца блока */
        for (int j = k + 1; j < N; j++)
            for (int l = a0; l < a1; l++)
                for (int s = l + 1; s < a1; s++)
                    for (int q = j * alpha; q < END(j); q++)
                        A(s, q) -= A(s, l) * A(l, q);

        /* 4. A'(b,c) = A(b,c) - L(b,a) U(a,c): основная операция */
        for (int i = k + 1; i < N; i++)
            for (int j = k + 1; j < N; j++)
                for (int s = i * alpha; s < END(i); s++)
                    for (int l = a0; l < a1; l++)
                        for (int q = j * alpha; q < END(j); q++)
                            A(s, q) -= A(s, l) * A(l, q);
    }
}

/* Решение L U x = b (x на входе содержит b) */
static void lu_solve(int n, const double *A, double *x)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++) x[i] -= A(i, j) * x[j];
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) x[i] -= A(i, j) * x[j];
        x[i] /= A(i, i);
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
    /* Сдвиг диагонали не гарантирует строгого диагонального преобладания. */
    for (int i = 0; i < n; i++) A0[(size_t)i * n + i] += n / 4.0;
    for (int i = 0; i < n; i++) {           /* b = A x*, x* = (1,...,1) */
        double s = 0;
        for (int j = 0; j < n; j++) s += A0[(size_t)i * n + j];
        b[i] = s;
    }

    double tmin = 1e30;
    for (int rep = 0; rep < REPS; rep++) {
        for (size_t i = 0; i < (size_t)n * n; i++) A[i] = A0[i];
        double t = wtime();
        block_lu(n, ALPHA, A);
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
    printf("%d,%d,%.6f,%.3e,%.3e\n", n, ALPHA, tmin, err, res / bn);
    fflush(stdout);
    free(A0); free(A); free(b); free(x);
}

int main(int argc, char **argv)
{
    static const int def[] = {100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000};
    printf("n,alpha,time,err,res\n");
    if (argc > 1)
        for (int i = 1; i < argc; i++) run(atoi(argv[i]));
    else
        for (size_t i = 0; i < sizeof def / sizeof *def; i++) run(def[i]);
    return 0;
}
