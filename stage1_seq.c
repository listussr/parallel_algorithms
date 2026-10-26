/*
 * Этап 1. Последовательное блочное LU через GAXPY (left-looking, jki).
 * Единичная диагональ L не записывается.
 * Выбора ведущего элемента и проверки нулевых пивотов нет.
 * Сборка и запуск — README.md; алгоритм и методика замеров — REPORT.md.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>

#define R    32   /* размер блока */
#define REPS 3    /* число повторов, берётся минимальное время */

static double wtime(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

/* A хранится по строкам: ниже диагонали — L, на диагонали и выше — U. */
#define a(i, j) A[(size_t)(i) * n + (j)]

void block_lu_gaxpy(int n, int r, double *A)
{
    for (int k = 0; k < n; k += r) {
        int w = (k + r <= n) ? r : n - k;

        /* 1) U(0:k,K) = L11^{-1} A(0:k,K):  A(i,K) -= L(i,j) * U(j,K), j < i */
        for (int i = 1; i < k; i++)
            for (int j = 0; j < i; j++) {
                double l = a(i, j);
                for (int c = k; c < k + w; c++)
                    a(i, c) -= l * a(j, c);
            }

        /* 2) GAXPY: A(i,K) -= L(i,0:k) * U(0:k,K) для всех строк i >= k */
        for (int i = k; i < n; i++)
            for (int j = 0; j < k; j++) {
                double l = a(i, j);
                for (int c = k; c < k + w; c++)
                    a(i, c) -= l * a(j, c);
            }

        /* 3) LU панели A(k:n,K): столбец c = gaxpy предыдущих столбцов панели */
        for (int c = k; c < k + w; c++) {
            for (int j = k; j < c; j++) {
                double u = a(j, c);
                for (int i = j + 1; i < n; i++)
                    a(i, c) -= a(i, j) * u;
            }
            double p = 1.0 / a(c, c);
            for (int i = c + 1; i < n; i++)
                a(i, c) *= p;
        }
    }
}

/* Решение L U x = b (x на входе содержит b) */
static void lu_solve(int n, const double *A, double *x)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++) x[i] -= a(i, j) * x[j];
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) x[i] -= a(i, j) * x[j];
        x[i] /= a(i, i);
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
    printf("%d,%d,%.6f,%.3e,%.3e\n", n, R, tmin, err, res / bn);
    fflush(stdout);
    free(A0); free(A); free(b); free(x);
}

int main(int argc, char **argv)
{
    static const int def[] = {100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000};
    printf("n,r,time,err,res\n");
    if (argc > 1)
        for (int i = 1; i < argc; i++) run(atoi(argv[i]));
    else
        for (size_t i = 0; i < sizeof def / sizeof *def; i++) run(def[i]);
    return 0;
}
