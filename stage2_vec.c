/*
 * Этап 2. Блочное LU через GAXPY с плотными буферами U и P.
 * В треугольном решении и GAXPY внутренний цикл идёт по соседним элементам;
 * restrict описывает отсутствие перекрытия данных для автовекторизатора.
 * Сравниваются две сборки этого файла. Флаги и запуск — README.md, REPORT.md.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>

#define R    32
#define REPS 3

static double wtime(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

/* L11 * U = U_исх; L11 — нижняя часть A с единичной диагональю.
 * Строка U(i,:) зависит от уже вычисленных U(j,:), j < i. */
static void trsm(int n, int k, int w, const double *restrict A, double *restrict U)
{
    for (int i = 1; i < k; i++) {
        const double *restrict li = A + (size_t)i * n;
        double *restrict ui = U + (size_t)i * w;
        for (int j = 0; j < i; j++) {
            double l = li[j];
            const double *restrict uj = U + (size_t)j * w;
            for (int c = 0; c < w; c++)
                ui[c] -= l * uj[c];
        }
    }
}

/* 2) P(m x w) -= L(k:n, 0:k) * U(k x w) - блочный GAXPY */
static void gaxpy(int n, int k, int m, int w, const double *restrict A,
                  const double *restrict U, double *restrict P)
{
    for (int i = 0; i < m; i++) {
        const double *restrict li = A + (size_t)(k + i) * n;
        double *restrict pi = P + (size_t)i * w;
        for (int j = 0; j < k; j++) {
            double l = li[j];
            const double *restrict uj = U + (size_t)j * w;
            for (int c = 0; c < w; c++)
                pi[c] -= l * uj[c];
        }
    }
}

/* 3) LU панели P (m x w) в jki форме */
static void panel_lu(int m, int w, double *restrict P)
{
    for (int c = 0; c < w; c++) {
        for (int j = 0; j < c; j++) {
            double u = P[(size_t)j * w + c];
            for (int i = j + 1; i < m; i++)
                P[(size_t)i * w + c] -= P[(size_t)i * w + j] * u;
        }
        double p = 1.0 / P[(size_t)c * w + c];
        for (int i = c + 1; i < m; i++)
            P[(size_t)i * w + c] *= p;
    }
}

/* A (n x n, по строкам) заменяется на LU без выбора ведущего элемента.
 * Буферы вмещают n*r элементов, но текущий шаг строки в них равен w. */
void block_lu_gaxpy(int n, int r, double *A)
{
    double *U = malloc((size_t)n * r * sizeof(double));
    double *P = malloc((size_t)n * r * sizeof(double));
    for (int k = 0; k < n; k += r) {
        int w = (k + r <= n) ? r : n - k, m = n - k;

        for (int i = 0; i < k; i++)
            for (int c = 0; c < w; c++) U[(size_t)i * w + c] = A[(size_t)i * n + k + c];
        trsm(n, k, w, A, U);
        for (int i = 0; i < k; i++)
            for (int c = 0; c < w; c++) A[(size_t)i * n + k + c] = U[(size_t)i * w + c];

        for (int i = 0; i < m; i++)
            for (int c = 0; c < w; c++) P[(size_t)i * w + c] = A[(size_t)(k + i) * n + k + c];
        gaxpy(n, k, m, w, A, U, P);
        panel_lu(m, w, P);
        for (int i = 0; i < m; i++)
            for (int c = 0; c < w; c++) A[(size_t)(k + i) * n + k + c] = P[(size_t)i * w + c];
    }
    free(U); free(P);
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
