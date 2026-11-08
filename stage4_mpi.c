/*
 * Этап 4. MPI: блочное LU-разложение (Алгоритм 37) на распределённой памяти.
 * Блочные строки матрицы циклически распределены по процессам: блок i хранится
 * только у процесса i % size. На итерации k владелец блочной строки k выполняет
 * шаги 1 и 3 и рассылает её, остальные шаги каждый делает над своими строками.
 * Генерация матрицы, проверка решения и вывод CSV — только на процессе 0.
 * Сборка и запуск — README.md; алгоритм и методика замеров — REPORT.md.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ALPHA 32
#define REPS  3
static int rank, size;

#define END(i) ((i) * alpha + alpha < n ? (i) * alpha + alpha : n)
/* Начало строк блока i в локальном массиве владельца: все предшествующие
 * блоки этого процесса полные, короче может быть только последний блок матрицы. */
#define LOCAL_ROW(i) ((size_t)((i) / size) * alpha * n)

static inline void saxpy(int m, double s, const double *restrict x, double *restrict y)
{
    for (int q = 0; q < m; q++)
        y[q] -= s * x[q];
}

/* A — локальные блочные строки этого процесса, по n элементов в строке. */
void block_lu_mpi(int n, int alpha, double *A)
{
    int N = (n + alpha - 1) / alpha;
    double *W = malloc((size_t)alpha * n * sizeof(double));   /* блочная строка k */

    for (int k = 0; k < N; k++) {
        int a0 = k * alpha, w = END(k) - a0, t = n - a0;      /* W: w строк по t чисел */
        int owner = k % size;

        if (rank == owner) {
            double *Ak = A + LOCAL_ROW(k);
            /* 1. A(a,a) = L(a,a) U(a,a) */
            for (int l = 0; l < w; l++) {
                const double *rl = Ak + (size_t)l * n + a0;
                for (int s = l + 1; s < w; s++) {
                    double *rs = Ak + (size_t)s * n + a0;
                    rs[l] /= rl[l];
                    saxpy(w - l - 1, rs[l], rl + l + 1, rs + l + 1);
                }
            }
            /* 3. L(a,a) U(a,c) = A(a,c) */
            for (int j = k + 1; j < N; j++) {
                int c0 = j * alpha, m = END(j) - c0;
                for (int l = 0; l < w; l++) {
                    const double *rl = Ak + (size_t)l * n;
                    for (int s = l + 1; s < w; s++) {
                        double *rs = Ak + (size_t)s * n;
                        saxpy(m, rs[a0 + l], rl + c0, rs + c0);
                    }
                }
            }
            for (int s = 0; s < w; s++)
                for (int q = 0; q < t; q++)
                    W[(size_t)s * t + q] = Ak[(size_t)s * n + a0 + q];
        }
        /* После последнего блока строк ниже нет: рассылать некому и незачем. */
        if (k == N - 1) break;
        MPI_Bcast(W, w * t, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        /* Первый свой блок ниже k: i >= k + 1, i % size == rank */
        for (int i = k + 1 + ((rank - k - 1) % size + size) % size; i < N; i += size) {
            double *Ai = A + LOCAL_ROW(i);
            int h = END(i) - i * alpha;
            /* 2. L(b,a) U(a,a) = A(b,a) */
            for (int s = 0; s < h; s++) {
                double *rs = Ai + (size_t)s * n + a0;
                for (int l = 0; l < w; l++) {
                    const double *ul = W + (size_t)l * t;
                    rs[l] /= ul[l];
                    saxpy(w - l - 1, rs[l], ul + l + 1, rs + l + 1);
                }
            }
            /* 4. A'(b,c) = A(b,c) - L(b,a) U(a,c) */
            for (int j = k + 1; j < N; j++) {
                int c0 = j * alpha, m = END(j) - c0;
                for (int s = 0; s < h; s++) {
                    double *rs = Ai + (size_t)s * n;
                    for (int l = 0; l < w; l++)
                        saxpy(m, rs[a0 + l], W + (size_t)l * t + c0 - a0, rs + c0);
                }
            }
        }
    }
    free(W);
}

/* Раздать блочные строки A0 (процесс 0) по локальным массивам A. Вне замера. */
static void scatter_rows(int n, int alpha, const double *A0, double *A)
{
    int N = (n + alpha - 1) / alpha;
    for (int i = 0; i < N; i++) {
        int h = END(i) - i * alpha, owner = i % size;
        if (rank == 0 && owner == 0) {
            for (size_t e = 0; e < (size_t)h * n; e++)
                A[LOCAL_ROW(i) + e] = A0[(size_t)i * alpha * n + e];
        } else if (rank == 0) {
            MPI_Send(A0 + (size_t)i * alpha * n, h * n, MPI_DOUBLE, owner, i, MPI_COMM_WORLD);
        } else if (rank == owner) {
            MPI_Recv(A + LOCAL_ROW(i), h * n, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

/* Собрать разложение на процессе 0, чтобы проверить решение. Вне замера. */
static void gather_rows(int n, int alpha, const double *A, double *LU)
{
    int N = (n + alpha - 1) / alpha;
    for (int i = 0; i < N; i++) {
        int h = END(i) - i * alpha, owner = i % size;
        if (rank == 0 && owner == 0) {
            for (size_t e = 0; e < (size_t)h * n; e++)
                LU[(size_t)i * alpha * n + e] = A[LOCAL_ROW(i) + e];
        } else if (rank == 0) {
            MPI_Recv(LU + (size_t)i * alpha * n, h * n, MPI_DOUBLE, owner, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == owner) {
            MPI_Send(A + LOCAL_ROW(i), h * n, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
        }
    }
}

static void run(int n)
{
    int alpha = ALPHA, N = (n + alpha - 1) / alpha, rows = 0;
    for (int i = rank; i < N; i += size) rows += END(i) - i * alpha;
    /* При числе процессов больше N у процесса может не быть строк. */
    double *A = malloc(((size_t)rows * n + 1) * sizeof(double));
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
        scatter_rows(n, alpha, A0, A);
        MPI_Barrier(MPI_COMM_WORLD);
        double t = MPI_Wtime();
        block_lu_mpi(n, alpha, A);
        /* Конечный барьер входит во время: ждём самый медленный процесс. */
        MPI_Barrier(MPI_COMM_WORLD);
        t = MPI_Wtime() - t;
        if (t < tmin) tmin = t;
    }

    double *LU = rank == 0 ? malloc((size_t)n * n * sizeof(double)) : NULL;
    gather_rows(n, alpha, A, LU);

    if (rank == 0) {
        double *x = malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) x[i] = b[i];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < i; j++) x[i] -= LU[(size_t)i * n + j] * x[j];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) x[i] -= LU[(size_t)i * n + j] * x[j];
            x[i] /= LU[(size_t)i * n + i];
        }
        double err = 0, res = 0, bn = 0;
        for (int i = 0; i < n; i++) {
            err = fmax(err, fabs(x[i] - 1.0));
            double s = -b[i];
            for (int j = 0; j < n; j++) s += A0[(size_t)i * n + j] * x[j];
            res = fmax(res, fabs(s));
            bn = fmax(bn, fabs(b[i]));
        }
        printf("%d,%d,%d,%.6f,%.3e,%.3e\n", size, n, alpha, tmin, err, res / bn);
        fflush(stdout);
        free(x); free(A0); free(b); free(LU);
    }
    free(A);
}

int main(int argc, char **argv)
{
    static const int def[] = {500, 1000, 2000, 3000};
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) printf("procs,n,alpha,time,err,res\n");
    if (argc > 1)
        for (int i = 1; i < argc; i++) run(atoi(argv[i]));
    else
        for (size_t i = 0; i < sizeof def / sizeof *def; i++) run(def[i]);
    MPI_Finalize();
    return 0;
}
