"""Трассировки программ на примере 6x6 (alpha = 2) для гайда docs/07_guide.md.

Скрипт берёт настоящие stage1_seq.c ... stage4_mpi.c, вставляет в их копии только вывод
(printf) в отмеченных местах, собирает копии вместе с небольшим main и сохраняет вывод в
docs/trace/*.txt. Исходники проекта не меняются.

Запуск из корня репозитория:  python docs/trace/make_traces.py
Нужны gcc с OpenMP и MPI (как для build.sh). Пути к MS-MPI — как в build.sh.
"""
import os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "trace")
WIN = os.name == "nt"
EXE = ".exe" if WIN else ""
MSMPI = r"C:\Program Files (x86)\Microsoft SDKs\MPI"
B = ["-std=gnu99", "-O0"] + (["-msse2", "-mfpmath=sse"] if WIN else [])

# Матрица примера: A = L U, L и U с целыми элементами.
EXAMPLE = r"""
static const double Lx[6][6] = {{1,0,0,0,0,0},{2,1,0,0,0,0},{-1,1,1,0,0,0},
                                {1,0,2,1,0,0},{0,2,-1,1,1,0},{1,-1,1,0,2,1}};
static const double Ux[6][6] = {{2,1,-1,0,1,2},{0,3,1,2,0,-1},{0,0,2,1,-1,0},
                                {0,0,0,4,1,1},{0,0,0,0,3,-2},{0,0,0,0,0,2}};
static void example(double *M)
{
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
            double s = 0;
            for (int t = 0; t < 6; t++) s += Lx[i][t] * Ux[t][j];
            M[i * 6 + j] = s;
        }
}
static void print_matrix(const char *title, const double *M, int n)
{
    printf("%s\n", title);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) printf("%5g", M[i * n + j]);
        printf("\n");
    }
}
"""


def patch(src, pairs):
    """Заменить каждую строку-образец на новую; k-е вхождение образца -> k-я замена из списка."""
    for find, repls in pairs:
        repls = repls if isinstance(repls, list) else [repls]
        parts = src.split(find)
        assert len(parts) - 1 == len(repls), (find, len(parts) - 1)
        out = parts[0]
        for r, p in zip(repls, parts[1:]):
            out += r + p
        src = out
    return src


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def build_run(tag, code, extra=(), run=None, env=None):
    with tempfile.TemporaryDirectory() as td:
        c = os.path.join(td, tag + ".c")
        with open(c, "w", encoding="utf-8") as f:
            f.write(code)
        exe = os.path.join(td, tag + EXE)
        cc = ["gcc"] if not extra or "mpi" not in tag or WIN else ["mpicc"]
        subprocess.run(cc + B + [c, "-o", exe] + list(extra), check=True)
        cmd = (run or []) + [exe]
        r = subprocess.run(cmd, capture_output=True, check=True, env={**os.environ, **(env or {})})
        return r.stdout.decode("utf-8", "replace").replace("\r\n", "\n")


def save(name, text):
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("записано", name, text.count("\n"), "строк")


# ---------------------------------------------------------------- этап 1: каждая операция
def stage1():
    src = read("stage1_seq.c")
    div = "A(s, l) /= A(l, l);"
    sub = "A(s, q) -= A(s, l) * A(l, q);"
    d = lambda st, ctx, cv: ("{ double o_ = A(s, l); A(s, l) /= A(l, l); "
                             f'printf("k=%d шаг {st}{ctx} l=%d s=%d:      A(%d,%d) = %g / %g = %g\\n", k{cv}, l, s, s, l, o_, A(l, l), A(s, l)); }}')
    m = lambda st, ctx, cv: ("{ double o_ = A(s, q); A(s, q) -= A(s, l) * A(l, q); "
                             f'printf("k=%d шаг {st}{ctx} l=%d s=%d q=%d: A(%d,%d) = %g - (%g)*(%g) = %g\\n", k{cv}, l, s, q, s, q, o_, A(s, l), A(l, q), A(s, q)); }}')
    src = patch(src, [
        (div, [d(1, "", ""), d(2, " i=%d", ", i")]),
        (sub, [m(1, "", ""), m(2, " i=%d", ", i"), m(3, " j=%d", ", j"), m(4, " i=%d j=%d", ", i, j")]),
    ])
    code = ("#define main stage_main\n" + src + "\n#undef main\n" + EXAMPLE + r"""
int main(void)
{
    double M[36];
    example(M);
    print_matrix("исходная A:", M, 6);
    block_lu(6, 2, M);
    print_matrix("после block_lu (L ниже диагонали, U на диагонали и выше):", M, 6);
    return 0;
}
""")
    save("stage1_trace.txt", build_run("stage1", code, ["-lm"] if not WIN else []))


# ---------------------------------------------------------------- этап 2: вызовы saxpy
def stage2():
    src = read("stage2_vec.c")
    sig = "static inline void saxpy(int m, double s, const double *restrict x, double *restrict y)\n{\n"
    src = patch(src, [
        (sig, sig + "    tr_saxpy(m, s, x, y);\n"),
        ("        /* 1. A(a,a) = L(a,a) U(a,a) */\n", "        /* 1. A(a,a) = L(a,a) U(a,a) */\n        tr_step(1, k);\n"),
        ("        /* 2. L(b,a) U(a,a) = A(b,a) */\n", "        /* 2. L(b,a) U(a,a) = A(b,a) */\n        tr_step(2, k);\n"),
        ("        /* 3. L(a,a) U(a,c) = A(a,c) */\n", "        /* 3. L(a,a) U(a,c) = A(a,c) */\n        tr_step(3, k);\n"),
        ("        /* 4. A'(b,c) = A(b,c) - L(b,a) U(a,c) */\n", "        /* 4. A'(b,c) = A(b,c) - L(b,a) U(a,c) */\n        tr_step(4, k);\n"),
    ])
    pre = r"""
#include <stdio.h>
static const double *TR_BASE; static int TR_N;
static void tr_step(int st, int k) { printf("--- k=%d, шаг %d\n", k, st); }
static void tr_saxpy(int m, double s, const double *x, double *y)
{
    long ox = x - TR_BASE, oy = y - TR_BASE;
    if (m <= 0) return;
    printf("saxpy(m=%d, s=%g, x = строка %ld столбцы %ld..%ld, y = строка %ld столбцы %ld..%ld):",
           m, s, ox / TR_N, ox % TR_N, ox % TR_N + m - 1, oy / TR_N, oy % TR_N, oy % TR_N + m - 1);
    printf("  y = (");
    for (int q = 0; q < m; q++) printf(q ? ", %g" : "%g", y[q]);
    printf(") - %g*(", s);
    for (int q = 0; q < m; q++) printf(q ? ", %g" : "%g", x[q]);
    printf(") = (");
    for (int q = 0; q < m; q++) printf(q ? ", %g" : "%g", y[q] - s * x[q]);
    printf(")\n");
}
"""
    code = pre + "#define main stage_main\n" + src + "\n#undef main\n" + EXAMPLE + r"""
int main(void)
{
    double M[36];
    example(M);
    TR_BASE = M; TR_N = 6;
    block_lu(6, 2, M);
    print_matrix("после block_lu:", M, 6);
    return 0;
}
"""
    save("stage2_saxpy.txt", build_run("stage2", code, ["-lm"] if not WIN else []))


# ---------------------------------------------------------------- этап 3: кто какой блок считает
def stage3():
    src = read("stage3_omp.c")
    c0 = "int c0 = j * alpha, m = END(j) - c0;"
    src = patch(src, [
        ("for (int s = i * alpha; s < END(i); s++)\n                for (int l = a0; l < a1; l++) {",
         "for (int s = (tr_omp(2, k, i, -1), i * alpha); s < END(i); s++)\n                for (int l = a0; l < a1; l++) {"),
        (c0, [c0 + " tr_omp(3, k, -1, j);", c0 + " tr_omp(4, k, i, j);"]),
    ])
    pre = r"""
#include <stdio.h>
#include <omp.h>
static int tr_omp(int st, int k, int i, int j)
{
    #pragma omp critical
    {
        if (st == 2) printf("k=%d шаг 2 блок i=%d -> поток %d\n", k, i, omp_get_thread_num());
        if (st == 3) printf("k=%d шаг 3 блок j=%d -> поток %d\n", k, j, omp_get_thread_num());
        if (st == 4) printf("k=%d шаг 4 пара (%d,%d) -> поток %d\n", k, i, j, omp_get_thread_num());
    }
    return 0;
}
"""
    main6 = r"""
int main(void)
{
    double M[36];
    example(M);
    block_lu(6, 2, M);
    print_matrix("после block_lu:", M, 6);
    return 0;
}
"""
    main12 = r"""
int main(void)
{
    enum { n = 12 };
    static double M[n * n];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) M[i * n + j] = (i == j) ? n : ((i * 7 + j * 3) % 5) - 2;
    block_lu(n, 2, M);
    return 0;
}
"""
    def sorted_run(main, threads):
        code = pre + "#define main stage_main\n" + src + "\n#undef main\n" + EXAMPLE + main
        out = build_run("stage3", code, ["-fopenmp"] + (["-lm"] if not WIN else []),
                        env={"OMP_NUM_THREADS": str(threads)})
        lines = out.splitlines()
        tr = [l for l in lines if l.startswith("k=")]
        rest = [l for l in lines if not l.startswith("k=")]
        key = lambda l: tuple(int(x) for x in re.findall(r"-?\d+", l.split("->")[0]))
        return "\n".join(sorted(tr, key=key) + rest) + "\n"
    save("stage3_threads_n6_p2.txt", sorted_run(main6, 2))
    save("stage3_threads_n12_p3.txt", sorted_run(main12, 3))


# ---------------------------------------------------------------- этап 4: рассылки и блоки процессов
def stage4(p):
    src = read("stage4_mpi.c")
    loop = "for (int i = k + 1 + ((rank - k - 1) % size + size) % size; i < N; i += size) {\n"
    src = patch(src, [
        ("#include <mpi.h>\n", "#include <mpi.h>\n#define MPI_Bcast tr_bcast\n"),
        (loop, loop + "            printf(\"k=%d процесс %d: шаги 2 и 4 для своего блока %d (строки %d..%d)\\n\", k, rank, i, i * alpha, END(i) - 1); fflush(stdout);\n"),
    ])
    pre = r"""
#include <stdio.h>
#include <mpi.h>
static int tr_rank_, tr_k_ = 0;
static int tr_bcast(void *buf, int count, MPI_Datatype dt, int root, MPI_Comm comm)
{
    if (tr_rank_ == root) {
        const double *w = buf;
        int t = count / 2;
        printf("k=%d процесс %d: шаги 1 и 3 выполнены, рассылаю W (%d x %d = %d чисел):", tr_k_, root, 2, t, count);
        for (int r = 0; r < 2; r++) {
            printf("\n    ");
            for (int q = 0; q < t; q++) printf("%5g", w[r * t + q]);
        }
        printf("\n");
        fflush(stdout);
    }
    tr_k_++;
    return PMPI_Bcast(buf, count, dt, root, comm);
}
"""
    main = r"""
static void show_local(const char *what, const double *A, int n, int alpha)
{
    int N = (n + alpha - 1) / alpha;
    for (int i = rank; i < N; i += size)
        for (int s = i * alpha; s < END(i); s++) {
            printf("%s процесс %d: глобальная строка %d (локальная %d):", what, rank,
                   s, (int)(LOCAL_ROW(i) / n) + s - i * alpha);
            for (int q = 0; q < n; q++) printf("%5g", A[LOCAL_ROW(i) + (size_t)(s - i * alpha) * n + q]);
            printf("\n");
        }
    fflush(stdout);
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    tr_rank_ = rank;
    int n = 6, alpha = 2;
    double A0[36], LU[36], A[36];
    example(A0);
    scatter_rows(n, alpha, A0, A);
    show_local("до", A, n, alpha);
    MPI_Barrier(MPI_COMM_WORLD);
    block_lu_mpi(n, alpha, A);
    MPI_Barrier(MPI_COMM_WORLD);
    show_local("после", A, n, alpha);
    gather_rows(n, alpha, A, LU);
    if (rank == 0) print_matrix("собрано на процессе 0:", LU, n);
    MPI_Finalize();
    return 0;
}
"""
    code = pre + "#define main stage_main\n" + src + "\n#undef main\n" + EXAMPLE + main
    extra = (["-Iinclude", "-I" + os.path.join(MSMPI, "Include"), "-D__int64=long long",
              "-L" + os.path.join(MSMPI, "Lib", "x86"), "-lmsmpi"] if WIN else ["-lm"])
    if WIN:
        extra[0] = "-I" + os.path.join(ROOT, "include")
    out = build_run("stage4mpi", code, extra, run=["mpiexec", "-n", str(p)])
    lines = out.splitlines()
    # вывод процессов перемешан: «до» и «после» упорядочиваем по строкам, события — по итерациям
    def key(l):
        m = re.match(r"(до|после) процесс \d+: глобальная строка (\d+)", l)
        if m:
            return (0 if m.group(1) == "до" else 2, int(m.group(2)), 0)
        m = re.match(r"k=(\d+) процесс (\d+): (шаги 1)?", l)
        if m:
            return (1, int(m.group(1)), 0 if m.group(3) else 1 + int(m.group(2)))
        return (3, 0, 0)
    blocks, cur = [], []
    for l in lines:                      # строки матрицы W идут следом за заголовком рассылки
        if l.startswith("    ") and cur:
            cur[-1] += "\n" + l
        else:
            cur.append(l)
    tail = [l for l in cur if key(l)[0] == 3]
    body = sorted([l for l in cur if key(l)[0] != 3], key=key)
    save(f"stage4_mpi_p{p}.txt", "\n".join(body + tail) + "\n")


if __name__ == "__main__":
    stage1(); stage2(); stage3(); stage4(2); stage4(3)
