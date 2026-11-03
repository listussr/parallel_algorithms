"""Запустить все эксперименты и перезаписать их CSV и PNG в results/.

Запуск из корня репозитория после sh build.sh: python plots.py (python3 на Linux/macOS).
Требуются Matplotlib и доступная в PATH команда mpiexec (MS-MPI, Open MPI или MPICH).
Готовые CSV скрипт не переиспользует.
"""
import csv, io, os, platform, subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B, R = "bin", "results"
os.makedirs(R, exist_ok=True)
NS = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000]   # этапы 1-2
NP = [500, 1000, 2000, 3000]                              # этапы 3-4
PS = list(range(1, 9))


def run(cmd, out, env=None):
    """Запустить программу, сохранить её CSV-вывод и вернуть строки."""
    text = subprocess.run(cmd, capture_output=True, text=True, check=True,
                          env=None if env is None else {**os.environ, **env}).stdout
    with open(os.path.join(R, out), "w", newline="") as f:
        f.write(text)
    return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(io.StringIO(text))]


def fig(name, xlabel, ylabel, title, logy=False):
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    if logy: plt.yscale("log")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(R, name), dpi=130); plt.close()


def exe(name):
    """Путь к программе: bin/name.exe на Windows, bin/name на Linux/macOS."""
    return os.path.join(B, name + (".exe" if os.name == "nt" else ""))


def mpiexec(p):
    """Open MPI по умолчанию не запускает процессов больше, чем ядер."""
    out = subprocess.run(["mpiexec", "--version"], capture_output=True, text=True).stdout
    return ["mpiexec", "-n", str(p)] + (["--oversubscribe"] if "Open MPI" in out else [])


X86 = platform.machine().lower() in ("x86_64", "amd64", "i386", "i686")


# ---- Этап 1: погрешность от размерности ----
s1 = run([exe("stage1_seq"), *map(str, NS)], "stage1_seq.csv")
plt.plot([r["n"] for r in s1], [r["err"] for r in s1], "o-", label="||x - x*||∞")
plt.plot([r["n"] for r in s1], [r["res"] for r in s1], "s-", label="||Ax - b||∞ / ||b||∞")
fig("stage1_error.png", "n", "погрешность", "Этап 1. Погрешность блочного LU (GAXPY)", logy=True)

# ---- Этап 2: ускорение от векторизации ----
nv = run([exe("stage2_novec"), *map(str, NS)], "stage2_novec.csv")
v = run([exe("stage2_vec"), *map(str, NS)], "stage2_vec.csv")
plt.plot(NS, [a["time"] / b["time"] for a, b in zip(nv, v)], "o-",
         label="-O3 -ftree-vectorize" + (" -mavx2 -mfma" if X86 else " (NEON)"))
plt.axhline(1, color="k", lw=.8, ls="--")
fig("stage2_vec_speedup.png", "n", "ускорение T_novec / T_vec", "Этап 2. Ускорение от векторизации")

# ---- Этап 3: OpenMP, база — та же программа с одним потоком ----
t3 = {}
for p in PS:
    for r in run([exe("stage3_omp"), *map(str, NP)], f"stage3_omp_t{p}.csv",
                 env={"OMP_NUM_THREADS": str(p)}):
        t3[int(r["n"]), p] = r["time"]
for n in NP:
    plt.plot(PS, [t3[n, 1] / t3[n, p] for p in PS], "o-", label=f"n={n}")
plt.plot(PS, PS, "k--", lw=.8, label="идеальное")
fig("stage3_omp_speedup_p.png", "число потоков", "ускорение T1 / Tp",
    "Этап 3. OpenMP: ускорение от числа потоков")
for p in [2, 4, 6, 8]:
    plt.plot(NP, [t3[n, 1] / t3[n, p] for n in NP], "o-", label=f"p={p}")
fig("stage3_omp_speedup_n.png", "n", "ускорение T1 / Tp", "Этап 3. OpenMP: ускорение от размера матрицы")

# ---- Этап 4: база — MPI с одним процессом, включая его накладные расходы ----
t4 = {}
for p in PS:
    for r in run([*mpiexec(p), exe("stage4_mpi"), *map(str, NP)], f"stage4_mpi_p{p}.csv"):
        t4[int(r["n"]), p] = r["time"]
for n in NP:
    plt.plot(PS, [t4[n, 1] / t4[n, p] for p in PS], "o-", label=f"n={n}")
plt.plot(PS, PS, "k--", lw=.8, label="идеальное")
fig("stage4_mpi_speedup_p.png", "число процессов", "ускорение T1 / Tp",
    "Этап 4. MPI: ускорение от числа процессов")
for p in [2, 4, 6, 8]:
    plt.plot(NP, [t4[n, 1] / t4[n, p] for n in NP], "o-", label=f"p={p}")
fig("stage4_mpi_speedup_n.png", "n", "ускорение T1 / Tp", "Этап 4. MPI: ускорение от размера матрицы")
print("done")
