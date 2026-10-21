"""Запуск программ всех этапов и построение графиков.
Использование: sh build.sh && python plots.py   -> results/*.csv, results/*.png"""
import csv, io, os, subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B, R = "bin", "results"
os.makedirs(R, exist_ok=True)
NS = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000]   # этапы 1-2
NP = [500, 1000, 2000, 3000]                              # этапы 3-4
PS = list(range(1, 9))


def run(cmd, out):
    """Запустить программу, сохранить её CSV-вывод и вернуть строки."""
    text = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    with open(os.path.join(R, out), "w", newline="") as f:
        f.write(text)
    return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(io.StringIO(text))]


def fig(name, xlabel, ylabel, title, logy=False):
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    if logy: plt.yscale("log")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(R, name), dpi=130); plt.close()


def exe(name):
    return os.path.join(B, name)


# ---- Этап 1: погрешность от размерности ----
s1 = run([exe("stage1_seq.exe"), *map(str, NS)], "stage1_seq.csv")
plt.plot([r["n"] for r in s1], [r["err"] for r in s1], "o-", label="||x - x*||∞")
plt.plot([r["n"] for r in s1], [r["res"] for r in s1], "s-", label="||Ax - b||∞ / ||b||∞")
fig("stage1_error.png", "n", "погрешность", "Этап 1. Погрешность блочного LU (GAXPY)", logy=True)

# ---- Этап 2: ускорение от векторизации ----
nv = run([exe("stage2_novec.exe"), *map(str, NS)], "stage2_novec.csv")
v = run([exe("stage2_vec.exe"), *map(str, NS)], "stage2_vec.csv")
plt.plot(NS, [a["time"] / b["time"] for a, b in zip(nv, v)], "o-",
         label="-O3 -ftree-vectorize -mavx2 -mfma")
plt.axhline(1, color="k", lw=.8, ls="--")
fig("stage2_vec_speedup.png", "n", "ускорение T_novec / T_vec", "Этап 2. Ускорение от векторизации")

# ---- Этап 3: автоматическое распараллеливание ----
t3 = {}
for p in PS:
    for r in run([exe(f"stage3_autopar{p}.exe"), *map(str, NP)], f"stage3_autopar{p}.csv"):
        t3[int(r["n"]), p] = r["time"]
for n in NP:
    plt.plot(PS, [t3[n, 1] / t3[n, p] for p in PS], "o-", label=f"n={n}")
plt.plot(PS, PS, "k--", lw=.8, label="идеальное")
fig("stage3_autopar_speedup.png", "число потоков", "ускорение T1 / Tp",
    "Этап 3. Автораспараллеливание gcc")

# ---- Этап 4: MPI ----
t4 = {}
for p in PS:
    for r in run(["mpiexec", "-n", str(p), exe("stage4_mpi.exe"), *map(str, NP)], f"stage4_mpi_p{p}.csv"):
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
