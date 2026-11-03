#!/bin/sh
# Сборка программ всех четырёх этапов (MinGW gcc 9.2 + MS-MPI)
set -e
mkdir -p bin
B="-std=gnu99 -msse2 -mfpmath=sse"
# Путь в формате Git Bash; Lib/x86 ниже требует компилятор той же разрядности.
M="/c/Program Files (x86)/Microsoft SDKs/MPI"

# Этап 1: последовательная программа
gcc $B -O2 -fno-tree-vectorize stage1_seq.c -o bin/stage1_seq.exe

# Этап 2: одна программа, две сборки - без векторизации и с ней
gcc $B -O2 -fno-tree-vectorize           stage2_vec.c -o bin/stage2_novec.exe
gcc $B -O3 -ftree-vectorize -mavx2 -mfma stage2_vec.c -o bin/stage2_vec.exe

# Этап 3: OpenMP (#pragma omp parallel for); число потоков — OMP_NUM_THREADS при запуске
gcc $B -O2 -fno-tree-vectorize -fopenmp stage3_omp.c -o bin/stage3_omp.exe

# Этап 4: MPI
gcc $B -O2 -fno-tree-vectorize -Iinclude -I"$M/Include" -D"__int64=long long" \
    stage4_mpi.c -L"$M/Lib/x86" -lmsmpi -o bin/stage4_mpi.exe
echo "build ok"
