#!/bin/sh
# Сборка программ всех четырёх этапов (MinGW gcc 9.2 + MS-MPI)
set -e
mkdir -p bin
B="-std=gnu99 -msse2 -mfpmath=sse"
M="/c/Program Files (x86)/Microsoft SDKs/MPI"

# Этап 1: последовательная программа
gcc $B -O2 -fno-tree-vectorize stage1_seq.c -o bin/stage1_seq.exe

# Этап 2: одна программа, две сборки - без векторизации и с ней
gcc $B -O2 -fno-tree-vectorize           stage2_vec.c -o bin/stage2_novec.exe
gcc $B -O3 -ftree-vectorize -mavx2 -mfma stage2_vec.c -o bin/stage2_vec.exe

# Этап 3: автоматическое распараллеливание, P = 1..8 потоков
for p in 1 2 3 4 5 6 7 8; do
  gcc $B -O2 -fno-tree-vectorize -ftree-parallelize-loops=$p -DNTHREADS=$p \
      stage3_autopar.c -o bin/stage3_autopar$p.exe
done

# Этап 4: MPI
gcc $B -O2 -fno-tree-vectorize -Iinclude -I"$M/Include" -D"__int64=long long" \
    stage4_mpi.c -L"$M/Lib/x86" -lmsmpi -o bin/stage4_mpi.exe
echo "build ok"
