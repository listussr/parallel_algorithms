#!/bin/sh
# Сборка программ всех четырёх этапов.
#   Windows (Git Bash): MinGW gcc 9.2 + MS-MPI, файлы bin/*.exe
#   Linux / macOS:      gcc или clang + Open MPI / MPICH, файлы bin/* без расширения
# Компилятор можно переопределить: CC=gcc-14 sh build.sh
set -e
mkdir -p bin

# Флаги «без векторизации»: clang, в отличие от gcc, векторизует ещё и SLP-проходом
novec() {
    if $1 --version 2>/dev/null | grep -qi clang; then
        echo "-O2 -fno-tree-vectorize -fno-slp-vectorize"
    else
        echo "-O2 -fno-tree-vectorize"
    fi
}

case "$(uname -s)" in
MINGW*|MSYS*|CYGWIN*) OS=windows ;;
Darwin)               OS=macos ;;
*)                    OS=linux ;;
esac

if [ $OS = windows ]; then
    CC=${CC:-gcc}; X=.exe; LM=
    # 32-битный MinGW по умолчанию считает на x87; -mfpmath=sse даёт ту же точность, что на x86-64
    B="-std=gnu99 -msse2 -mfpmath=sse"
else
    CC=${CC:-cc}; X=; LM=-lm
    B="-std=gnu99"
fi

O=$(novec $CC)

# Флаги векторизации: AVX2+FMA на x86, на ARM (Apple M*, aarch64) NEON и FMA есть в базовом наборе
case "$(uname -m)" in
x86_64|amd64|i?86) V="-mavx2 -mfma" ;;
*)                 V= ;;
esac

# OpenMP: Apple clang не знает -fopenmp, нужна libomp из Homebrew
OMP=-fopenmp
if [ $OS = macos ] && $CC --version 2>/dev/null | grep -qi clang; then
    P=$(brew --prefix libomp)
    OMP="-Xpreprocessor -fopenmp -I$P/include -L$P/lib -lomp"
fi

# Этап 1: последовательная программа
$CC $B $O stage1_seq.c -o bin/stage1_seq$X $LM

# Этап 2: одна программа, две сборки - без векторизации и с ней
$CC $B $O                    stage2_vec.c -o bin/stage2_novec$X $LM
$CC $B -O3 -ftree-vectorize $V stage2_vec.c -o bin/stage2_vec$X $LM

# Этап 3: OpenMP (#pragma omp parallel for); число потоков — OMP_NUM_THREADS при запуске
$CC $B $O $OMP stage3_omp.c -o bin/stage3_omp$X $LM

# Этап 4: MPI
if [ $OS = windows ]; then
    # Путь в формате Git Bash; Lib/x86 ниже требует компилятор той же разрядности.
    M="/c/Program Files (x86)/Microsoft SDKs/MPI"
    $CC $B $O -Iinclude -I"$M/Include" -D"__int64=long long" \
        stage4_mpi.c -L"$M/Lib/x86" -lmsmpi -o bin/stage4_mpi.exe
else
    mpicc $B $(novec mpicc) stage4_mpi.c -o bin/stage4_mpi $LM
fi
echo "build ok ($OS, $(uname -m))"
