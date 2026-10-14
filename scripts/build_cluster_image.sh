#!/bin/sh
# Build the ollama-cluster image natively on an NVIDIA DGX Spark.
#
# Needs: CUDA 13 toolkit (/usr/local/cuda), cmake >= 3.24, gcc/g++,
# libibverbs-dev (RDMA for the RPC link), Go (version from go.mod), Docker.
#
#   scripts/build_cluster_image.sh                    # tag ollama-cluster:<version>
#   VERSION=0.40.3-cluster scripts/build_cluster_image.sh
set -eu

cd "$(dirname "$0")/.."
VERSION=${VERSION:-$(git describe --tags --always 2>/dev/null | sed 's/^v//')-cluster}
TAG=${TAG:-ollama-cluster:$VERSION}
JOBS=${JOBS:-$(nproc)}
export PATH=/usr/local/cuda/bin:$PATH

for tool in cmake go nvcc docker; do
    command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 1; }
done
[ -e /usr/include/infiniband/verbs.h ] || { echo "missing: libibverbs-dev (sudo apt install libibverbs-dev)" >&2; exit 1; }

# llama-server, the RPC worker (ggml-rpc-server, libggml-rpc) and the CPU
# backend. GGML_CPU_ALL_VARIANTS is off because the host gcc does not know
# every arm64 variant (+sme); the native CPU backend matches the DGX Spark.
cmake -S llama/server --preset cpu -DOLLAMA_RPC=ON -DGGML_CPU_ALL_VARIANTS=OFF
cmake --build build/llama-server-cpu -j "$JOBS"
cmake --install build/llama-server-cpu

# CUDA backend for the GB10 (compute capability 12.1).
cmake -S llama/server --preset llama_cuda_v13_user_arch -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build/llama-server-cuda_v13 -j "$JOBS"
cmake --install build/llama-server-cuda_v13

CGO_ENABLED=1 go build -trimpath -buildmode=pie \
    -ldflags "-w -s -X github.com/ollama/ollama/version.Version=$VERSION" \
    -o dist/bin/ollama .

ctx=build/cluster-image
rm -rf "$ctx"
mkdir -p "$ctx/bin" "$ctx/lib"
cp dist/bin/ollama "$ctx/bin/"
cp -a dist/lib/ollama "$ctx/lib/"
[ -x "$ctx/lib/ollama/ggml-rpc-server" ] || { echo "ggml-rpc-server missing from the build" >&2; exit 1; }

docker build -f Dockerfile.cluster -t "$TAG" "$ctx"
echo "built $TAG"
