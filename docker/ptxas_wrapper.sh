#!/bin/bash
# ptxas wrapper installed in the GPU image when the host GPU is newer than Hopper (see docker/run.sh).
# XLA in jax 0.4.30 predates Blackwell (sm_100+): it emits PTX with ".target sm_90a" while asking ptxas to compile
# for the real GPU (e.g. -arch=sm_120), which fails because arch-specific (a) PTX is not forward compatible.
# Plain sm_90 PTX is forward compatible, so the target is downgraded. If the PTX used Hopper-only instructions
# ptxas would fail loudly rather than produce wrong code. On GPUs up to Hopper (sm_90) this wrapper does nothing.
REAL_PTXAS="$(dirname "$(readlink -f "$0")")/ptxas.real"
arch=""
prev=""
for a in "$@"; do
    if [[ "$a" =~ ^-arch=sm_([0-9]+) ]]; then arch="${BASH_REMATCH[1]}"; fi
    if [[ "$prev" == "-arch" && "$a" =~ ^sm_([0-9]+) ]]; then arch="${BASH_REMATCH[1]}"; fi
    prev="$a"
done
if [[ -n "$arch" && "$arch" -gt 90 ]]; then
    prev=""
    for a in "$@"; do
        if [[ -f "$a" && "$prev" != "-o" ]]; then sed -i 's/^\.target sm_90a/.target sm_90/' "$a"; fi
        prev="$a"
    done
fi
exec "$REAL_PTXAS" "$@"
