#!/bin/sh
set -eu

# Default: complete solutions and benchmark. Pass ex1/ex2/ex3 to test a starter.
# Never run the deliberately racy demonstration as a correctness test.
if [ "$#" -eq 0 ]; then
    set -- sol1 sol2 sol3 bench
fi

for program in "$@"; do
    case "$program" in
        ex1|ex2|ex3|sol1|sol2|sol3)
            sizes='0 1 2 3 31 32 33 63 64 65 255 256 257 511 512 513 1003 1048579 16777216'
            ;;
        bench) sizes='0 1 33 257 1003 1048579' ;;
        *) printf 'Unknown test program: %s\n' "$program" >&2; exit 1 ;;
    esac
    for n in $sizes; do
        printf '\nTEST %s N=%s\n' "$program" "$n"
        "./build/$program" "$n"
    done

    # Require the parser's diagnostic, so a missing GPU cannot masquerade as
    # successful argument rejection. Parsing happens before CUDA initialization.
    for bad in '' -1 +1 abc 1.5 16777217 9999999999999999999999999; do
        if "./build/$program" "$bad" >build/test-argument.log 2>&1; then
            printf 'FAIL: %s accepted invalid N=%s\n' "$program" "$bad" >&2
            exit 1
        fi
        if ! grep -q 'N must' build/test-argument.log; then
            cat build/test-argument.log >&2
            exit 1
        fi
    done
    if "./build/$program" 1 2 >build/test-argument.log 2>&1; then
        printf 'FAIL: accepted extra argument\n' >&2
        exit 1
    fi
    grep -q 'Usage:' build/test-argument.log
done

rm -f build/test-argument.log
printf '\nPASS: all selected correctness checks completed.\n'
