#!/bin/sh

set -e
set -u

make
#build/src/build-mutation-matrices -i testdata/test.bam -o testdata/test.muts.pb.gz -f testdata/ighvdj.fasta
#build/src/fit-star -i testdata/test.muts.pb.gz -o test-gtr.json -m GTR --threshold 1e-5
#build/src/fit-star -i testdata/test.muts.pb.gz -o test-word.json -m WORD1 --threshold 1e-5
build/src/build-mutation-matrices -i testdata/test.bam -o testdata/test.muts2.pb.gz -f testdata/ighvdj.fasta -k 2 --random-frame
build/src/fit-star -i testdata/test.muts2.pb.gz -o test-word2_AA_AG.json -m WORD2 --threshold 1e-5 -p AA_AG
build/src/fit-star -i testdata/test.muts2.pb.gz -o test-word2.json -m WORD2 --threshold 1e-5
