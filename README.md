# Requirements

    libprotobuf-2.4.1-dev
    beagle-2.1
    bpp-core-dev 2.1.0
    bpp-phyl-dev 2.1.0
    bpp-seq-dev 2.1.0
    libz-dev

# Programs

This package fits the GTR model to pairwise alignments in BAM files.

Functionality is separated into two parts:

    * `build-mutation-matrices` transforms each aligned sequence in a BAM file into a 4x4 matrix containing counts of each nucleotide substitution
    * `fit-star` fits the GTR model the output of `build-mutation-matrices`, outputting a JSON document with the results.
