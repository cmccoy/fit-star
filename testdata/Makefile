BASE := gtr_sim
ALL := $(addprefix gtr_sim., pb.gz fit.json.gz fit-unconditioned.json.gz)

JQP := ".partitions[0].Q"

display: $(ALL)
	@echo "conditioned"
	zcat gtr_sim.fit.json.gz | jq $(JQP)
	@echo "unconditioned"
	zcat gtr_sim.fit-unconditioned.json.gz | jq $(JQP)

all: $(ALL)

%.bam: %.fasta
	../../simulate/combine.py -o $@ $<

%.pb.gz: %.bam %.fasta %.fasta.fai
	../build/src/build-mutation-matrices -f $*.fasta -i $*.bam -o $@

%.fit.json.gz: %.pb.gz
	OMP_NUM_THREADS=6 ../build/src/fit-star -i $< -o $@ --model GTR --add-rates --max-rounds 200 --rate-dist gamma

%.fit-unconditioned.json.gz: %.pb.gz
	OMP_NUM_THREADS=6 ../build/src/fit-star -i $< -o $@ --model GTR --add-rates --max-rounds 200 --rate-dist gamma --no-fix-root-frequencies

gtr_sim.fasta: simulate_mixture.xml
	beast -beagle -seed 1 $<

%.fasta.fai: %.fasta
	samtools faidx $<

.PHONY: all display
