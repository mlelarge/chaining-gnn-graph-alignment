# Reproduction driver — one target per paper table.
# Each target downloads the relevant pretrained release and runs inference
# (no GPU needed). Override the interpreter with e.g. `make PYTHON='uv run python' ...`.
#
#   make env            # uv sync
#   make verify         # quick end-to-end smoke test
#   make synthetic      # Table tab:ER-Reg  (sparse / dense / regular)
#   make realworld      # Table tab:realworld-noisy + tab:multimagna-full
#
# Prepared parquets land in $(DATA) (gitignored); raw edge lists stay in data/raw/.

PYTHON ?= python
DATA   ?= ./data/prepared
SEED         ?= 0
NUM_EXAMPLES ?= 100
OUT          ?= repro_results.jsonl

.PHONY: help env data verify \
        synthetic synthetic-sparse synthetic-dense synthetic-regular \
        realworld ca-netscience euroroad yeast25lc multimagna \
        train-sparse reproduce clean

help:
	@echo "Targets:"
	@echo "  env                 uv sync (create the environment)"
	@echo "  verify              quick end-to-end smoke: pytest + a real-world round-trip"
	@echo "  data                build all real-world parquets into $(DATA)"
	@echo "  synthetic           tab:ER-Reg  — sparse + dense + regular"
	@echo "  synthetic-sparse    sparse Erdos-Renyi (d=4)"
	@echo "  synthetic-dense     dense Erdos-Renyi (d=80)"
	@echo "  synthetic-regular   regular graphs (d=10)"
	@echo "  realworld           tab:realworld-noisy + tab:multimagna-full"
	@echo "  ca-netscience       ca-netscience @ noise 0.1 and 0.2"
	@echo "  euroroad            inf-euroroad @ noise 0.1 and 0.2"
	@echo "  yeast25lc           yeast25LC @ noise 0.05 and 0.1"
	@echo "  multimagna          MultiMAGNA yeast (held-out 20% / 25%)"
	@echo "  train-sparse        train ChFGNN from scratch on sparse ER (needs a GPU)"
	@echo "  reproduce           seeded full-grid -> JSON (OUT=$(OUT), SEED=$(SEED), NUM_EXAMPLES=$(NUM_EXAMPLES))"
	@echo "  clean               remove prepared data, checkpoints and outputs"

env:
	uv sync

# Quick end-to-end smoke test: tests, then prepare a tiny dataset and reproduce
# one real-world cell from a published release.
verify:
	$(PYTHON) -m pytest tests/ -q
	$(PYTHON) -m repro.prepare_data --dataset ca-netscience --n-train 2 --n-val 4 --output-dir $(DATA)
	$(PYTHON) run_inference_real.py --release v1.1.0-canetscience-pn0.1 --data-dir $(DATA) --num-examples 4 --N-max 15

data:
	$(PYTHON) -m repro.prepare_data --all --output-dir $(DATA)

# ---- Synthetic graphs — Table tab:ER-Reg (FAQ baselines + ChFGNN) -------------
synthetic: synthetic-sparse synthetic-dense synthetic-regular

synthetic-sparse:
	$(PYTHON) run_inference.py --release v1.0.0-er500-d4-pn0.22
	$(PYTHON) run_baseline.py  --release v1.0.0-er500-d4-pn0.22

synthetic-dense:
	$(PYTHON) run_inference.py --release v1.0.0-er500-d80-pn0.24
	$(PYTHON) run_baseline.py  --release v1.0.0-er500-d80-pn0.24

synthetic-regular:
	$(PYTHON) run_inference.py --release v1.0.0-reg500-d10-pn0.11
	$(PYTHON) run_baseline.py  --release v1.0.0-reg500-d10-pn0.11

# ---- Real-world graphs — Tables tab:realworld-noisy / tab:multimagna-full -----
realworld: ca-netscience euroroad yeast25lc multimagna

ca-netscience:
	$(PYTHON) -m repro.prepare_data --dataset ca-netscience --output-dir $(DATA)
	$(PYTHON) run_inference_real.py --release v1.1.0-canetscience-pn0.1 --data-dir $(DATA)
	$(PYTHON) run_inference_real.py --release v1.1.0-canetscience-pn0.2 --data-dir $(DATA)

euroroad:
	$(PYTHON) -m repro.prepare_data --dataset inf-euroroad --output-dir $(DATA)
	$(PYTHON) run_inference_real.py --release v1.1.0-euroroad-pn0.1 --data-dir $(DATA)
	$(PYTHON) run_inference_real.py --release v1.1.0-euroroad-pn0.2 --data-dir $(DATA)

yeast25lc:
	$(PYTHON) -m repro.prepare_data --dataset multimagna-noisy --output-dir $(DATA)
	$(PYTHON) run_inference_real.py --release v1.1.0-yeast25lc-pn0.05 --data-dir $(DATA)
	$(PYTHON) run_inference_real.py --release v1.1.0-yeast25lc-pn0.1  --data-dir $(DATA)

multimagna:
	$(PYTHON) -m repro.prepare_data --dataset multimagna-full --output-dir $(DATA)
	$(PYTHON) run_inference_real.py --release v1.1.0-multimagna --data-dir $(DATA) --test-name multimagna_yeast20_test
	$(PYTHON) run_inference_real.py --release v1.1.0-multimagna --data-dir $(DATA) --test-name multimagna_yeast25_test

# ---- Training from scratch (needs a GPU; no released checkpoints) -------------
train-sparse:
	$(PYTHON) commander.py dataset=sparse

# Seeded reproduction of every results table -> JSONL (for the cluster). Split the
# heavy dense cell off with e.g.: make reproduce ARGS, or call the module directly
# with --family dense / --real. See `python -m repro.reproduce_results -h`.
reproduce:
	$(PYTHON) -m repro.reproduce_results --all --seed $(SEED) --num-examples $(NUM_EXAMPLES) --out $(OUT) --data-dir $(DATA)

clean:
	rm -rf $(DATA) ./checkpoints ./outputs
