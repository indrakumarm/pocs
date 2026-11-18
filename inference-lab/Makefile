MODEL ?= Qwen/Qwen3-8B

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

up:
	MODEL=$(MODEL) docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f vllm

bench:
	.venv/bin/python scripts/bench.py --model $(MODEL) --tag $(TAG) --concurrency 1 2 4 8 16 32

sweep3:
	.venv/bin/python scripts/sweep.py weeks/week3/config.yaml

sweep4:
	.venv/bin/python scripts/sweep.py weeks/week4/config.yaml

WEEK5_CONFIG ?= weeks/week5/config.yaml
WEEK5_ARGS ?=
sweep5:
	.venv/bin/python scripts/replicas.py $(WEEK5_CONFIG) $(WEEK5_ARGS)

check5:
	.venv/bin/python scripts/replicas.py $(WEEK5_CONFIG) --check

POLICY ?= round-robin
router5:
	@test -n "$(REPLICA_A_URL)" -a -n "$(REPLICA_B_URL)" || (echo "Set REPLICA_A_URL and REPLICA_B_URL"; exit 1)
	.venv/bin/python scripts/router.py --backends "$(REPLICA_A_URL)" "$(REPLICA_B_URL)" --policy $(POLICY)

plot5:
	@test -n "$(RUN)" || (echo "Usage: make plot5 RUN=results/week5/<run-id>"; exit 1)
	.venv/bin/python scripts/plot_replicas.py "$(RUN)"

test5:
	.venv/bin/python -m unittest discover -s tests -v

test-aws:
	.venv/bin/python -m unittest discover -s tests -p test_aws.py -v

WEEK6_CONFIG ?= weeks/week6/config.yaml
WEEK6_ARGS ?=
sweep6:
	.venv/bin/python scripts/topologies.py $(WEEK6_CONFIG) $(WEEK6_ARGS)

plan6:
	.venv/bin/python scripts/topologies.py $(WEEK6_CONFIG) $(WEEK6_ARGS) --plan

plot6:
	@test -n "$(RUN)" || (echo "Usage: make plot6 RUN=results/week6/<run-id>"; exit 1)
	.venv/bin/python scripts/plot_replicas.py "$(RUN)"

test6:
	.venv/bin/python -m unittest discover -s tests -p test_week6.py -v

plot:
	.venv/bin/python scripts/plot.py results/week3.csv

plot4:
	.venv/bin/python scripts/plot.py results/week4.csv

metrics:
	curl -s localhost:8000/metrics | grep -E "^vllm:(num_requests|gpu_cache|kv_cache|prefix_cache|time_to_first)" | head -40

# make estimate GPU=L40S IN=2000 OUT=256 [MODEL=...]
GPU ?= A10G
IN ?= 1024
OUT ?= 256
estimate:
	.venv/bin/python scripts/estimate.py --model $(MODEL) --gpu $(GPU) --input-tokens $(IN) --output-tokens $(OUT)

# make loginfo                -> live container;  make loginfo LOG=results/logs/baseline.log
loginfo:
	$(if $(LOG),.venv/bin/python scripts/loginfo.py $(LOG),docker compose logs vllm 2>&1 | .venv/bin/python scripts/loginfo.py -)

# AWS GPU host (run from laptop): aws-up, aws-sync, aws-ssh, aws-tunnel, aws-pull, aws-status, aws-start, aws-stop, aws-down
INSTANCE ?=
VLLM_PORT ?= 8000
GRAFANA_PORT ?= 3000
PROMETHEUS_PORT ?= 9090
aws-%:
	VLLM_PORT=$(VLLM_PORT) GRAFANA_PORT=$(GRAFANA_PORT) PROMETHEUS_PORT=$(PROMETHEUS_PORT) ./aws/lab.sh $* "$(INSTANCE)"
		