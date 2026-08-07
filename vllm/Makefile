IMAGE_NAME     ?= my-vllm
VLLM_VERSION   ?= latest
CONTAINER_NAME ?= vllm-server
HOST_PORT      ?= 8000
VLLM_MODEL     ?=
HF_TOKEN       ?=
HF_CACHE_DIR   ?= $(HOME)/.cache/huggingface
# Shared Docker network — must match 'name:' in docker-compose.yml networks section
PLATFORM_NETWORK ?= vllm_platform
# IP of the GPU host — required for monitoring stack scrape targets
GPU_HOST_IP      ?=

.PHONY: build run stop restart logs verify clean preflight \
        gpu-stack-up gpu-stack-up-gateway gpu-stack-down gpu-stack-logs \
        monitoring-up monitoring-down monitoring-logs status

## Validate the host before deploying anything
preflight:
	bash scripts/preflight.sh

## Validate the host and attempt automatic fixes (NVIDIA runtime install)
preflight-fix:
	bash scripts/preflight.sh --fix

build:
	docker build --build-arg VLLM_VERSION=$(VLLM_VERSION) -t $(IMAGE_NAME) .

## Start the vLLM inference engine container
## Usage: make run VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct HF_TOKEN=hf_xxx
run:
	docker run -d \
	    --name $(CONTAINER_NAME) \
	    --gpus all \
	    --restart unless-stopped \
	    --network $(PLATFORM_NETWORK) \
	    -p $(HOST_PORT):8000 \
	    -v $(HF_CACHE_DIR):/root/.cache/huggingface \
	    -e VLLM_MODEL=$(VLLM_MODEL) \
	    -e HF_TOKEN=$(HF_TOKEN) \
	    $(IMAGE_NAME)

stop:
	docker stop $(CONTAINER_NAME) && docker rm $(CONTAINER_NAME)

restart: stop run

logs:
	docker logs -f $(CONTAINER_NAME)

verify:
	VLLM_VERIFY_HOST=localhost VLLM_PORT=$(HOST_PORT) bash scripts/verify.sh

clean: stop
	docker rmi $(IMAGE_NAME)

## ── GPU HOST STACK (nginx + dcgm-exporter + node-exporter) ──────────────────

## Start the GPU host stack (nginx, DCGM exporter, node exporter)
gpu-stack-up:
	docker compose up -d

## Start the GPU host stack including the optional FastAPI gateway
gpu-stack-up-gateway:
	docker compose --profile gateway up -d

## Stop the GPU host stack
gpu-stack-down:
	docker compose down

## Tail GPU host stack logs
gpu-stack-logs:
	docker compose logs -f

## ── MONITORING STACK (Prometheus + Grafana — runs on separate host) ──────────

## Start the monitoring stack (Prometheus + Grafana + Alertmanager)
## Requires GPU_HOST_IP to be set: make monitoring-up GPU_HOST_IP=54.x.x.x
monitoring-up:
	GPU_HOST_IP=$(GPU_HOST_IP) docker compose -f docker-compose.monitoring.yml up -d

## Stop the monitoring stack
monitoring-down:
	docker compose -f docker-compose.monitoring.yml down

## Tail monitoring stack logs
monitoring-logs:
	docker compose -f docker-compose.monitoring.yml logs -f

## Show status of all platform containers (vLLM + GPU stack + monitoring stack)
status:
	@echo "=== vLLM Container ==="
	@docker ps --filter name=$(CONTAINER_NAME) --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || true
	@echo ""
	@echo "=== GPU Host Stack ==="
	@docker compose ps 2>/dev/null || true
	@echo ""
	@echo "=== Monitoring Stack ==="
	@docker compose -f docker-compose.monitoring.yml ps 2>/dev/null || true
