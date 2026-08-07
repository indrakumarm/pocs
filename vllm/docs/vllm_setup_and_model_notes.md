# vLLM Setup, PyTorch/CUDA, and Model Evaluation Notes

Date: 2026-07-06

## 1. Environment

Target machine:

- AWS EC2 GPU instance
- GPU: Tesla T4, 15 GB VRAM
- OS: Amazon Linux 2023
- NVIDIA driver verified with `nvidia-smi`

Example `nvidia-smi` output:

```text
Driver Version: 595.71.05
CUDA Version: 13.2
GPU: Tesla T4
Memory: 15360 MiB
```

Important interpretation:

- `nvidia-smi` showing `CUDA Version: 13.2` means the installed NVIDIA driver can support CUDA runtimes up to that level.
- It does not mean the system CUDA Toolkit must be installed.
- For PyTorch pip wheels, the key requirement is that the NVIDIA driver is new enough for the CUDA runtime bundled inside the PyTorch wheel.

## 2. NVIDIA driver vs CUDA vs PyTorch vs vLLM

The GPU software stack is:

```text
NVIDIA GPU hardware
  -> NVIDIA driver
  -> CUDA runtime/libraries
  -> PyTorch
  -> vLLM
  -> Model, e.g. Qwen
```

Definitions:

- NVIDIA driver: lets Linux communicate with and control the GPU. Verified by `nvidia-smi`.
- CUDA: NVIDIA's GPU computing platform and runtime libraries used by ML frameworks.
- PyTorch: deep learning framework. CUDA-enabled PyTorch wheels include the CUDA runtime they need.
- vLLM: inference server/runtime that loads the model and exposes an OpenAI-compatible API.

Usually, for PyTorch installed via pip, the full system CUDA Toolkit is not required.

## 3. PyTorch installation issue and fix

The original command failed:

```bash
python3.11 -m pip install --user torch --index-url https://pytorch.org
```

Error:

```text
ERROR: Could not find a version that satisfies the requirement torch
```

Cause:

- `https://pytorch.org` is not the correct pip wheel index.

Correct CUDA wheel indexes use this pattern:

```text
https://download.pytorch.org/whl/cuXXX
```

Examples:

```text
cu121 = CUDA 12.1
cu124 = CUDA 12.4
cu126 = CUDA 12.6
```

Working command:

```bash
python3.11 -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
```

Verification:

```bash
python3.11 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Successful output:

```text
2.6.0+cu124
12.4
True
Tesla T4
```

## 4. Root vs user Python environment issue

We saw:

```text
WARNING: Running pip as the 'root' user...
ModuleNotFoundError: No module named 'torch'
```

Cause:

- Packages were installed as root or into root's user site-packages.
- The `ec2-user` Python could not see them.

Recommendation:

- Use a Python virtual environment as the actual runtime user.
- Use `sudo` only for OS packages, not for pip installs.

Recommended setup:

```bash
sudo dnf install -y python3.11-devel python3.11-pip

python3.11 -m venv ~/vllm-venv
source ~/vllm-venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install vllm
```

## 5. Starting vLLM

Basic command:

```bash
source ~/vllm-venv/bin/activate

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 9090 \
  --dtype half
```

Recommended safer T4 command:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 9090 \
  --dtype half \
  --served-model-name qwen \
  --gpu-memory-utilization 0.90 \
  --max-model-len 15000
```

Meaning:

- `--served-model-name qwen`: lets clients use `"model": "qwen"` instead of the full Hugging Face model ID.
- `--gpu-memory-utilization 0.90`: lets vLLM use up to 90% of GPU memory.
- `--max-model-len 15000`: maximum context length in tokens.
- `--dtype half`: use FP16/float16. Good for Tesla T4.

Context length:

```text
input prompt tokens + output tokens <= max-model-len
```

Rough estimate:

```text
1 token ~= 3-4 English characters
```

## 6. Testing vLLM

Health:

```bash
curl http://localhost:9090/health
```

Loaded models:

```bash
curl http://localhost:9090/v1/models
```

Chat completion:

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [
      {"role": "user", "content": "Say hello and confirm vLLM is working."}
    ],
    "max_tokens": 100,
    "temperature": 0.0
  }'
```

A standalone test client was created at:

```text
/Users/indrakumar.m/Downloads/vllm_test_client.py
```

Example:

```bash
python3 vllm_test_client.py \
  --base-url http://localhost:9090/v1 \
  --model qwen \
  --prompt "Say hello and confirm vLLM is working."
```

## 7. vLLM model download behavior

When starting vLLM with a Hugging Face model ID, vLLM downloads the model automatically on first run.

Cache location is usually:

```text
~/.cache/huggingface/hub
```

If the model is private or gated:

```bash
huggingface-cli login
```

or:

```bash
export HF_TOKEN=...
```

## 8. vLLM as provider in the Airflow failure analyzer

The project has an LLM factory at:

```text
/Users/indrakumar.m/Documents/git_ws/ai-assisted-ops/poc/dag-failure-analyzer/llm/llm_factory.py
```

Best approach:

- Add `vllm` as a provider.
- Internally use `langchain_openai.ChatOpenAI`.
- Point `base_url` to vLLM's OpenAI-compatible endpoint.

Conceptual flow:

```text
Airflow analyzer
  -> LLMFactory
  -> LangChain ChatOpenAI
  -> http://vllm-host:9090/v1/chat/completions
  -> vLLM
  -> PyTorch/CUDA/GPU
```

The analyzer app does not need CUDA/PyTorch/vLLM unless it hosts the model locally. Only the vLLM server machine needs those.

Recommended env config:

```env
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://<gpu-ec2-ip>:9090/v1
VLLM_MODEL=qwen
VLLM_API_KEY=EMPTY
VLLM_TIMEOUT=300
VLLM_TEMPERATURE=0.0
VLLM_MAX_TOKENS=1024
```

Recommended factory branch:

```python
elif provider == 'vllm':
    from langchain_openai import ChatOpenAI

    vllm_config = config.get('vllm', {})
    base_url = vllm_config.get('base_url', 'http://localhost:9090/v1')

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=vllm_config.get('api_key', 'EMPTY'),
        timeout=vllm_config.get('timeout', 300),
        max_tokens=vllm_config.get('max_tokens', 1024),
    )

    logger.info(f"Created vLLM client: {model} at {base_url}")
    return LLMClient(llm, provider='vllm', model=model)
```

## 9. Chat vs invoke vs converse

The project uses LangChain `invoke()`.

Relevant code:

```text
LLMClient.invoke(prompt, schema)
LLMClient.invoke_raw(prompt)
```

For the Airflow analyzer:

```text
plugins/base_analyzer.py
  -> get_llm_client()
  -> client.invoke(prompt, schema)
```

Meaning:

- `invoke()` is LangChain's provider-neutral call method.
- `chat` usually refers to OpenAI-style chat messages.
- `converse()` is Bedrock-specific and should not be used in analyzer code if portability is desired.

Recommendation:

- Keep analyzer code using `client.invoke()`.
- Hide provider-specific details inside `llm_factory.py`.

## 10. Token usage

The project already tracks token usage from LangChain response metadata:

```text
response.usage_metadata
response.response_metadata['usage' or 'token_usage']
```

For vLLM/OpenAI-compatible responses, usage usually includes:

```text
prompt_tokens
completion_tokens
total_tokens
```

Set max output tokens explicitly for analyzer quality:

```env
VLLM_MAX_TOKENS=1024
```

Increase if responses are too short:

```env
VLLM_MAX_TOKENS=1536
```

## 11. Replay script hang

Command that appeared stuck:

```bash
python scripts/replay_failures_from_logs.py \
  --log-root /efs/airflow-venv/tvd-prod/logs/ \
  --match "*dag_id=.../attempt=1.log*" \
  --limit 1 \
  --failed-only
```

Root cause:

```python
all_files = sorted(
    _iter_log_files(log_root),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
```

The script scans and stats every `*.log` recursively under EFS before applying `--match`.

Evidence from traceback:

```text
for path in log_root.rglob("*.log")
KeyboardInterrupt
```

Workaround:

Use a narrow `--log-root` close to the target file:

```bash
python -u scripts/replay_failures_from_logs.py \
  --log-root "/efs/airflow-venv/tvd-prod/logs/dag_id=.../run_id=.../task_id=check_job" \
  --match "*attempt=1.log" \
  --limit 1 \
  --failed-only
```

Long-term improvement:

- Apply `--match` during walking.
- Avoid sorting/stat-ing the entire EFS logs tree when `--limit` is small or exact match is provided.

## 12. Quantization

Quantization is model compression by storing model weights with fewer bits.

Example:

```text
FP16: 16 bits per weight
INT8:  8 bits per weight
INT4:  4 bits per weight
```

Analogy:

```text
High-resolution photo -> best quality, large file
Compressed JPEG       -> smaller file, tiny quality loss

FP16 model            -> larger, more precise
AWQ/GPTQ model        -> smaller, approximate, usually still good
```

For a 7B model:

```text
FP16: roughly 14 GB just for weights
AWQ/INT4: roughly 4-5 GB for weights
```

This matters on Tesla T4 15 GB because memory is also needed for:

- KV cache
- CUDA runtime
- vLLM worker overhead
- temporary compute buffers

For T4, prefer AWQ/GPTQ quantized 7B/8B models.

## 13. Model experiments and findings

Models discussed/tried:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct-AWQ`
- `Qwen/Qwen3-8B-AWQ`
- Candidate: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Candidate: `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`
- Candidate: `mistralai/Mistral-7B-Instruct-v0.3`
- Candidate: `meta-llama/Llama-3.1-8B-Instruct`
- Candidate: Gemma 3 4B class models

Key findings:

- `Qwen2.5-1.5B-Instruct` works but gives generic remediation.
- `Qwen2.5-7B-Instruct` in FP16 is tight/OOM-prone on T4.
- `Qwen2.5-7B-Instruct-AWQ` fits better and improves quality.
- `Qwen3-8B-AWQ` looked better than Qwen2.5-7B-AWQ on the tested Airflow failures.
- Claude Sonnet via Bedrock still produced more polished and specific remediation.

## 14. OOM during model startup

Error example:

```text
CUDA out of memory. Tried to allocate 1.02 GiB.
GPU has 14.56 GiB total, only 111 MiB free.
```

If vLLM is stopped, previous model memory should be released.

Check:

```bash
nvidia-smi
```

If memory is still used, kill the specific PID shown by `nvidia-smi`:

```bash
kill <PID>
kill -9 <PID>
```

If memory is 0 MiB and startup still OOMs, the new model/config is too large.

Safer startup for FP16 7B:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 9090 \
  --dtype half \
  --served-model-name qwen \
  --gpu-memory-utilization 0.75 \
  --max-model-len 4096 \
  --enforce-eager
```

Better T4 path:

```bash
vllm serve Qwen/Qwen3-8B-AWQ \
  --host 0.0.0.0 \
  --port 9090 \
  --served-model-name qwen \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
```

Increase context gradually:

```text
8192 -> 12000 -> 15000
```

## 15. Hugging Face model lookup

vLLM does not provide a universal list of all possible models.

Use Hugging Face:

```bash
hf models ls --search "Qwen3-8B-AWQ"
```

Example result:

```text
Qwen/Qwen3-8B-AWQ
```

This is the model ID to pass to vLLM:

```bash
vllm serve Qwen/Qwen3-8B-AWQ ...
```

Mistake seen:

```text
Qwen/Qwen/Qwen3-8B-AWQ
```

Correct:

```text
Qwen/Qwen3-8B-AWQ
```

## 16. Prompt tuning lessons

Changing model alone helped, but prompt tuning helped significantly.

For EMR failures, the prompt should force the model to use:

- exact exception line
- exit code
- Spark driver/executor memory values
- record counts/input size
- cluster ID
- step ID
- application ID
- S3 path/config key if present

Useful rule:

```text
Do not give generic Spark tuning advice if concrete config values are present.
Use before/after values when available, e.g. "--driver-memory 4g -> 8g".
```

ID accuracy rule:

```text
IDs starting with s- are EMR Step IDs.
IDs starting with application_ are YARN/Spark Application IDs.
Do not call an EMR Step ID a Spark Application ID.
```

For GenericAnalyzer Python failures, prompt should force:

- exact exception line
- file/function/line number
- actual task_id, not run_id
- innermost exception, not wrapper exception
- safe validation fixes instead of blindly defaulting values

Useful rule:

```text
Do not confuse DAG ID, task ID, and run ID.
run_id/execution_date is execution metadata, not a task/component.
```

## 17. Response quality comparison

### EMR OOM failure

Claude Sonnet:

- Identified Spark driver OOM.
- Cited `--driver-memory 4g`.
- Cited 1.6M records.
- Suggested `4g -> 8g/12g`, executor memory, dynamic allocation, GC tuning.

Earlier Qwen:

- Correct error but generic.
- Missed concrete values.

After prompt changes and stronger model:

- Qwen picked up `Java heap space`, exit code `137`, `--driver-memory 4g`, 1.6M records, cluster ID, step ID.
- Still produced one speculative Spark config suggestion.

Recommendation:

```text
For fix_suggestions, prefer standard, directly supported remediations first.
Do not suggest advanced Spark configs unless the logs show evidence for that mechanism.
Mark speculative tuning as optional investigation, not primary fix.
```

### Python/Airflow TypeError failure

Failure:

```text
TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'
File: adhoc_channel_check_dag.py
Function: validateData
Line: 112
```

Qwen:

- Correctly identified TypeError and line.
- Confused run_id as task in affected components.

Sonnet:

- More complete remediation.
- Suggested null checks, specific line fix, debugging upstream variables.

Recommendation:

```text
GenericAnalyzer prompt should explicitly say:
Never list run_id as task_id.
Use Context Task as the task.
For Python tracebacks, identify exact file/function/line.
```

### Missing data AirflowFailException

`Qwen3-8B-AWQ` was better than `Qwen2.5-7B-Instruct-AWQ`.

It produced:

- data missing for `2026-06-28`
- function `repair_and_check_data`
- file `ATMS_load_opensearch.py`
- line 78
- better fix suggestions around data availability and date calculation

Remaining issue:

- Still included `run_id` as affected component.

## 18. Practical recommendation

For current T4 setup:

```text
Best local model candidate so far: Qwen/Qwen3-8B-AWQ
Good alternate: Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
Best quality overall: Claude Sonnet via Bedrock
```

For production:

- Use Sonnet/Bedrock if quality is more important than cost/local hosting.
- Use Qwen3-8B-AWQ if local hosting/cost control is important.
- Keep improving prompts and schema to reduce generic/speculative answers.

Recommended vLLM defaults:

```bash
vllm serve Qwen/Qwen3-8B-AWQ \
  --host 0.0.0.0 \
  --port 9090 \
  --served-model-name qwen \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
```

Recommended app env:

```env
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://<gpu-ec2-ip>:9090/v1
VLLM_MODEL=qwen
VLLM_API_KEY=EMPTY
VLLM_TIMEOUT=300
VLLM_TEMPERATURE=0.0
VLLM_MAX_TOKENS=1024
```

