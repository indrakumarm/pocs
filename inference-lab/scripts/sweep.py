"""Automated experiment runner: for each config, launch vLLM in Docker via the
Python SDK, wait until healthy, run bench.py, save logs, tear down.

  python scripts/sweep.py weeks/week3/config.yaml
"""
import subprocess, sys, time, os, yaml, docker, httpx

client = docker.from_env()


ENGINES = {
    "vllm": {"image": "vllm/vllm-openai:latest", "port": 8000,
             "cmd": lambda m: ["--model", m, "--host", "0.0.0.0", "--port", "8000"]},
    "sglang": {"image": "lmsysorg/sglang:latest", "port": 30000,
               "cmd": lambda m: ["python3", "-m", "sglang.launch_server", "--model-path", m,
                                 "--host", "0.0.0.0", "--port", "30000"]},
}


def launch(name, engine, image, model, args, gpus, port, env):
    try:
        client.containers.get(name).remove(force=True)
    except docker.errors.NotFound:
        pass
    ids = [str(g) for g in gpus] if gpus else None
    e = ENGINES[engine]
    return client.containers.run(
        image or e["image"], command=[*e["cmd"](model), *map(str, args)],
        name=name, detach=True, ipc_mode="host",
        ports={f"{e['port']}/tcp": port},
        device_requests=[docker.types.DeviceRequest(
            device_ids=ids, count=None if ids else -1, capabilities=[["gpu"]])],
        environment={"HF_TOKEN": os.getenv("HF_TOKEN", ""), **env},
        volumes={os.path.expanduser("~/.cache/huggingface"):
                 {"bind": "/root/.cache/huggingface", "mode": "rw"}},
    )


def wait_ready(c, port, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        c.reload()
        if c.status != "running":
            raise RuntimeError("container died:\n" + c.logs(tail=50).decode())
        try:
            if httpx.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise TimeoutError("server not ready")


def main(path):
    cfg = yaml.safe_load(open(path))
    base = cfg["base"]
    os.makedirs("results/logs", exist_ok=True)
    for exp in cfg["experiments"]:
        name = exp["tag"]
        print(f"=== {name} ===")
        # per-experiment keys (engine, image, model, args, gpus, env) override base
        m = {**base, **exp}
        port = m.get("port", 8000)
        c = launch("lab-" + name, m.get("engine", "vllm"), m.get("image"), m["model"],
                   m.get("args", []), m.get("gpus"), port, m.get("env", {}))
        try:
            wait_ready(c, port)
            subprocess.run([
                sys.executable, "scripts/bench.py", "--model", m["model"],
                "--tag", name, "--base-url", f"http://localhost:{port}/v1",
                "--out", cfg.get("out", "results/bench.csv"),
                *map(str, cfg.get("bench_args", [])), *map(str, exp.get("bench_args", [])),
            ], check=True)
        except (RuntimeError, TimeoutError, subprocess.CalledProcessError) as e:
            print(f"!!! {name} failed, skipping: {str(e).splitlines()[0]}")
        finally:
            open(f"results/logs/{name}.log", "wb").write(c.logs())
            c.remove(force=True)


if __name__ == "__main__":
    main(sys.argv[1])
