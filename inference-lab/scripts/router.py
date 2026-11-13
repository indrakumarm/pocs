"""Streaming router for independent, identical vLLM replicas (not tensor parallelism)."""
import argparse
import asyncio
from dataclasses import asdict, dataclass
import logging
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector, web


LOG = logging.getLogger(__name__)
POLICIES = ("round-robin", "least-inflight")


@dataclass
class Replica:
    """Backend health and counters local to this router, not vLLM's internal queue."""

    name: str
    url: str
    healthy: bool = False
    inflight: int = 0
    peak_inflight: int = 0
    requests: int = 0
    completed: int = 0
    failures: int = 0
    cancelled: int = 0
    last_error: str = ""

    def __post_init__(self):
        self.url = self.url.rstrip("/")
        parts = urlsplit(self.url)
        if (parts.scheme not in ("http", "https") or not parts.hostname
                or parts.path or parts.query or parts.fragment or parts.username):
            raise ValueError(f"{self.name}: expected an HTTP(S) server root URL without credentials")


class Router:
    def __init__(self, replicas, policy="round-robin", health_interval=2.0):
        if not replicas or len({r.name for r in replicas}) != len(replicas):
            raise ValueError("Provide replicas with unique names")
        if len({r.url for r in replicas}) != len(replicas):
            raise ValueError("Replica URLs must be distinct")
        if policy not in POLICIES or health_interval <= 0:
            raise ValueError("Invalid routing policy or health interval")
        self.replicas = replicas
        self.policy = policy
        self.health_interval = health_interval
        self.cursor = 0
        self.session = None

    async def probe(self, replica):
        try:
            async with self.session.get(replica.url + "/health",
                                        timeout=ClientTimeout(total=2)) as response:
                response.raise_for_status()
            replica.healthy = True
        except (ClientError, asyncio.TimeoutError) as error:
            if replica.healthy or not replica.last_error:
                LOG.warning("%s health check failed: %s", replica.name, error)
            replica.healthy = False
            replica.last_error = str(error) or type(error).__name__

    async def check_health(self):
        await asyncio.gather(*(self.probe(r) for r in self.replicas))

    async def health_loop(self):
        while True:
            await asyncio.sleep(self.health_interval)
            await self.check_health()

    async def lifecycle(self, app):
        # Reuse backend connections; avoid a connector queue becoming the load bottleneck.
        async with ClientSession(
            connector=TCPConnector(limit=0),
            timeout=ClientTimeout(total=None, sock_connect=3, sock_read=600),
        ) as self.session:
            await self.check_health()
            task = asyncio.create_task(self.health_loop())
            try:
                # aiohttp serves requests while this cleanup context is suspended at yield.
                yield
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def acquire(self):
        """Select a healthy backend and reserve it for the full inference response."""
        candidates = [i for i, r in enumerate(self.replicas) if r.healthy]
        if not candidates:
            raise web.HTTPServiceUnavailable(text="No healthy vLLM replicas\n")
        if self.policy == "least-inflight":
            minimum = min(self.replicas[i].inflight for i in candidates)
            candidates = [i for i in candidates if self.replicas[i].inflight == minimum]
        # Circular distance implements round-robin, including least-inflight ties
        # and skipping unhealthy replicas without resetting the cursor.
        index = min(candidates, key=lambda i: (i - self.cursor) % len(self.replicas))
        self.cursor = (index + 1) % len(self.replicas)
        replica = self.replicas[index]
        # Selection and reservation have no await: concurrent handlers cannot overbook a tie.
        replica.inflight += 1
        replica.peak_inflight = max(replica.peak_inflight, replica.inflight)
        replica.requests += 1
        return replica

    def snapshot(self):
        return {"policy": self.policy, "replicas": [asdict(r) for r in self.replicas]}

    async def stats(self, request):
        return web.json_response(self.snapshot())

    async def health(self, request):
        healthy = any(r.healthy for r in self.replicas)
        return web.json_response({"healthy": healthy}, status=200 if healthy else 503)

    async def proxy(self, request):
        inference = request.path in ("/v1/chat/completions", "/v1/completions")
        body = await request.read()
        if inference:
            replica = self.acquire()
        else:
            # Model discovery and token calibration must not affect inference routing counters.
            replica = next((r for r in self.replicas if r.healthy), None)
            if replica is None:
                raise web.HTTPServiceUnavailable(text="No healthy vLLM replicas\n")
        headers = {key: request.headers[key] for key in ("Content-Type", "Authorization", "Accept")
                   if key in request.headers}
        response = None
        try:
            async with self.session.request(
                request.method, replica.url + request.path, params=request.query,
                data=body, headers=headers,
            ) as upstream:
                response = web.StreamResponse(status=upstream.status, headers={
                    "Content-Type": upstream.headers.get("Content-Type", "application/octet-stream"),
                    "X-Lab-Replica": replica.name,
                })
                await response.prepare(request)
                # Forward bytes as they arrive, without buffering the generation or parsing SSE.
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                if inference:
                    if upstream.status < 400:
                        replica.completed += 1
                    else:
                        replica.failures += 1
                        replica.last_error = f"HTTP {upstream.status}"
                        LOG.warning("%s inference failed: %s", replica.name, replica.last_error)
                if upstream.status in (502, 503, 504):
                    replica.healthy = False
                return response
        except asyncio.CancelledError:
            if inference:
                replica.cancelled += 1
            raise
        except (ClientError, asyncio.TimeoutError, ConnectionError) as error:
            replica.last_error = str(error) or type(error).__name__
            LOG.warning("%s proxy failed: %s", replica.name, replica.last_error)
            if inference:
                replica.failures += 1
            replica.healthy = False
            if response is not None and response.prepared:
                # Never turn a truncated SSE stream into a successful response or replay it.
                if request.transport is not None:
                    request.transport.close()
                return response
            raise web.HTTPBadGateway(text=f"Replica {replica.name} connection failed\n") from error
        finally:
            if inference:
                # Headers/first token do not end the reservation; completion, error or
                # cancellation does. This keeps streaming requests visible to least-inflight.
                replica.inflight -= 1

    def application(self):
        # Cancel a disconnected client's handler so its backend request and reservation close.
        app = web.Application(client_max_size=8 * 1024 * 1024,
                              handler_args={"handler_cancellation": True})
        app.cleanup_ctx.append(self.lifecycle)
        app.router.add_get("/health", self.health)
        app.router.add_get("/router/stats", self.stats)
        app.router.add_get("/v1/models", self.proxy)
        for path in ("/v1/chat/completions", "/v1/completions", "/tokenize"):
            app.router.add_post(path, self.proxy)
        return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", required=True, help="Server root URLs, no /v1")
    parser.add_argument("--policy", choices=POLICIES, default="round-robin")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    router = Router([Replica(f"replica-{i + 1}", url) for i, url in enumerate(args.backends)],
                    args.policy)
    web.run_app(router.application(), host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
