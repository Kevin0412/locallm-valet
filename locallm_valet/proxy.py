"""OpenAI-compatible reverse proxy towards the managed backend.

Transparent forwarding: method, path, query, headers and body go to the backend
verbatim; the response (including SSE streaming) comes back as-is.  Only hop-
by-hop headers are stripped.  The client only ever talks to
``http://manager:8000/v1`` and never learns about the backend's address.
"""

from __future__ import annotations

import logging
from typing import Callable

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from .errors import BackendUnavailable
from .usage import SseUsageScanner

logger = logging.getLogger(__name__)

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

_RequestFinished = Callable[[], None]


def _forward_headers(headers: dict) -> dict:
    """Forward client headers except hop-by-hop ones and ``content-length``.

    ``content-length`` must never be forwarded verbatim: the manager may
    rewrite the request body (e.g. injecting ``stream_options`` for usage
    accounting), and httpx recomputes the correct length from the actual
    bytes.  Forwarding a stale length breaks the upstream HTTP/1.1
    connection (h11: "Too much data for declared Content-Length").
    """

    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() != "content-length"
    }


class Proxy:
    """One shared upstream HTTP client; safe for concurrent requests."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=None)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _send(self, method: str, path_and_query: str, headers: dict, body: bytes,
                    stream: bool, base_url: str | None = None):
        base = (base_url or self.base_url).rstrip("/")
        url = f"{base}/{path_and_query.lstrip('/')}"
        request = self._client.build_request(
            method, url, headers=_forward_headers(headers), content=body
        )
        try:
            return await self._client.send(request, stream=stream)
        except httpx.HTTPError as exc:
            raise BackendUnavailable(f"backend unreachable at {base}: {exc}") from exc

    @staticmethod
    def _response_headers(upstream_headers) -> dict:
        return {
            k: v
            for k, v in upstream_headers.items()
            if k.lower() not in _HOP_BY_HOP and k.lower() != "content-length"
        }

    async def plain(self, method: str, path_and_query: str, headers: dict, body: bytes,
                    base_url: str | None = None) -> Response:
        """Non-streaming proxy: buffer the upstream response and return it."""

        upstream = await self._send(method, path_and_query, headers, body, stream=False,
                                    base_url=base_url)
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=self._response_headers(upstream.headers),
        )

    async def stream(
        self,
        method: str,
        path_and_query: str,
        headers: dict,
        body: bytes,
        on_finished: _RequestFinished,
        on_usage: Callable[[dict | None, int], None] | None = None,
        base_url: str | None = None,
    ) -> StreamingResponse:
        """Streaming proxy.

        The upstream connection is opened eagerly (so the response status and
        headers are known), then handed to a generator that pumps the body.
        ``on_finished`` runs only when the stream has been fully consumed *or*
        closed by the client — this is what makes active-request accounting
        correct for ``stream=true``.  ``on_usage`` (if given) is invoked once
        the stream ends with the last SSE ``usage`` frame (or None) and the
        upstream status code.
        """

        upstream = None
        try:
            upstream = await self._send(method, path_and_query, headers, body, stream=True,
                                        base_url=base_url)
        except Exception:
            on_finished()
            raise

        headers_out = self._response_headers(upstream.headers)
        status_code = upstream.status_code
        scanner = SseUsageScanner() if on_usage is not None else None

        async def gen():
            try:
                async for chunk in upstream.aiter_raw():
                    if scanner is not None:
                        scanner.feed(chunk)
                    yield chunk
            finally:
                if scanner is not None:
                    try:
                        on_usage(scanner.finish(), status_code)
                    except Exception:  # noqa: BLE001 - recording must not break streaming
                        logger.exception("usage capture callback failed")
                await upstream.aclose()
                on_finished()

        return StreamingResponse(
            gen(),
            status_code=status_code,
            headers=headers_out,
            media_type=headers_out.get("content-type"),
        )
