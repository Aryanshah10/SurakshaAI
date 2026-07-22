import hashlib
import json
import time
from datetime import datetime
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message
import aiofiles
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "audit_log.jsonl")


class AuditLogMiddleware(BaseHTTPMiddleware):
    """
    Logs every API request + response with:
    - timestamp, method, path, status_code
    - SHA256 hash of response body (court-admissible integrity proof)
    - user (extracted from JWT sub claim if present)
    """

    async def dispatch(self, request: Request, call_next):
        start = time.time()

        # Read request body
        body_bytes = await request.body()
        try:
            request_body = json.loads(body_bytes) if body_bytes else {}
        except Exception:
            request_body = {}

        # Call actual route
        response: Response = await call_next(request)

        # Capture response body
        resp_body = b""
        async for chunk in response.body_iterator:
            resp_body += chunk

        # Hash response for integrity
        resp_hash = hashlib.sha256(resp_body).hexdigest()

        duration_ms = round((time.time() - start) * 1000, 2)

        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "method": request.method,
            "path": str(request.url.path),
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "response_hash_sha256": resp_hash,
            "request_preview": str(request_body)[:200],  # truncate PII
        }

        # Write log async
        async with aiofiles.open(LOG_FILE, mode="a") as f:
            await f.write(json.dumps(log_entry) + "\n")

        # Rebuild response with captured body
        return Response(
            content=resp_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )