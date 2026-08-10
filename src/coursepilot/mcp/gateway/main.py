"""CoursePilot MCP Gateway 启动入口（薄壳）。

职责仅剩 uvicorn 启动；应用构建（create_app）与访问日志已分别拆分至
``gateway/app.py`` 与 ``gateway/observability.py``（P1-T3 拆分）。

启动：
    PYTHONPATH=src uv run python -m coursepilot.mcp.gateway.main --port 8080
    # 生产 HTTPS（自签或正式证书）
    PYTHONPATH=src uv run python -m coursepilot.mcp.gateway.main \\
        --ssl-certfile cert.pem --ssl-keyfile key.pem

TLS/HTTPS：

  Gateway 只暴露 HTTPS，有两种方式：

  1. uvicorn 直启 TLS（适合单机/测试）——先生成自签证书：

         openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem \\
             -days 365 -nodes -subj "/CN=localhost"

     启动后验证：

         curl -k https://localhost:8080/health          # 应返回 {"status":"ok"}
         curl -k -H "Authorization: Bearer cp_xxx" \\
             https://localhost:8080/mcp -d @tools/list.json

  2. Nginx 反代 TLS（生产推荐）——Nginx 终结 TLS，回源到 Gateway 的 HTTP：

         server {
             listen 443 ssl;
             server_name mcp.coursepilot.example.com;
             ssl_certificate     /etc/ssl/certs/coursepilot.pem;
             ssl_certificate_key /etc/ssl/private/coursepilot.key;
             location / {
                 proxy_pass http://127.0.0.1:8080;
                 proxy_http_version 1.1;
                 proxy_set_header Host $host;
                 proxy_buffering off;
                 proxy_read_timeout 300s;
             }
         }

     此时 Gateway 用 HTTP 启动（--ssl-certfile 留空），只监听 127.0.0.1。
"""

from __future__ import annotations

import argparse
import logging

from coursepilot.config import settings
from coursepilot.mcp.gateway.app import create_app

_LOGGER = logging.getLogger("coursepilot.mcp.gateway")


def main() -> None:
    """Gateway 入口：用 uvicorn 启动。"""
    parser = argparse.ArgumentParser(description="CoursePilot MCP Gateway")
    parser.add_argument("--host", default=settings.mcp_host, help="监听地址")
    parser.add_argument("--port", type=int, default=settings.mcp_port, help="监听端口")
    parser.add_argument("--ssl-certfile", default=None, help="TLS 证书文件（启用 HTTPS）")
    parser.add_argument("--ssl-keyfile", default=None, help="TLS 私钥文件（启用 HTTPS）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import uvicorn

    scheme = "https" if args.ssl_certfile else "http"
    _LOGGER.info("启动 Gateway（%s://%s:%d）", scheme, args.host, args.port)
    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )


if __name__ == "__main__":
    main()
