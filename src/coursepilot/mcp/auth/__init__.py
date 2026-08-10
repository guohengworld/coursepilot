"""MCP 鉴权 / 授权包。

分层：
- ``keys.py``      KeyStore：启动时一次性载入 key 表（支持轮换）
- ``middleware.py`` AuthenticationMiddleware：解析 Bearer → 校验 → 注入 Principal
- ``policy.py``    AuthorizationPolicy：租户断言 / scope 断言装饰器
"""
