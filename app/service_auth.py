"""Token de serviço entre o SaaS e a IA (brief, seção 36; risco da auditoria).

Até aqui toda rota da IA era aberta — inclusive ``/review/{id}/image`` (imagem de
cliente) e ``/fields/*`` (talhões chaveados só por id). Com ``SERVICE_TOKEN``
definido, o middleware abaixo exige ``Authorization: Bearer <token>`` em toda
rota, menos as que dizem só "estou no ar" e a documentação.

Middleware, e não dependência por rota, para valer também nas rotas dos módulos
de terreno sem precisar alterá-los.
"""

from __future__ import annotations

import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

ABERTAS = frozenset({"/health", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"})


def token_valido(authorization: str | None, esperado: str) -> bool:
    if not authorization or not authorization.lower().startswith("bearer "):
        return False
    recebido = authorization.split(" ", 1)[1].strip()
    # Comparação em tempo constante: a diferença de tempo não revela o prefixo certo.
    return hmac.compare_digest(recebido.encode(), esperado.encode())


class ServiceTokenMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token_getter) -> None:
        super().__init__(app)
        self._token_getter = token_getter

    async def dispatch(self, request: Request, call_next) -> Response:
        esperado = self._token_getter()
        if (
            esperado
            and request.method != "OPTIONS"
            and request.url.path not in ABERTAS
            and not token_valido(request.headers.get("authorization"), esperado)
        ):
            return JSONResponse(
                {"detail": "Token de serviço ausente ou inválido."},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)
