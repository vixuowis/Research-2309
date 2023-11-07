from typing import *
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import pipeline
import asyncio

async def homepage(request):
    payload = await request.body()
    string = payload.decode("utf-8")
    response_q = asyncio.Queue()
    await request.app.model_queue.put((string, response_q))
    output = await response_q.get()
    return JSONResponse(output)
