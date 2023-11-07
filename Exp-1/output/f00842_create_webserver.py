from typing import *
import asyncio

async def create_webserver(q: asyncio.Queue) -> None:
    # Load the model only once
    # Queuing mechanism allows for dynamic batching
    strings = []
    queues = []
    while True:
        try:
            (string, rq) = await asyncio.wait_for(q.get(), timeout=0.001)  # 1ms
        except asyncio.exceptions.TimeoutError:
            break
        strings.append(string)
        queues.append(rq)
    strings
    outs = pipe(strings, batch_size=len(strings))
    for rq, out in zip(queues, outs):
        await rq.put(out)
