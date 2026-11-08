"""
agent.py - YOUR poker bot.

Run it:
python agent.py
PORT=5010 python agent.py

Test it:
python -m pytest -q
python baseline-bots/baseline_random.py
python judge_engine.py --deals 200
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup
from game import Action, AgentConfig, HandResult, Observation, Street


# Rename these before submission.
PARTICIPANT = "three fourths"
AGENT_NAME = "3/7"


# ===== YOUR CODE HERE =====
from strategy import initialize, choose


def new_half(config: AgentConfig, memory: dict) -> None:
    initialize(config, memory)


def choose_action(obs: Observation, memory: dict) -> Action:
    return choose(obs, memory)


def on_hand_end(result: HandResult, memory: dict) -> None:
    memory["hands_played"] = memory.get("hands_played", 0) + 1
    memory["net_chips"] = memory.get("net_chips", 0) + result.my_net


# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING =====
# Do not normally change anything below this line.

app = FastAPI()
MEMORY: dict = {}
STATUS = {"started": False}
TOKEN = os.environ.get("AGENT_TOKEN") or None


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check token and half status before parsing the JSON body."""
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse(
            {"error": "bad or missing X-Agent-Token"},
            status_code=401,
        )

    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse(
            {"error": "call /new-half first"},
            status_code=409,
        )

    try:
        body = await request.json()
    except ValueError:
        body = None

    if not isinstance(body, dict):
        return {}, JSONResponse(
            {"error": "the request body must be a JSON object"},
            status_code=400,
        )

    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from strategy code."""
    return {
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report a strategy exception to the judge."""
    return JSONResponse(
        {
            **_error_info(exc),
            "hand_end_errors": hand_end_errors or [],
        },
        status_code=500,
    )


def _deliver_results(raw_results) -> list:
    """Call on_hand_end for completed hands supplied by the judge."""
    errors = []

    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))

    return errors


@app.get("/")
async def info():
    """Health-check endpoint used by the judge."""
    return {
        "participant": PARTICIPANT,
        "agent_name": AGENT_NAME,
    }


@app.post("/new-half")
async def route_new_half(request: Request):
    """Start a fresh 1,000-hand half-match."""
    body, refused = await _read_request(request, "/new-half")

    if refused:
        return refused

    MEMORY.clear()
    STATUS["started"] = True

    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)

    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Receive an observation and return an action."""
    body, refused = await _read_request(request, "/act")

    if refused:
        return refused

    hand_end_errors = _deliver_results(body.get("hand_results"))

    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)

    # Send the enum name expected by the judge.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)

    return {
        "action": answer,
        "hand_end_errors": hand_end_errors,
    }


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Receive final hand results at the end of a half-match."""
    body, refused = await _read_request(request, "/hand-end")

    if refused:
        return refused

    return {
        "ok": True,
        "hand_end_errors": _deliver_results(body.get("results")),
    }


if __name__ == "__main__":
    # Build omaha_score's lookup table while starting the server rather than
    # during the first action request.
    warmup()

    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port)