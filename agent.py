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
PARTICIPANT = "3/4ths"
AGENT_NAME = "3/7"


# ===== YOUR CODE HERE =====

def _safe_action(obs: Observation) -> Action:
    """Always return a legal low-risk fallback action."""
    if Action.CHECK_CALL in obs.legal_actions:
        return Action.CHECK_CALL
    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return obs.legal_actions[0]


def _raise(obs: Observation, pot: bool = False) -> Action:
    """Return the requested raise if legal; otherwise safely continue."""
    wanted = Action.RAISE_POT if pot else Action.RAISE_HALF
    other = Action.RAISE_HALF if pot else Action.RAISE_POT

    if wanted in obs.legal_actions:
        return wanted
    if other in obs.legal_actions:
        return other
    return _safe_action(obs)


def _connectedness(ranks: list[int]) -> int:
    """
    Score Short-Deck connectivity.

    Ranks are 0=6 through 8=A. The ace may also be used below a six for
    A-6-7-8-9 straights, so Ace and six count as adjacent.
    """
    unique = set(ranks)
    score = 0

    for rank in unique:
        if (rank + 1) in unique:
            score += 1

    # A can be low in A-6-7-8-9.
    if 8 in unique and 0 in unique:
        score += 1

    # Reward three/four-card runs more heavily.
    extended = set(unique)
    if 8 in unique:
        extended.add(-1)

    best_run = 1
    current = 1
    ordered = sorted(extended)

    for i in range(1, len(ordered)):
        if ordered[i] == ordered[i - 1] + 1:
            current += 1
            best_run = max(best_run, current)
        else:
            current = 1

    if best_run >= 4:
        score += 5
    elif best_run == 3:
        score += 2

    return score


def _preflop_strength(hole_cards: list[int]) -> float:
    """
    Fast preflop approximation for five-card Short Deck Omaha.

    Good hands generally have:
    - high ranks
    - pairs, especially high pairs
    - multiple suited cards
    - connected/near-connected ranks
    """
    ranks = [card_rank(card) for card in hole_cards]
    suits = [card_suit(card) for card in hole_cards]

    rank_counts = [ranks.count(rank) for rank in range(9)]
    suit_counts = [suits.count(suit) for suit in range(4)]

    strength = 0.0

    # High-card quality: Q/K/A matter especially in short deck.
    strength += sum(ranks) * 1.35
    strength += sum(1 for rank in ranks if rank >= 6) * 2.0
    strength += sum(1 for rank in ranks if rank == 8) * 1.0

    # Pairs are powerful because Omaha must use two hole cards.
    for rank, count in enumerate(rank_counts):
        if count >= 2:
            strength += 8.0 + rank * 2.0
        if count >= 3:
            strength += 4.0 + rank
        if count >= 4:
            strength += 8.0

    # Suited pairs / double-suited structures provide flush potential.
    for count in suit_counts:
        if count >= 2:
            strength += 3.5
        if count >= 3:
            strength += 2.0
        if count >= 4:
            strength += 1.0

    strength += _connectedness(ranks) * 2.2

    # Avoid weak, disconnected, low rainbow hands.
    unique_ranks = len(set(ranks))
    if max(ranks) <= 3 and unique_ranks >= 4:
        strength -= 5.0

    return strength


def _postflop_strength(obs: Observation) -> float:
    """
    Approximate current made-hand strength and draw quality before the river.

    This deliberately avoids expensive Monte Carlo work on every decision:
    the tournament time bank is only eight seconds per 1,000-hand half.
    """
    hole = list(obs.hole_cards)
    board = list(obs.board)

    if len(board) == 5:
        score = omaha_score(hole, board)
        category = score >> 20
        # Categories: 0 high card ... 8 straight flush.
        return 100.0 + category * 18.0 + (score & ((1 << 20) - 1)) / 100000.0

    ranks = [card_rank(card) for card in hole]
    suits = [card_suit(card) for card in hole]
    board_ranks = [card_rank(card) for card in board]
    board_suits = [card_suit(card) for card in board]

    rank_counts = [ranks.count(rank) for rank in range(9)]
    hole_suit_counts = [suits.count(suit) for suit in range(4)]
    board_suit_counts = [board_suits.count(suit) for suit in range(4)]

    strength = _preflop_strength(hole) * 0.55

    # Hole pairs plus matching board cards can make trips/full houses.
    for rank, count in enumerate(rank_counts):
        if count >= 2:
            matches = board_ranks.count(rank)
            strength += 10.0 + rank * 1.4
            if matches >= 1:
                strength += 16.0 + rank * 1.5
            if matches >= 2:
                strength += 12.0

    # Board pairing gives full-house possibilities when holding matching ranks.
    if len(set(board_ranks)) < len(board_ranks):
        strength += 4.0

    # Omaha flushes require exactly two hole cards and three board cards.
    for suit in range(4):
        if hole_suit_counts[suit] >= 2:
            strength += 3.0
            matching_board = board_suit_counts[suit]

            if matching_board >= 3:
                strength += 28.0
            elif matching_board == 2:
                strength += 13.0
            elif matching_board == 1:
                strength += 4.0

    # Straight potential: reward hole cards that connect with board ranks.
    combined = set(ranks + board_ranks)
    straight_windows = [
        {0, 1, 2, 3, 4},  # 6-T
        {1, 2, 3, 4, 5},  # 7-J
        {2, 3, 4, 5, 6},  # 8-Q
        {3, 4, 5, 6, 7},  # 9-K
        {4, 5, 6, 7, 8},  # T-A
        {8, 0, 1, 2, 3},  # A-9, ace low
    ]

    for window in straight_windows:
        board_in_window = sum(1 for rank in board_ranks if rank in window)
        hole_in_window = sum(1 for rank in ranks if rank in window)

        # Must eventually use exactly 2 hole cards and 3 board cards.
        if board_in_window >= 3 and hole_in_window >= 2:
            strength += 22.0
        elif board_in_window >= 2 and hole_in_window >= 2:
            strength += 7.0

    # High board cards reduce the relative value of weak holdings.
    if board_ranks:
        strength += sum(board_ranks) * 0.25

    return strength


def _record_opponent_actions(obs: Observation, memory: dict) -> None:
    """Count opponent raises without double-counting repeated history."""
    current_hand = obs.hand_index
    processed = memory.get("history_seen", {})

    seen = processed.get(current_hand, 0)
    history = obs.action_history

    if seen > len(history):
        seen = 0

    for item in history[seen:]:
        if not item.is_me:
            memory["opp_actions"] = memory.get("opp_actions", 0) + 1
            if item.action in (Action.RAISE_HALF, Action.RAISE_POT):
                memory["opp_raises"] = memory.get("opp_raises", 0) + 1

    processed[current_hand] = len(history)

    # Keep this small; old entries are never needed again.
    if len(processed) > 12:
        oldest = sorted(processed)[:-8]
        for hand_index in oldest:
            processed.pop(hand_index, None)

    memory["history_seen"] = processed


def _opponent_is_aggressive(memory: dict) -> bool:
    actions = memory.get("opp_actions", 0)
    raises = memory.get("opp_raises", 0)

    if actions < 12:
        return False

    return raises / max(actions, 1) > 0.30


def new_half(config: AgentConfig, memory: dict) -> None:
    """Initialize per-half state."""
    memory["hands_played"] = 0
    memory["opp_actions"] = 0
    memory["opp_raises"] = 0
    memory["showdowns"] = 0
    memory["history_seen"] = {}
    memory["starting_stack"] = getattr(config, "starting_stack", 80)


def choose_action(obs: Observation, memory: dict) -> Action:
    """
    Choose a legal action.

    Strategy outline:
    - Raise strong/connected/paired/suited preflop holdings.
    - Fold bad hands facing substantial pressure.
    - Continue with draws and made hands postflop.
    - Use omaha_score only on the river, when it is valid.
    - Bet/raise strong river hands for value.
    """
    try:
        _record_opponent_actions(obs, memory)

        if not obs.legal_actions:
            # This should never happen, but prevents an exception if malformed.
            return Action.CHECK_CALL

        aggressive_opponent = _opponent_is_aggressive(memory)

        # ---------- PREFLOP ----------
        if obs.street == Street.PREFLOP:
            strength = _preflop_strength(list(obs.hole_cards))

            # Slightly loosen up against frequent raisers: calling their raises
            # with reasonable holdings is better than over-folding.
            call_adjustment = -2.0 if aggressive_opponent else 0.0

            # No raise yet / free big-blind option.
            if obs.to_call == 0:
                if strength >= 45:
                    return _raise(obs, pot=True)
                if strength >= 34:
                    return _raise(obs, pot=False)
                return _safe_action(obs)

            # Opening from the button: calling one chip is cheap, but premium
            # hands should build the pot.
            if obs.to_call <= obs.big_blind:
                if strength >= 49:
                    return _raise(obs, pot=True)
                if strength >= 38:
                    return _raise(obs, pot=False)
                if strength >= 20:
                    return Action.CHECK_CALL
                return _safe_action(obs)

            # Facing a normal raise or re-raise.
            pot_odds = obs.to_call / max(obs.pot + obs.to_call, 1)

            if strength >= 57:
                return _raise(obs, pot=True)
            if strength >= 47:
                return _raise(obs, pot=False)
            if strength >= 33 + call_adjustment:
                return Action.CHECK_CALL

            # Cheap calls are allowed with borderline playable holdings.
            if pot_odds <= 0.20 and strength >= 25:
                return Action.CHECK_CALL

            if Action.FOLD in obs.legal_actions:
                return Action.FOLD
            return _safe_action(obs)

        # ---------- FLOP / TURN / RIVER ----------
        strength = _postflop_strength(obs)

        # River strengths have a 100+ base. Category 5 = trips, category 6 =
        # straight, category 7 = full house, category 8 = flush/quads/SF
        # according to the evaluator's encoding.
        is_river = obs.street == Street.RIVER

        if is_river:
            category = omaha_score(obs.hole_cards, obs.board) >> 20

            # Flushes beat full houses in Short Deck, and trips beat straights.
            # Therefore category thresholds must follow evaluator categories,
            # not standard Hold'em intuition.
            if obs.to_call == 0:
                if category >= 7:
                    return _raise(obs, pot=True)
                if category >= 5:
                    return _raise(obs, pot=False)
                return _safe_action(obs)

            if category >= 7:
                return _raise(obs, pot=True)
            if category >= 5:
                return _raise(obs, pot=False)
            if category >= 3:
                return Action.CHECK_CALL

            if obs.to_call <= obs.big_blind:
                return Action.CHECK_CALL
            if Action.FOLD in obs.legal_actions:
                return Action.FOLD
            return _safe_action(obs)

        # Flop and turn.
        # Strong made hands / powerful draws.
        if obs.to_call == 0:
            if strength >= 64:
                return _raise(obs, pot=True)
            if strength >= 48:
                return _raise(obs, pot=False)
            return _safe_action(obs)

        pot_odds = obs.to_call / max(obs.pot + obs.to_call, 1)

        if strength >= 68:
            return _raise(obs, pot=True)
        if strength >= 53:
            return _raise(obs, pot=False)

        # Draws and good pairs can call reasonable prices.
        required_strength = 27.0 + pot_odds * 55.0
        if aggressive_opponent:
            required_strength -= 3.0

        if strength >= required_strength:
            return Action.CHECK_CALL

        # Always defend very cheap calls slightly more often.
        if obs.to_call <= obs.big_blind and strength >= 18:
            return Action.CHECK_CALL

        if Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return _safe_action(obs)

    except Exception:
        # A legal fallback is much better than an error/forfeit.
        return _safe_action(obs)


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Record simple per-half information without risking expensive work."""
    memory["hands_played"] = memory.get("hands_played", 0) + 1

    if result.went_to_showdown:
        memory["showdowns"] = memory.get("showdowns", 0) + 1

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