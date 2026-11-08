"""Bounded, reproducible equity policy for short-deck PLO5."""
import random
import time
from itertools import combinations
from evaluator import get_table, omaha_score
from game import Action, Street


def safe_action(obs):
    if obs.to_call == 0 and Action.CHECK_CALL in obs.legal_actions:
        return Action.CHECK_CALL
    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    if Action.CHECK_CALL in obs.legal_actions:
        return Action.CHECK_CALL
    if obs.legal_actions:
        return obs.legal_actions[0]
    raise ValueError('No legal action: observation is terminal or invalid')


def made_score(hole, board):
    """Exact Omaha combination rule, also valid on flop and turn."""
    table = get_table()
    return max(table[sum(1 << c for c in h + b)]
               for h in combinations(hole, 2) for b in combinations(board, 3))


def initialize(config, memory):
    memory.clear()
    memory.update(errors=0, last_error='', equity_cache={}, hand=None,
                  compute_seconds=0.0, compute_budget=max(0.0, config.time_bank_seconds * .35),
                  num_hands=max(1, config.num_hands), hands_played=0, net_chips=0)


def estimate_equity(obs, memory):
    if memory.get('hand') != obs.hand_index:
        memory['hand'] = obs.hand_index
        memory['equity_cache'] = {}
    key = (tuple(obs.hole_cards), tuple(obs.board))
    cache = memory['equity_cache']
    if key in cache:
        return cache[key]
    known = key[0] + key[1]
    if len(key[0]) != 5 or len(key[1]) not in (0, 3, 4, 5) or len(set(known)) != len(known) or any(c < 0 or c >= 36 for c in known):
        raise ValueError('Invalid cards')
    # Stable per-observation sampling: repeated raises cannot resample a lucky estimate.
    seed = obs.hand_index + 117
    for c in known:
        seed = seed * 37 + c + 1
    rng = random.Random(seed)
    deck = [c for c in range(36) if c not in known]
    remaining = memory.get('compute_budget', 2.8) - memory.get('compute_seconds', 0)
    hands_left = max(1, memory.get('num_hands', 1000) - obs.hand_index)
    quota = max(0.00005, remaining / hands_left / 4)
    samples = 32 if obs.street == Street.RIVER else 16
    start = time.perf_counter()
    wins = 0.0
    fixed = omaha_score(obs.hole_cards, obs.board) if len(obs.board) == 5 else None
    count = 0
    for _ in range(samples):
        draw = rng.sample(deck, 10 - len(obs.board))
        board = tuple(obs.board) + tuple(draw[5:])
        ours = fixed if fixed is not None else omaha_score(obs.hole_cards, board)
        theirs = omaha_score(draw[:5], board)
        wins += (ours > theirs) + .5 * (ours == theirs)
        count += 1
        if count >= 8 and time.perf_counter() - start >= quota:
            break
    memory['compute_seconds'] = memory.get('compute_seconds', 0) + time.perf_counter() - start
    # Mild shrinkage keeps tiny noisy samples away from absolute certainty.
    equity = (wins + 1) / (count + 2)
    cache[key] = equity
    return equity


def decide(obs, memory):
    equity = estimate_equity(obs, memory)
    price = obs.to_call / max(1, obs.pot + obs.to_call)
    # Uniform equity is optimistic against repeated value raises. This modest
    # correction is deliberately bounded; it is not a solved opponent range.
    pressure = min(.10, max(0, obs.raises_this_street - 1) * .035)
    raise_threshold = .66 + pressure
    if obs.street == Street.PREFLOP:
        raise_threshold = .62 + pressure
    if equity >= raise_threshold:
        wanted = Action.RAISE_POT if equity >= .76 else Action.RAISE_HALF
        if wanted in obs.legal_actions:
            return wanted
    margin = .025 + pressure
    if obs.to_call == 0 or equity >= price + margin:
        return Action.CHECK_CALL
    return safe_action(obs)


def choose(obs, memory):
    try:
        action = decide(obs, memory)
        if action not in obs.legal_actions:
            raise ValueError('Policy produced an illegal action')
        return action
    except Exception as exc:
        memory['errors'] = min(1000000, memory.get('errors', 0) + 1)
        memory['last_error'] = f'{type(exc).__name__}: {exc}'[:240]
        return safe_action(obs)
