from dataclasses import replace
from itertools import combinations
import pytest
import strategy
from game import Action,AgentConfig,Game,make_deal,Street
from evaluator import cards_from_str,score_five,warmup

@pytest.fixture(autouse=True,scope='module')
def table(): warmup()

def observation(): return Game(make_deal('regression',0),0).observation(0)
def memory():
 m={}; strategy.initialize(AgentConfig(),m); return m

@pytest.mark.parametrize('board',['Th Jh Qs','Th Jh Qs 6c','Th Jh Qs 6c As'])
def test_exact_combinations(board):
 h=tuple(cards_from_str('Tc Td Jc Jd 6s')); b=tuple(cards_from_str(board))
 expected=max(score_five(x+y) for x in combinations(h,2) for y in combinations(b,3))
 assert strategy.made_score(h,b)==expected

def test_fake_straight_is_trips():
 assert strategy.made_score(tuple(cards_from_str('Tc Td Jc Jd 6s')),tuple(cards_from_str('Th Jh Qs'))) >> 20 == 4

def test_four_hole_aces_are_only_pair():
 assert strategy.made_score(tuple(cards_from_str('Ac Ad Ah As Kc')),tuple(cards_from_str('6c 7d 9h'))) >> 20 == 1

@pytest.mark.parametrize('cost,expected',[(0,Action.CHECK_CALL),(40,Action.FOLD)])
def test_fault_is_conservative_and_visible(monkeypatch,cost,expected):
 o=replace(observation(),to_call=cost); m=memory()
 def fail(*args): raise RuntimeError('injected')
 monkeypatch.setattr(strategy,'decide',fail)
 assert strategy.choose(o,m)==expected
 assert m['errors']==1 and 'injected' in m['last_error']

def test_invalid_action_is_caught(monkeypatch):
 o=replace(observation(),legal_actions=(Action.FOLD,Action.CHECK_CALL)); m=memory()
 monkeypatch.setattr(strategy,'decide',lambda *args: Action.RAISE_POT)
 assert strategy.choose(o,m)==Action.FOLD and m['errors']==1

def test_cache_stable_and_bounded():
 m=memory(); o=observation(); q=strategy.estimate_equity(o,m); elapsed=m['compute_seconds']
 assert strategy.estimate_equity(o,m)==q and m['compute_seconds']==elapsed
 strategy.estimate_equity(replace(o,hand_index=1),m)
 assert len(m['equity_cache'])==1

def test_equity_responds_to_price(monkeypatch):
 monkeypatch.setattr(strategy,'estimate_equity',lambda *args:.25)
 o=replace(observation(),pot=100,to_call=2)
 assert strategy.choose(o,memory())==Action.CHECK_CALL
 assert strategy.choose(replace(o,to_call=60),memory())==Action.FOLD

def test_duplicate_cards_fail_closed():
 o=replace(observation(),hole_cards=(0,0,1,2,3));m=memory()
 assert strategy.choose(o,m)==Action.FOLD and m['errors']==1

def test_fresh_half_clears_all_state():
 m=memory();m['junk']=1;strategy.initialize(AgentConfig(),m)
 assert m==memory()
