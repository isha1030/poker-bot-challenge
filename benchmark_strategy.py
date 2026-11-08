import sys,importlib.util,statistics,math,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import agent, strategy
import random
from evaluator import omaha_score
from game import *
from evaluator import warmup

def load(path):
 spec=importlib.util.spec_from_file_location('old',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def run(old_path,n=1000):
 old=load(old_path); warmup(); results=[]
 for mode in ('original','pot','call','equity'):
  for seed in ('holdout-71','holdout-83','holdout-97'):
   paired=[0]*n; times=[]; errors=0
   for seat in (0,1):
    mem={}; other={}; agent.new_half(AgentConfig(num_hands=n),mem); old.new_half(AgentConfig(),other)
    elapsed=0
    for i in range(n):
     g=Game(make_deal(seed,i),i%2,i)
     while not g.is_terminal:
      o=g.observation(g.to_act)
      if g.to_act==seat:
       t=time.perf_counter(); a=agent.choose_action(o,mem); elapsed+=time.perf_counter()-t
      elif mode=='original': a=old.choose_action(o,other)
      elif mode=='equity':
       if o.street==Street.PREFLOP: a=old.choose_action(o,other)
       else:
        deck=[c for c in range(36) if c not in o.hole_cards and c not in o.board]
        rng=random.Random(str((seed,i,seat,len(g.history))))
        wins=0
        for _ in range(12):
         draw=rng.sample(deck,10-len(o.board)); board=tuple(o.board)+tuple(draw[5:])
         x=omaha_score(o.hole_cards,board);y=omaha_score(draw[:5],board)
         wins+=(x>y)+.5*(x==y)
        q=wins/12
        a=Action.CHECK_CALL
        if q>.64 and Action.RAISE_POT in o.legal_actions: a=Action.RAISE_POT
        elif o.to_call and q<o.to_call/(o.pot+o.to_call)+.04: a=Action.FOLD
      elif mode=='pot': a=Action.RAISE_POT if Action.RAISE_POT in o.legal_actions else Action.CHECK_CALL
      else: a=Action.CHECK_CALL
      assert a in o.legal_actions; g.apply(a)
     paired[i]+=g.result().nets[seat]/BIG_BLIND
     agent.on_hand_end(g.hand_result(seat),mem); old.on_hand_end(g.hand_result(1-seat),other)
    times.append(elapsed); errors+=mem['errors']
   avg=statistics.mean(paired); delta=1.96*statistics.stdev(paired)/math.sqrt(n)
   r=dict(opponent=mode,seed=seed,bb_per_deal=avg,ci95=[avg-delta,avg+delta],strategy_seconds=times,errors=errors); print(json.dumps(r),flush=True);results.append(r)
 return results
if __name__=='__main__':
 Path(sys.argv[2]).write_text(json.dumps(run(sys.argv[1]),indent=2))
