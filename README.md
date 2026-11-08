# Poker Bot Challenge — 3/7

**3/7** is team **three fourths**’ entry in the TAMU Datathon Poker Bot Challenge. It plays heads-up short-deck, five-card pot-limit Omaha. With eight seconds of thinking time across 1,000 hands, the bot samples possible showdowns and uses the price of a call to decide how to play.

## Play a local match

Use Python 3.10 or newer. Open PowerShell in the project folder containing `agent.py` and install the runtime and test dependencies.

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Open three terminals in that same folder. Start the two servers first and leave them running while the judge plays the match.

| Terminal | Command | Purpose |
| --- | --- | --- |
| 1 | `python agent.py` | Start 3/7 on port 5008. |
| 2 | `python baseline-bots/baseline_random.py` | Start the starter kit’s random opponent on port 5009. |
| 3 | `python judge_engine.py --deals 200 --seed 7` | Play 200 deals twice, swapping seats for the second half. |

To check the first server, open <http://127.0.0.1:5008/>. It should return the participant and agent names. The match report appears in terminal 3. Look for `RESULT` and check Agent 1’s faults and time usage in both halves. For a full 2,000-hand match, change `--deals 200` to `--deals 1000`.

The random opponent comes with the [TAMU Datathon starter kit](https://github.com/tamu-datathon-org/heads-up-bundle).

## How 3/7 chooses a move

1. **Sample the unseen cards.** The bot removes its five hole cards and the visible board from the 36-card deck. It samples an opponent hand and any missing board cards, then scores both hands using exactly two private cards and three board cards. A tie counts as half a win.
2. **Compare hand strength with the price.** Calling 10 chips into a pot that will total 40 costs 25% of the final pot. The bot compares its estimated share with that price, adds a small margin for uncertainty, and becomes more cautious after repeated raises. Strong hands can trigger a half-pot or pot-sized raise.
3. **Reuse the estimate.** If a player raises again on the same board, the price changes while the known cards stay the same. The bot reuses its estimate, checks the new price, and verifies that its chosen move is legal.

The decision logic lives in [strategy.py](strategy.py). The FastAPI server in [agent.py](agent.py) passes observations from the judge to the strategy. We used ChatGPT (5.6 Terra) during development. Match decisions run locally in Python.

## Test results

In one recorded 2,000-hand match through the HTTP judge, 3/7 averaged **+6.835 big blinds per paired deal** with **zero errors**. It used **3.73 s** in the first half and **4.18 s** in the second, out of 8 s available per half. Each paired deal combines the results from playing the same cards in both seats.

Across three benchmark seeds, 3/7 finished ahead of the previous strategy, an always-call bot, and an always-pot bot. A challenger that also estimates showdown equity finished slightly ahead of 3/7 on all three seeds, though each 95% confidence interval included zero. One next experiment is to weight stronger opponent hands more heavily after repeated raises.

See [TESTING.md](TESTING.md) for per-seed results and benchmark reproduction instructions.
