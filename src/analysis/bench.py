import time
from src.deck.simulation import load_cards, get_paths, evaluate_scenario

cd = load_cards('pirate_cards/pirate_20_25.json')
paths = get_paths()
_ = evaluate_scenario([0,1,2,3,4,5], paths[:2], cd)

for mp in [100, 1000, 5000, 10000, 50000]:
    p = paths[:mp]
    t0 = time.time()
    s = evaluate_scenario([10,20,30,40,50,60], p, cd)
    el = time.time() - t0
    us = el / (mp * 32) * 1e6
    full = el * 729529 / mp
    print(f'{mp:>6} paths: {el:>6.2f}s | {us:>5.1f} us/cfg | full: {full/60:>5.1f} min | 2K scenarios: {el*2000/3600:>5.1f}h')
