import collections
import datetime
import enum
import random
from collections import Counter
from typing import final

import pandas as pd

from src.symbols import Symbols

CARD_SIZE = 16
LOOKAHEAD_SIMS = 50
TOTAL_TURNS = 6

def generate_card() -> collections.Counter:
    card_symbols = random.choices([symbol for symbol in Symbols], weights=[symbol.weight for symbol in Symbols], k=CARD_SIZE)
    return collections.Counter(card_symbols)

def calculate_score(collection: collections.Counter) -> int:
    total_score = 0
    for symbol, count in collection.items():
        total_score += symbol.points[min(count, 12)]
    return total_score

def evaluate_lookahead(current_collection: collections.Counter, card: collections.Counter, turns: int) -> float:
    total_future_score = 0
    for _ in range(LOOKAHEAD_SIMS):
        lookahead_collection = current_collection + card
        for i in range(turns):
            choices = [generate_card() for _ in range(3)]
            best_card = max(choices, key=lambda chosen_card: calculate_score(lookahead_collection + card))
            lookahead_collection += best_card
        total_future_score += calculate_score(lookahead_collection)
    return total_future_score / LOOKAHEAD_SIMS

def simulate_game() -> (int, Counter):
    player_collection = Counter()
    for turn in range(TOTAL_TURNS):
        card_choices = [generate_card() for _ in range(3)]
        best_card = max(card_choices, key=lambda  card: evaluate_lookahead(player_collection, card, TOTAL_TURNS - (turn + 1)))
        player_collection += best_card
    final_score = calculate_score(player_collection)
    return final_score, player_collection

def simulate_evil_game() -> (int, Counter):
    player_collection = Counter()
    for turn in range(TOTAL_TURNS):
        card_choices = [generate_card() for _ in range(3)]
        best_card = min(card_choices, key=lambda  card: evaluate_lookahead(player_collection, card, TOTAL_TURNS - (turn + 1)))
        player_collection += best_card
    final_score = calculate_score(player_collection)
    return final_score, player_collection

def simulate_mixed_game() -> (int, Counter):
    player_collection = Counter()
    for turn in range(TOTAL_TURNS):
        card_choices = [generate_card() for _ in range(3)]
        if turn < 3:
            best_card = min(card_choices, key=lambda  card: evaluate_lookahead(player_collection, card, TOTAL_TURNS - (turn + 1)))
        else:
            best_card = max(card_choices,
                            key=lambda card: evaluate_lookahead(player_collection, card, TOTAL_TURNS - (turn + 1)))
        player_collection += best_card
    final_score = calculate_score(player_collection)
    return final_score, player_collection

def simulate_5_turn_mixed_game_more_evil() -> (int, Counter):
    player_collection = Counter()
    for turn in range(TOTAL_TURNS-1):
        card_choices = [generate_card() for _ in range(3)]
        if turn < 3:
            best_card = min(card_choices, key=lambda  card: evaluate_lookahead(player_collection, card, TOTAL_TURNS - (turn + 1)))
        else:
            best_card = max(card_choices,
                            key=lambda card: evaluate_lookahead(player_collection, card, TOTAL_TURNS - (turn + 1)))
        player_collection += best_card
    final_score = calculate_score(player_collection)
    return final_score, player_collection

def simulate_5_turn_mixed_game_less_evil() -> (int, Counter):
    player_collection = Counter()
    for turn in range(TOTAL_TURNS-1):
        card_choices = [generate_card() for _ in range(3)]
        if turn < 2:
            best_card = min(card_choices, key=lambda  card: evaluate_lookahead(player_collection, card, TOTAL_TURNS - (turn + 1)))
        else:
            best_card = max(card_choices,
                            key=lambda card: evaluate_lookahead(player_collection, card, TOTAL_TURNS - (turn + 1)))
        player_collection += best_card
    final_score = calculate_score(player_collection)
    return final_score, player_collection


def run(num_games: int, game_style):
    distributions = {symbol: collections.Counter() for symbol in Symbols}
    total_score = 0

    for _ in range(num_games):
        final_score, final_collection = game_style()
        total_score += final_score
        for symbol in Symbols:
            count = final_collection.get(symbol, 0)
            distributions[symbol][count] += 1
    print(total_score/num_games)
    df = pd.DataFrame(distributions).sort_index()
    print(df)
    df.to_csv(f"simpler_simulate_5_{num_games}_{game_style.__name__}.csv")




if __name__ == "__main__":
    run(10000, simulate_5_turn_mixed_game_more_evil)
    run(10000, simulate_5_turn_mixed_game_less_evil)

