import collections
import json
from collections import Counter

from src.symbols import Symbols

# Derived from the enum rather than hardcoded, so a rename cannot leave this
# list silently counting symbols that no longer exist. See src/symbols.py.
symbols = [symbol for symbol in Symbols if symbol is not Symbols.NOTHING]

class Card:
    """
    Represents a single card with its symbols and dimensions.
    """
    def __init__(self, dimensions, quarters):
        self.dimensions = dimensions
        # Resolve names as they are read, so decks in any vocabulary work.
        self.quarters = {
            name: [Symbols.of(s) for s in quarter_symbols]
            for name, quarter_symbols in quarters.items()
        }
        self.symbols = []
        for quarter_symbols in self.quarters.values():
            self.symbols.extend(quarter_symbols)

    def __repr__(self):
        return f"Card(dimensions={self.dimensions}, quarters={self.quarters})"

def analyze_card_data(file_path):
    """
    Reads card data from a JSON file, creates a list of Card objects,
    and analyzes the symbol counts for both the whole card and its quarters.

    Args:
        file_path (str): The path to the JSON file containing card data.

    Returns:
        A tuple containing:
            - A list of Card objects.
            - A Counter for the total number of each symbol across all cards.
            - A dictionary of Counter objects for each quarter, showing symbol distribution.
            - A Counter for the number of cards with exactly 'key' number of symbols.
            - A dictionary of Counter objects for each quarter, showing the number of quarters
              with exactly 'key' number of symbols.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None, None, None, None

    card_list = []
    for item in data:
        card_data = item.get("card", {})
        dimensions = card_data.get("dimensions", {})
        quarters = card_data.get("quarters", {})
        card = Card(dimensions, quarters)
        card_list.append(card)

    symbols_in_cards = collections.defaultdict(lambda : collections.defaultdict(lambda : 0))
    symbols_in_quarters = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    for card in card_list:
        card_counter = collections.Counter(card.symbols)
        for symbol in symbols:
            if symbol in card_counter.keys():
                symbols_in_cards[symbol][card_counter[symbol]] += 1
            else:
                symbols_in_cards[symbol][0] += 1
        for quarter_symbols in card.quarters.values():
            quarter_counter = collections.Counter(quarter_symbols)
            for symbol in symbols:
                if symbol in quarter_counter.keys():
                    symbols_in_quarters[symbol][quarter_counter[symbol]] += 1
                else:
                    symbols_in_quarters[symbol][0] += 1

    for symbol in Symbols.arrows():
        for no, occ in symbols_in_cards.pop(symbol).items():
            symbols_in_cards["arrow"][no] += occ
        for no, occ in symbols_in_quarters.pop(symbol).items():
            symbols_in_quarters["arrow"][no] += occ

    # Report against display names; the Symbols members are the internal keys.
    def relabel(counts):
        return {getattr(k, "display", k): v for k, v in counts.items()}

    return relabel(symbols_in_cards), relabel(symbols_in_quarters)


if __name__ == '__main__':
    file_names = ['decks/cards_20_25.json', 'decks/cards_40_45.json']
    for file_name in file_names:
        print(f"{file_name}\n")
        symbols_in_cards, symbols_in_quarters = analyze_card_data(file_name)
        for symbol, hist in symbols_in_cards.items():
            print(f"{symbol}: {dict(sorted(hist.items()))}")
        print("-"*20)

        for symbol, hist in symbols_in_quarters.items():
            print(f"{symbol}: {dict(sorted(hist.items()))}")

