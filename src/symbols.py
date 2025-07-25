import collections
import enum
import math
import os
from itertools import accumulate
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import plotly.express as px
import plotly.graph_objs as go
from plotly.graph_objs import Figure, Scatter

NUMBER_OF_SYMBOLS_IN_PLAY = 96

class Symbols(enum.Enum):
    CIRCLE = ("flamingo", 7,       [0,3,7,12,18,25,33,42,52,63,75,88,102], "#F3A2BD")
    SQUARE = ("red_panda", 5,    [0,1,4,9,16,25,36,49,64,81,100,121,144], "#EB7B36")
    TRIANGLE = ("cat", 4,        [0,-20,-10,0,20,39,57,74,90,105,119,132,144], "#F7DCB4")
    STAR = ("koala", 6,      [0,10,19,27,34,40,45,50,56,63,71,80,90], "#C2CCDA")
    X = ("crocodile", 8,             [0,-1,-3,-6,-10,-15,-21,-28,-36,-45,-55, -66, -78], "#739C41")
    MOON = ("shark", 6,           [0,-10,-12,-14,-17,-20,-24,-28,-33,-38,-43,-48,-54], "#4E598B")
    DIAMOND = ("butterfly", 2,       [0,25,0,50,0,75,0,100,0,125,0,150,0], "#8D79B6")
    SKULL = ("spider", 3,       [0,-36,-27,-19,-12,-6,-1,0,0,0,0,0,0], "#0F1421")
    SUN = ("elephant", 5,           [0,2,8,18,32,50,72,50,32,18,8,2,0], "#88CDF3")
    ARROW_LEFT = ("arrow_left", 1.5, [0]*13)
    ARROW_RIGHT = ("arrow_right", 1.5, [0]*13)
    ARROW_UP = ("arrow_up", 1.5,  [0, 12, 27, 41, 53, 62, 68, 70, 71, 72, 72,72,72])
    ARROW_DOWN = ("arrow_down", 1.5, [0]*13)
    NOTHING = ("", 44, [0]*13)

    def __init__(self, display: str, weigh: int, points: List, color_hex: str = "white"):
        self.display = display
        self.weight = weigh
        self.points = points
        self.color_hex = color_hex
        self.values = self.symbol_on_card_value()
        self.exact_probability = [self.calculate_probability_of_exactly_number(i, NUMBER_OF_SYMBOLS_IN_PLAY) for i in range(12)]
        self.exact_probability.append(1 - sum(self.exact_probability))

        self.expected_value = self.calculate_expected_value()

        self.probability_of_symbol_in_card = [self.calculate_probability_of_exactly_number(i, 16) for i in range(17)]
        self.probability_of_at_most_in_card = list(accumulate(self.probability_of_symbol_in_card))
        self.probability_of_at_least_in_card =  list(reversed(list(accumulate(reversed(self.probability_of_symbol_in_card)))))
        self.probability_of_max_out_of_3_cards = [self.calculate_probability_of_exactly_k_best_out_of_3_card(i) for i in range(17)]
        self.probability_of_min_out_of_3_cards = [self.calculate_probability_of_exactly_k_worse_out_of_3_card(i) for i in range(17)]
        self.probability_of_collecting_6_best_cards = np.array(self.probability_of_max_out_of_3_cards)
        for i in range(5):
            self.probability_of_collecting_6_best_cards = np.convolve(self.probability_of_collecting_6_best_cards, self.probability_of_max_out_of_3_cards)
        self.probability_of_collecting_6_best_cards[12] = np.sum(self.probability_of_collecting_6_best_cards[12:])
        self.probability_of_collecting_6_best_cards = self.probability_of_collecting_6_best_cards[:13]

        self.probability_of_collecting_6_worse_cards = np.array(self.probability_of_min_out_of_3_cards)
        for i in range(5):
            self.probability_of_collecting_6_worse_cards = np.convolve(self.probability_of_collecting_6_worse_cards, self.probability_of_min_out_of_3_cards)
        self.probability_of_collecting_6_worse_cards[12] = np.sum(self.probability_of_collecting_6_worse_cards[12:])
        self.probability_of_collecting_6_worse_cards = self.probability_of_collecting_6_worse_cards[:13]

        self.probability_of_best_and_worse_cards = np.array(self.probability_of_max_out_of_3_cards)
        for i in range(5):
            if i % 2 == 0:
                self.probability_of_best_and_worse_cards = np.convolve(self.probability_of_best_and_worse_cards,
                                                                      self.probability_of_min_out_of_3_cards)
            else:
                self.probability_of_best_and_worse_cards = np.convolve(self.probability_of_best_and_worse_cards,
                                                                       self.probability_of_max_out_of_3_cards)
        self.probability_of_best_and_worse_cards[12] = np.sum(self.probability_of_best_and_worse_cards[12:])
        self.probability_of_best_and_worse_cards = self.probability_of_best_and_worse_cards[:13]

    def calculate_probability_of_exactly_k_best_out_of_3_card(self, k: int) -> float:
        if k == 0:
            return self.probability_of_at_most_in_card[k] ** 3
        return self.probability_of_at_most_in_card[k] ** 3 - self.probability_of_at_most_in_card[k-1] ** 3

    def calculate_probability_of_exactly_k_worse_out_of_3_card(self, k: int) -> float:
        if k >= len(self.probability_of_at_least_in_card) - 1:
            return self.probability_of_at_least_in_card[k] ** 3
        return self.probability_of_at_least_in_card[k] ** 3 - self.probability_of_at_least_in_card[k+1] ** 3


    def calculate_expected_value(self, already_exists: int = 0, total_symbols: int =  NUMBER_OF_SYMBOLS_IN_PLAY):
        expected_value = 0
        total_probabilities = 0
        for i in range(len(self.points)-1-already_exists):
            probability_of_exactly_i = self.calculate_probability_of_exactly_number(i, total_symbols)
            total_probabilities += probability_of_exactly_i
            expected_value += probability_of_exactly_i * self.points[i+already_exists]
        return expected_value + (1-total_probabilities) * self.points[-1]

    def calculate_probability_of_exactly_number(self, x: int, total_symbols: int):
        p = self.weight / NUMBER_OF_SYMBOLS_IN_PLAY
        q = 1-p
        binom_coeff = scipy.special.binom(total_symbols, x)
        return binom_coeff * math.pow(p, x) * math.pow(q, total_symbols-x)

    def value_symbol(self) -> bool:
        return self not in [Symbols.NOTHING, Symbols.ARROW_UP, Symbols.ARROW_RIGHT, Symbols.ARROW_DOWN, Symbols.ARROW_LEFT]

    @staticmethod
    def arrows() -> List["Symbols"]:
        return [Symbols.ARROW_LEFT, Symbols.ARROW_UP, Symbols.ARROW_RIGHT, Symbols.ARROW_DOWN]


    @staticmethod
    def arrow_coordinate(arrow: "Symbols") -> Tuple[int, int]:
        if arrow is Symbols.ARROW_UP:
            return -1, 0
        elif arrow is Symbols.ARROW_DOWN:
            return 1, 0
        elif arrow is Symbols.ARROW_RIGHT:
            return 0, 1
        elif arrow is Symbols.ARROW_LEFT:
            return 0, -1
        else:
            raise ValueError(f"{arrow} is not an arrow.")

    @staticmethod
    def good_symbols_to_multiply():
        return [Symbols.CIRCLE, Symbols.SQUARE, Symbols.TRIANGLE, Symbols.STAR, Symbols.SKULL, Symbols.SUN]

    @staticmethod
    def bad_symbols_to_multiply():
        return [Symbols.MOON, Symbols.X]

    def symbol_on_card_value(self) -> List[int]:
        symbol_points = collections.defaultdict(lambda: list())
        average_symbol_points = list()

        for symbol_number in range(11):
            for no_cards in range(1, 7):
                old_number_of_symbols = self.weight * ((no_cards - 1) / 7)
                old_point_smaller = self.calculate_expected_value(int(old_number_of_symbols), 3 * 4 * (7-no_cards))
                old_point_higher = self.calculate_expected_value(int(old_number_of_symbols) + 1, 3 * 4 * (7-no_cards))
                old_point = old_point_higher * (old_number_of_symbols % 1) + old_point_smaller * (
                            1 - old_number_of_symbols % 1)

                new_number_of_symbols = self.weight * ((no_cards - 1) / 7) + symbol_number
                new_point_smaller = self.calculate_expected_value(int(new_number_of_symbols), 3 * 4 * (6-no_cards))
                new_point_higher = self.calculate_expected_value(int(new_number_of_symbols) + 1, 3 * 4 * (6-no_cards))
                new_point = new_point_higher * (new_number_of_symbols % 1) + new_point_smaller * (
                        1 - new_number_of_symbols % 1)
                point = new_point - old_point
                symbol_points[symbol_number].append(point)
            average_symbol_points.append(sum(symbol_points[symbol_number]) / len(
                symbol_points[symbol_number]))
        return average_symbol_points



    def __str__(self):
        return self.display

    def __repr__(self):
        return self.display

print(sum([symbol.weight for symbol in Symbols]))
assert sum([symbol.weight for symbol in Symbols]) == NUMBER_OF_SYMBOLS_IN_PLAY

def create_df() -> pd.DataFrame:
    plot_data = []
    for symbol in Symbols:
        if symbol.value_symbol():
            for i in range(13):
                plot_data.append({"Symbol": symbol.display,
                                  "Value": symbol.points[i],
                                  "Quantity": i,
                                  "Color": symbol.color_hex,
                                  "ExactlyRandomProbability": symbol.exact_probability[i],
                                  "ExactlyFocusMaxProbability": symbol.probability_of_collecting_6_best_cards[i],
                                  "ExactlyFocusMinProbability": symbol.probability_of_collecting_6_worse_cards[i],
                                  "ExactlyFocusMeanProbability": symbol.probability_of_best_and_worse_cards[i]
                                  })

    df = pd.DataFrame(plot_data).sort_values(by=['Symbol', 'Value'], ascending=[True, False])
    df["AtLeastValueRandomProbability"] = df.groupby("Symbol")["ExactlyRandomProbability"].cumsum()
    df["AtLeastValueFocusMaxProbability"] = df.groupby("Symbol")["ExactlyFocusMaxProbability"].cumsum()
    df["AtLeastValueFocusMinProbability"] = df.groupby("Symbol")["ExactlyFocusMinProbability"].cumsum()
    df["AtLeastValueFocusMeanProbability"] = df.groupby("Symbol")["ExactlyFocusMeanProbability"].cumsum()

    df = df.sort_values(by=['Symbol', 'Value'], ascending=[True, True])
    df["AtMostValueRandomProbability"] = df.groupby("Symbol")["ExactlyRandomProbability"].cumsum()
    df["AtMostValueFocusMaxProbability"] = df.groupby("Symbol")["ExactlyFocusMaxProbability"].cumsum()
    df["AtMostValueFocusMinProbability"] = df.groupby("Symbol")["ExactlyFocusMinProbability"].cumsum()
    df["AtMostValueFocusMeanProbability"] = df.groupby("Symbol")["ExactlyFocusMeanProbability"].cumsum()

    df = df.sort_values(by=['Symbol', 'Quantity'], ascending=[True, False])
    df["AtLeastQuantityRandomProbability"] = df.groupby("Symbol")["ExactlyRandomProbability"].cumsum()
    df["AtLeastQuantityFocusMaxProbability"] = df.groupby("Symbol")["ExactlyFocusMaxProbability"].cumsum()
    df["AtLeastQuantityFocusMinProbability"] = df.groupby("Symbol")["ExactlyFocusMinProbability"].cumsum()
    df["AtLeastQuantityFocusMeanProbability"] = df.groupby("Symbol")["ExactlyFocusMeanProbability"].cumsum()

    df = df.sort_values(by=['Symbol', 'Quantity'], ascending=[True, True])
    df["AtMostQuantityRandomProbability"] = df.groupby("Symbol")["ExactlyRandomProbability"].cumsum()
    df["AtMostQuantityFocusMaxProbability"] = df.groupby("Symbol")["ExactlyFocusMaxProbability"].cumsum()
    df["AtMostQuantityFocusMinProbability"] = df.groupby("Symbol")["ExactlyFocusMinProbability"].cumsum()
    df["AtMostQuantityFocusMeanProbability"] = df.groupby("Symbol")["ExactlyFocusMeanProbability"].cumsum()

    return df

def generate_symbol_point_graph_and_df(original_df: pd.DataFrame, color_map: Dict[str, str]) ->  Tuple[Figure, pd.DataFrame]:
    fig = px.line(original_df,
                  x="Quantity",
                  y="Value",
                  color="Symbol",
                  labels={ "Quantity": "Number of symbols"},
                  color_discrete_map=color_map,
                  markers="o",
                  title="Points of symbols")
    table_df = pd.DataFrame({symbol.display: symbol.points for symbol in Symbols if symbol.value_symbol()}).T
    table_df["Weight"] = {symbol.display: symbol.weight for symbol in Symbols if symbol.value_symbol()}
    table_df = table_df.drop([0], axis=1)
    table_df.loc["Arrow"] = [""] * 12 + [sum([symbol.weight for symbol in Symbols.arrows()])]
    table_df.loc[""] = [""] * 12 + [Symbols.NOTHING.weight]
    return fig, table_df

def figure_and_expected_value(df: pd.DataFrame, color_map: Dict[str, str], column_name: str, *args, **kwargs):
    fig = px.line(df,
                  x=column_name,
                  y='Value',
                  color='Symbol',
                  color_discrete_map=color_map,
                  markers="o",
                  labels={column_name: "Probability"},
                  hover_data=['Symbol', 'Value', column_name, "Quantity"],
                  **kwargs)
    fig.update_layout(xaxis={"tickformat": ',.0%'}).show()
    expected_value = (df['Value'] * df[column_name]).groupby(df['Symbol']).sum()
    return fig, pd.DataFrame({"expected value": expected_value})

def generate_image_and_table(function, name: str, *args, **kwargs):
    output_dir = "../content/plotly_graphs"
    os.makedirs(output_dir, exist_ok=True)
    image, df = function(*args, **kwargs)
    image.write_html(os.path.join(output_dir, f"{name}.html"), full_html=False,
                                  include_plotlyjs="cdn")
    df.to_html(os.path.join(output_dir, f"{name}_table.html"))

def generate_composite_figure(df: pd.DataFrame):
    probability_cols = ['ExactlyRandomProbability', 'ExactlyFocusMaxProbability', 'ExactlyFocusMinProbability', 'ExactlyFocusMeanProbability', 'AtLeastValueRandomProbability', 'AtLeastValueFocusMaxProbability', 'AtLeastValueFocusMinProbability', 'AtLeastValueFocusMeanProbability', 'AtMostValueRandomProbability', 'AtMostValueFocusMaxProbability', 'AtMostValueFocusMinProbability', 'AtMostValueFocusMeanProbability', 'AtLeastQuantityRandomProbability', 'AtLeastQuantityFocusMaxProbability', 'AtLeastQuantityFocusMinProbability', 'AtLeastQuantityFocusMeanProbability', 'AtMostQuantityRandomProbability', 'AtMostQuantityFocusMaxProbability', 'AtMostQuantityFocusMinProbability', 'AtMostQuantityFocusMeanProbability']
    initial_column = probability_cols[0]

    fig = px.line(df,
                  x=initial_column,
                  y='Value',
                  color='Symbol',
                  color_discrete_map=color_map,
                  markers="o",
                  hover_data=['Symbol', 'Value', initial_column, "Quantity"])

    buttons = []
    for col_name in probability_cols:
        temp_df =  df.sort_values(by=["Symbol", col_name])
        buttons.append(
            dict(
                method='update',
                label=col_name,  # Label displayed in the dropdown
                args=[
                    {'x': [temp_df[temp_df['Symbol'] == s][col_name].tolist() for s in df['Symbol'].unique()],
                     'y': [temp_df[temp_df['Symbol'] == s]["Value"].tolist() for s in df['Symbol'].unique()],
                     },
                    {'xaxis.title.text': col_name,  # Correct way to set x-axis title
                     'xaxis.tickformat': ',.0%'
                     }
                ]
            )
        )

    # Add the dropdown to the layout
    fig.update_layout(
        xaxis={"tickformat": ',.0%'},  # Apply initial tick format
        updatemenus=[
            go.layout.Updatemenu(
                type="dropdown",
                direction="down",
                x=0.01,  # Position of the dropdown
                y=1.15,
                showactive=True,
                active=0,  # Default selected item (first column)
                buttons=buttons,
                xanchor="left",
                yanchor="top"
            )
        ],
        title_text="Line Chart with Dynamic X-Axis Selection",
        title_x=0.5  # Center the title
    )

    fig.show()

def generate_melted_figure(df: pd.DataFrame):
    probability_cols = ['ExactlyRandomProbability', 'ExactlyFocusMaxProbability', 'ExactlyFocusMinProbability', 'ExactlyFocusMeanProbability',
                        'AtLeastValueRandomProbability', 'AtLeastValueFocusMaxProbability',
                        'AtLeastValueFocusMinProbability', 'AtLeastValueFocusMeanProbability',
                        'AtMostValueRandomProbability', 'AtMostValueFocusMaxProbability',
                        'AtMostValueFocusMinProbability', 'AtMostValueFocusMeanProbability',
                        'AtLeastQuantityRandomProbability', 'AtLeastQuantityFocusMaxProbability',
                        'AtLeastQuantityFocusMinProbability', 'AtLeastQuantityFocusMeanProbability',
                        'AtMostQuantityRandomProbability', 'AtMostQuantityFocusMaxProbability',
                        'AtMostQuantityFocusMinProbability', 'AtMostQuantityFocusMeanProbability']

    # Melt the DataFrame
    df_melted = df.melt(
        id_vars=['Symbol', 'Value', 'Quantity', 'Color'],
        value_vars=probability_cols,
        var_name='Probability_Type',
        value_name='Probability_Value'
    )

    df_melted["prefix"] = df_melted["Probability_Type"].str.removesuffix('RandomProbability').str.removesuffix('FocusMaxProbability').str.removesuffix('FocusMinProbability').str.removesuffix('FocusMeanProbability')
    df_melted["suffix"] = df_melted["Probability_Type"].str.removeprefix('AtLeastValue').str.removeprefix('AtMostValue').str.removeprefix('AtLeastQuantity').str.removeprefix('AtMostQuantity').str.removeprefix('Exactly')

    fig = px.line(df_melted,
                  x='Probability_Value',
                  y='Value',
                  color='Symbol',
                  color_discrete_map=color_map,
                  line_dash='prefix',
                  symbol='suffix',
                  title='Value vs. Different Probabilities per Symbol',
                  hover_data=['Symbol', 'Value', 'Probability_Value', "Quantity"]
                  )
    fig.update_traces(marker={'size': 10})
    fig.show()







if __name__ == "__main__":
    df = create_df()
    color_map = {symbol.display: symbol.color_hex for symbol in Symbols}

    generate_melted_figure(df)


    #
    # generate_image_and_table(generate_symbol_point_graph_and_df, "symbol_point", df, color_map)
    # generate_image_and_table(figure_and_expected_value, "symbol_point_random", df, color_map, "RandomProbability", title="Randomly collecting symbols")
    # generate_image_and_table(figure_and_expected_value, "symbol_max_prob_point", df, color_map, "FocusMaxProbability",
    #                          title="Collecting most symbols out of 3 cards")
    # generate_image_and_table(figure_and_expected_value, "symbol_min_prob_point", df, color_map, "FocusMinProbability",
    #                         title="Collecting least symbols out of 3 cards")
    # generate_image_and_table(figure_and_expected_value, "symbol_mean_prob_point", df, color_map, "FocusMeanProbability",
    #                          title="Half time collecting the most, half time the least symbols out of three cards")
