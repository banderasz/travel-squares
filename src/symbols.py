import collections
import enum
import math
import os
from collections import defaultdict
from itertools import accumulate
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import plotly.express as px
import plotly.graph_objs as go
from plotly.graph_objs import Figure, Scatter
from plotly.graph_objs.layout.map.layer import Circle

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
    ARROW_UP = ("arrow_up", 1.5,  [0]*13)
    ARROW_DOWN = ("arrow_down", 1.5, [0]*13)
    NOTHING = ("", 44, [0]*13)

    def __init__(self, display: str, weigh: int, points: List, color_hex: str = "white"):
        self.display = display
        self.weight = weigh
        self.points = points
        self.color_hex = color_hex



        self.exact_probability = [self.calculate_probability_of_exactly_number(i, NUMBER_OF_SYMBOLS_IN_PLAY) for i in range(12)]
        self.exact_probability.append(1 - sum(self.exact_probability))

        self.expected_value = self.calculate_expected_value()

        self.quarter_probabilities = np.array([self.calculate_probability_of_exactly_number(i, 4) for i in range(5)])

        self.probability_of_symbol_in_card = [self.calculate_probability_of_exactly_number(i, 16) for i in range(17)]

        self.probability_of_symbol_in_n_card = [np.array(self.probability_of_symbol_in_card)]
        for i in range(5):
            self.probability_of_symbol_in_n_card.append(np.convolve(self.probability_of_symbol_in_n_card[-1], self.probability_of_symbol_in_n_card[0]))

        self.probability_of_at_most_in_card = list(accumulate(self.probability_of_symbol_in_card))
        self.probability_of_at_least_in_card =  list(reversed(list(accumulate(reversed(self.probability_of_symbol_in_card)))))
        self.probability_of_max_out_of_3_cards = [self.calculate_probability_of_exactly_k_best_out_of_3_card(i) for i in range(17)]
        self.probability_of_min_out_of_3_cards = [self.calculate_probability_of_exactly_k_worse_out_of_3_card(i) for i in range(17)]


        self.probability_of_collecting_best_cards = [np.array(self.probability_of_max_out_of_3_cards)]
        for i in range(5):
            self.probability_of_collecting_best_cards.append(np.convolve(self.probability_of_collecting_best_cards[-1], self.probability_of_max_out_of_3_cards))

        self.probability_of_collecting_worse_cards = [np.array(self.probability_of_min_out_of_3_cards)]
        for i in range(5):
            self.probability_of_collecting_worse_cards.append(np.convolve(self.probability_of_collecting_worse_cards[-1], self.probability_of_min_out_of_3_cards))

        self.probability_of_mean_max_cards = [np.array(self.probability_of_max_out_of_3_cards)]
        self.probability_of_mean_min_cards = [np.array(self.probability_of_min_out_of_3_cards)]
        for i in range(5):
            if i % 2 == 0:
                self.probability_of_mean_max_cards.append(np.convolve(self.probability_of_mean_max_cards[-1],
                                                                      self.probability_of_min_out_of_3_cards))
                self.probability_of_mean_min_cards.append(np.convolve(self.probability_of_mean_min_cards[-1],
                                                                 self.probability_of_max_out_of_3_cards))
            else:
                self.probability_of_mean_max_cards.append(np.convolve(self.probability_of_mean_max_cards[-1],
                                                                 self.probability_of_max_out_of_3_cards))
                self.probability_of_mean_min_cards.append(np.convolve(self.probability_of_mean_min_cards[-1],
                                                                 self.probability_of_min_out_of_3_cards))

        # self.values = self.symbol_on_card_value()
        self.values = self.symbol_on_card_value_mean()

    def calculate_half_quarter_probability(self):
        p = self.quarter_probabilities
        fft_length = np.pow(len(p), 2)
        padded_p = np.pad(p, (0, fft_length - len(p)))

        # Compute the FFT, take the element-wise square root, and then the inverse FFT
        fft_p = np.fft.fft(padded_p)
        sqrt_fft_p = np.pow(fft_p, 1 / 2)
        q_padded = np.fft.ifft(sqrt_fft_p)
        q = q_padded.real[:len(p)]
        return q


    def calculate_expected_value_from_dist(self, prob: np.array) -> float:
        min_prob_5 = np.convolve(self.probability_of_mean_min_cards[4], prob)
        max_prob_5 = np.convolve(self.probability_of_mean_max_cards[4], prob)
        mean_prob_5 = (min_prob_5 + max_prob_5) / 2
        e_5 = self.expected_value_of_dist(mean_prob_5)
        e_6 =  self.expected_value_of_dist(self.probability_of_mean_min_cards[5])
        return e_5 - e_6

    def expected_value_of_dist(self, prob_orig: np.array) -> float:
        prob = prob_orig.copy()
        prob[12] = np.sum(prob[12:])
        prob = prob[:13]
        return (prob * np.array(self.points)).sum()


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
        if arrow == Symbols.ARROW_UP:
            return -1, 0
        elif arrow == Symbols.ARROW_DOWN:
            return 1, 0
        elif arrow == Symbols.ARROW_RIGHT:
            return 0, 1
        elif arrow == Symbols.ARROW_LEFT:
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
        q_to_value = collections.defaultdict(lambda: self.points[12])
        for i in range(len(self.points)):
            q_to_value[i] = self.points[i]

        probabilities_5 = np.array(self.probability_of_symbol_in_n_card[4])
        probabilities_5[12] = np.sum(probabilities_5[12:])
        probabilities_5 = probabilities_5[:13]

        probabilities_6 = np.array(self.probability_of_symbol_in_n_card[5])
        probabilities_6[12] = np.sum(probabilities_6[12:])
        probabilities_6 = probabilities_6[:13]

        values = list()
        for symbol_number in range(11):
            expected_value_5 = 0
            expected_value_6 = 0
            for i in range(13):
                expected_value_5 += probabilities_5[i] * q_to_value[i + symbol_number]
                expected_value_6 += probabilities_6[i] * q_to_value[i]
            values.append(expected_value_5-expected_value_6)
        return values

    def symbol_on_card_value_mean(self):
        q_to_value = collections.defaultdict(lambda: self.points[12])
        for i in range(len(self.points)):
            q_to_value[i] = self.points[i]

        probabilities_min_5 = np.array(self.probability_of_mean_min_cards[4])
        probabilities_min_5[12] = np.sum(probabilities_min_5[12:])
        probabilities_min_5 = probabilities_min_5[:13]

        probabilities_max_5 = np.array(self.probability_of_mean_max_cards[4])
        probabilities_max_5[12] = np.sum(probabilities_max_5[12:])
        probabilities_max_5 = probabilities_max_5[:13]

        probabilities_6 = np.array(self.probability_of_mean_min_cards[5])
        probabilities_6[12] = np.sum(probabilities_6[12:])
        probabilities_6 = probabilities_6[:13]

        values = list()
        for symbol_number in range(11):
            expected_value_5 = 0
            expected_value_6 = 0
            for i in range(13):
                expected_value_5 += (probabilities_min_5[i] * q_to_value[i + symbol_number] + probabilities_max_5[i] * q_to_value[i + symbol_number]) / 2
                expected_value_6 += probabilities_6[i] * q_to_value[i]
            values.append(expected_value_5 - expected_value_6)
        return values



    def __str__(self):
        return self.display

    def __repr__(self):
        return self.display

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

print(sum([symbol.weight for symbol in Symbols]))
assert sum([symbol.weight for symbol in Symbols]) == NUMBER_OF_SYMBOLS_IN_PLAY

def create_df() -> pd.DataFrame:
    plot_data = []
    for symbol in Symbols:
        best_6 = symbol.probability_of_collecting_best_cards[-1]
        best_6[12] = np.sum(best_6[12:])
        best_6 = best_6[:13]
        worse_6 = symbol.probability_of_collecting_worse_cards[-1]
        worse_6[12] = np.sum(worse_6[12:])
        worse_6 = worse_6[:13]
        mean_6 = symbol.probability_of_mean_max_cards[-1]
        mean_6[12] = np.sum(mean_6[12:])
        mean_6 = mean_6[:13]
        if symbol.value_symbol():
            for i in range(13):
                plot_data.append({"Symbol": symbol.display,
                                  "Value": symbol.points[i],
                                  "Quantity": i,
                                  "Color": symbol.color_hex,
                                  "ExactlyRandomProbability": symbol.exact_probability[i],
                                  "ExactlyFocusMaxProbability": best_6[i],
                                  "ExactlyFocusMinProbability": worse_6[i],
                                  "ExactlyFocusMeanProbability": mean_6[i]
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
    df = original_df.sort_values(by=['Symbol', 'Quantity'], ascending=[True, True])
    fig = px.line(df,
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

def figure_and_expected_value(df: pd.DataFrame, color_map: Dict[str, str], column_postfix: str, *args, **kwargs):
    probability_cols = ['Exactly', 'AtLeastQuantity', 'AtMostQuantity', 'AtLeastValue', 'AtMostValue']
    column_name = probability_cols[0] + column_postfix

    fig = px.line(df,
                  x=column_name,
                  y='Value',
                  color='Symbol',
                  color_discrete_map=color_map,
                  markers="o",
                  labels={column_name: "Probability"},
                  hover_data=['Symbol', 'Value', column_name, "Quantity"],
                  **kwargs)

    buttons = []
    for prefix in probability_cols:
        col_name = prefix + column_postfix
        temp_df = df.sort_values(by=["Symbol", col_name if not col_name.startswith("Exactly") else 'Quantity'])

        x_data_list = []
        y_data_list = []
        customdata_list = []
        hovertemplate_list = []

        for s in df['Symbol'].unique():
            symbol_df = temp_df[temp_df['Symbol'] == s]
            x_data_list.append(symbol_df[col_name].tolist())
            y_data_list.append(symbol_df["Value"].tolist())
            customdata_list.append(symbol_df[['Quantity']].values)

            hovertemplate = (
                f'Symbol: {s}<br>'
                f'Value: %{{y}}<br>'
                f'{col_name}: %{{x:.2%}}<br>'
                f'Quantity: %{{customdata[0]}}<extra></extra>'
            )
            hovertemplate_list.append(hovertemplate)

        buttons.append(
            dict(
                method='update',
                label=col_name,
                args=[
                    {'x': x_data_list,
                     'y': y_data_list,
                     'customdata': customdata_list,
                     'hovertemplate': hovertemplate_list
                     },
                    {'xaxis.title.text': col_name,
                     'xaxis.tickformat': ',.0%'
                     }
                ]
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.01,
                y=1.15,
                showactive=True,
                active=0,
                buttons=buttons
            )
        ],
        xaxis=dict(
            range=[-.02, 1.02],
            tickformat=',.0%',
            dtick=0.1,
        ),
        yaxis=dict(dtick=10)

    )


    probability_df = df.pivot_table(index="Symbol", columns="Quantity", values=column_name)
    for column in probability_df.columns:
        probability_df[column] = probability_df[column].apply(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else x)
    probability_df["ExpectedValue"] = (df['Value'] * df[column_name]).groupby(df['Symbol']).sum().round(2).apply(lambda x: str(x))
    probability_df = probability_df.rename(columns={12: "12+"})
    return fig, probability_df

def generate_image_and_table(function, name: str, *args, **kwargs):
    output_dir = "../content/plotly_graphs"
    os.makedirs(output_dir, exist_ok=True)
    image, df = function(*args, **kwargs)
    image.write_html(os.path.join(output_dir, f"{name}.html"), full_html=False,
                                  include_plotlyjs="cdn")
    df.to_html(os.path.join(output_dir, f"{name}_table.html"), index_names=False)

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
        temp_df =  df.sort_values(by=["Symbol", col_name if not col_name.startswith("Exactly") else 'Quantity'])

        x_data_list = []
        y_data_list = []
        customdata_list = []
        hovertemplate_list = []

        for s in df['Symbol'].unique():
            symbol_df = temp_df[temp_df['Symbol'] == s]
            x_data_list.append(symbol_df[col_name].tolist())
            y_data_list.append(symbol_df["Value"].tolist())
            customdata_list.append(symbol_df[['Quantity']].values)

            hovertemplate = (
                f'Symbol: {s}<br>'
                f'Value: %{{y}}<br>'
                f'{col_name}: %{{x:.2%}}<br>'
                f'Quantity: %{{customdata[0]}}<extra></extra>'
            )
            hovertemplate_list.append(hovertemplate)

        buttons.append(
            dict(
                method='update',
                label=col_name,
                args=[
                    {'x': x_data_list,
                     'y': y_data_list,
                     'customdata': customdata_list,
                     'hovertemplate': hovertemplate_list
                     },
                    {'xaxis.title.text': col_name,
                     'xaxis.tickformat': ',.0%'
                     }
                ]
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.01,
                y=1.15,
                showactive=True,
                active=0,
                buttons=buttons
            )
        ],
        xaxis=dict(
            range=[-.02, 1.02],
            tickformat=',.0%',
            dtick=0.01
        ),
        yaxis=dict(dtick=1)

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

def symbol_card_expected_value():
    data = dict()
    for symbol in Symbols:
        expected_min = list()
        expected_max = list()
        expected_random = list()

        min_prob = np.array(symbol.probability_of_min_out_of_3_cards)
        max_prob = np.array(symbol.probability_of_max_out_of_3_cards)
        random_prob = np.array(symbol.probability_of_symbol_in_card)


        current_min = min_prob
        current_max = max_prob
        current_random = random_prob
        current_mean_min = min_prob
        current_mean_max = max_prob

        expected_min.append(expected_value(current_min))
        expected_max.append(expected_value(current_max))
        expected_random.append(expected_value(current_random))


        for card_no in range(5):
            current_min = np.convolve(current_min, min_prob)
            current_max = np.convolve(current_max, max_prob)
            current_random = np.convolve(current_random, random_prob)
            if card_no % 2 == 0:
                current_mean_min = np.convolve(current_mean_min, max_prob)
                current_mean_max = np.convolve(current_mean_max, min_prob)
            else:
                current_mean_min = np.convolve(current_mean_min, min_prob)
                current_mean_max = np.convolve(current_mean_max, max_prob)

            expected_min.append(expected_value(current_min))
            expected_max.append(expected_value(current_max))
            expected_random.append(expected_value(current_random))


        data[symbol] = dict(collections.ChainMap(*[{
          f"ExpectedMin_{i}": expected_min[i],
          f"ExpectedMax_{i}": expected_max[i],
          f"ExpectedMean_{i}": (expected_max[i] +  expected_min[i]) / 2,
          f"ExpectedRandom_{i}": expected_random[i]} for i in range(6)]))


    df = pd.DataFrame(data)
    print(df)

def expected_value(array: np.array, existing_value: int = 0) -> float:
    return (np.array(range(existing_value, array.shape[0]+existing_value)) * array).sum()

def symbol_value():
    df = pd.DataFrame(Symbols.CIRCLE.probability_of_symbol_in_n_card)
    mean_probabilities = pd.DataFrame(Symbols.CIRCLE.probability_of_symbol_in_n_card).mean().to_numpy()
    mean_probabilities[12] = np.sum(mean_probabilities[12:])
    mean_probabilities = mean_probabilities[:13]

    mean_probabilities

    probabilities = pd.DataFrame(Symbols.CIRCLE.probability_of_symbol_in_n_card).mean().to_list()


    default_expected_quantity = expected_value(pd.DataFrame(Symbols.CIRCLE.probability_of_symbol_in_n_card).mean().to_numpy())
    q_to_value = collections.defaultdict(lambda: Symbols.CIRCLE.points[12])
    for i in range(len(Symbols.CIRCLE.points)):
        q_to_value[i] = Symbols.CIRCLE.points[i]
    exp_1 = 0
    exp_2 = 0
    for i in range(len(probabilities)):
        exp_1 += probabilities[i] * q_to_value[i+6]
    for i in range(13):
        exp_2 += mean_probabilities[i] * q_to_value[i+6]



    print(df)

if __name__ == "__main__":
    symbol_value()
    symbol_card_expected_value()
    df = create_df()
    color_map = {symbol.display: symbol.color_hex for symbol in Symbols}

    generate_melted_figure(df)
    generate_composite_figure(df)


    generate_image_and_table(generate_symbol_point_graph_and_df, "symbol_point", df, color_map)
    generate_image_and_table(figure_and_expected_value, "symbol_point_random", df, color_map, "RandomProbability", title="Randomly collecting symbols")
    generate_image_and_table(figure_and_expected_value, "symbol_max_prob_point", df, color_map, "FocusMaxProbability",
                             title="Collecting most symbols out of 3 cards")
    generate_image_and_table(figure_and_expected_value, "symbol_min_prob_point", df, color_map, "FocusMinProbability",
                            title="Collecting least symbols out of 3 cards")
    generate_image_and_table(figure_and_expected_value, "symbol_mean_prob_point", df, color_map, "FocusMeanProbability",
                             title="Half time collecting the most, half time the least symbols out of three cards")
