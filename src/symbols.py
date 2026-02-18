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
    CIRCLE = ("anchor", 7,       [0,3,7,12,18,25,33,42,52,63,75,88,102], "#F3A2BD", {
    "mixed_sim_5": [0.0034999999999999996,0.018000000000000002,0.0513,0.0938,0.13935,0.16620000000000001,0.1595,0.14340000000000003,0.09709999999999999,0.0654,0.03635,0.017849999999999998,0.00825],
    "mixed_sim_6": [0.0014,0.0084,0.0238,0.0543,0.085,0.1287,0.1537,0.1471,0.1294,0.1093,0.0688,0.0447,0.045399999999999996]})
    SQUARE = ("map", 5,    [0,1,4,9,16,25,36,49,64,81,100,121,144], "#EB7B36", {
        "mixed_sim_5": [0.019,0.06925,0.138,0.1832,0.1917,0.15485,0.111,0.06785,0.0387,0.016300000000000002,0.00655,0.0029999999999999996,0.0006000000000000001],
    "mixed_sim_6": [0.0084,0.0436,0.093,0.1471,0.1686,0.1649,0.1407,0.1021,0.0618,0.0375,0.0191,0.0092,0.004]})
    TRIANGLE = ("rum", 4,        [0,-20,-10,0,20,39,57,74,90,105,119,132,144], "#F7DCB4", {
        "mixed_sim_5": [0.048,0.1011,0.17709999999999998,0.20825,0.1992,0.1381,0.07655,0.033100000000000004,0.01315,0.00395,0.00125,0,0.0001],
    "mixed_sim_6": [0.0396,0.0666,0.1324,0.1716,0.1959,0.1729,0.1183,0.061,0.0267,0.0106,0.0032,0.0012,0.0]})
    STAR = ("spyglass", 6,      [0,10,19,27,34,40,45,50,56,63,71,80,90], "#C2CCDA", {
        "mixed_sim_5": [0.0049,0.03055,0.08410000000000001,0.1419,0.1882,0.1809,0.14495,0.09945000000000001,0.0624,0.0342,0.01815,0.006999999999999999,0.0033],
    "mixed_sim_6": [0.0022,0.0145,0.039,0.0864,0.1428,0.1705,0.1704,0.1398,0.0965,0.0674,0.0363,0.0178,0.0164]})
    X = ("rat", 8,             [0,-1,-3,-6,-10,-15,-21,-28,-36,-45,-55, -66, -78], "#739C41", {
        "mixed_sim_5": [0,0.00435,0.0188,0.04195,0.09154999999999999,0.14275000000000002,0.16685,0.1641,0.14215,0.0973,0.0606,0.032299999999999995,0.036849999999999994],
    "mixed_sim_6": [0.0002,0.0005,0.0041,0.0166,0.0399,0.076,0.1186,0.1504,0.1577,0.1507,0.1115,0.0668,0.107]})
    MOON = ("shark", 6,           [0,-10,-12,-14,-17,-20,-24,-28,-33,-38,-43,-48,-54], "#4E598B", {
        "mixed_sim_5": [0.0054,0.026250000000000002,0.07444999999999999,0.1377,0.1778,0.18645,0.15375,0.11325,0.06565,0.032600000000000004,0.0162,0.00655,0.00395],
    "mixed_sim_6": [0.0021,0.0097,0.0339,0.0811,0.1291,0.1623,0.1693,0.15,0.1081,0.0744,0.041,0.0204,0.0186]})
    DIAMOND = ("parrot", 2,       [0,25,0,50,0,75,0,100,0,125,0,150,0], "#8D79B6", {
        "mixed_sim_5": [0.1786,0.36655,0.197,0.1945,0.031450000000000006,0.02775,0.00215,0.0019,0,0,0,0,0.0],
    "mixed_sim_6": [0.1047,0.3776,0.1404,0.2918,0.0249,0.0532,0.0021,0.0049,0.0001,0.0003,0,0,0.0]})
    SKULL = ("kraken", 3,       [0,-36,-27,-19,-12,-6,-1,0,0,0,0,0,0], "#0F1421", {
        "mixed_sim_5": [0.11425,0.16365000000000002,0.2496,0.2281,0.1381,0.07095000000000001,0.025849999999999998,0.00735,0.0017499999999999998,0.00035,0,0,0.0],
    "mixed_sim_6": [0.0927,0.1114,0.2049,0.2242,0.1769,0.1091,0.0533,0.02,0.0059,0.0015,0.0001,0,0.0]})
    SUN = ("coin", 5,           [0,2,8,18,32,50,72,50,32,18,8,2,0], "#88CDF3", {
        "mixed_sim_5": [0.01255,0.04635,0.10285,0.16904999999999998,0.23099999999999998,0.22920000000000001,0.13615,0.0444,0.01725,0.006999999999999999,0.0022,0.00135,0.0006500000000000001],
    "mixed_sim_6": [0.0065,0.0247,0.0613,0.1044,0.163,0.2122,0.2664,0.0946,0.0377,0.0158,0.0065,0.004,0.0029]})
    ARROW_LEFT = ("arrow_left", 1.5, [0]*13)
    ARROW_RIGHT = ("arrow_right", 1.5, [0]*13)
    ARROW_UP = ("arrow_up", 1.5,  [0]*13)
    ARROW_DOWN = ("arrow_down", 1.5, [0]*13)
    NOTHING = ("", 44, [0]*13)

    def __init__(self, display: str, weigh: int, points: List, color_hex: str = "white", *args):
        self.args = args
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

        if self.args:
            self.expected_value_sim_6 = self.expected_value_of_dist(np.array(self.args[0]["mixed_sim_6"]))

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
        mean_prob_5 = np.convolve(np.array(self.args[0]["mixed_sim_5"]), prob)
        e_5 = self.expected_value_of_dist(mean_prob_5)
        e_6 =  self.expected_value_sim_6
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
                                  "ExactlyFocusMeanProbability": mean_6[i],
                                  "ExactlyMixedSim": symbol.args[0]["mixed_sim_6"][i],

                                  })

    df = pd.DataFrame(plot_data)
    for column_post_fix in ["RandomProbability", "FocusMaxProbability", "FocusMinProbability", "FocusMeanProbability", "MixedSim"]:
        df = df.sort_values(by=['Symbol', 'Value'], ascending=[True, False])
        df[f"AtLeastValue{column_post_fix}"] = df.groupby("Symbol")[f"Exactly{column_post_fix}"].cumsum()

        df = df.sort_values(by=['Symbol', 'Value'], ascending=[True, True])
        df[f"AtMostValue{column_post_fix}"] = df.groupby("Symbol")[f"Exactly{column_post_fix}"].cumsum()

        df = df.sort_values(by=['Symbol', 'Quantity'], ascending=[True, False])
        df[f"AtLeastQuantity{column_post_fix}"] = df.groupby("Symbol")[f"Exactly{column_post_fix}"].cumsum()

        df = df.sort_values(by=['Symbol', 'Quantity'], ascending=[True, True])
        df[f"AtMostQuantity{column_post_fix}"] = df.groupby("Symbol")[f"Exactly{column_post_fix}"].cumsum()

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
    probability_cols = []
    prefixes = ['Exactly', 'AtLeastValue', 'AtMostValue', 'AtLeastQuantity', 'AtMostQuantity']
    postfixes = ["RandomProbability", "FocusMaxProbability", "FocusMinProbability", "FocusMeanProbability", "BestSim",
                 "WorstSim", "MixedSim"]
    for prefix in prefixes:
        for postfix in postfixes:
            probability_cols.append(f"{prefix}{postfix}")
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
    probability_cols = []

    prefixes = ['AtLeastValue', 'AtMostValue', 'AtLeastQuantity', 'AtMostQuantity', 'Exactly']
    postfixes = ["RandomProbability", "FocusMaxProbability", "FocusMinProbability", "FocusMeanProbability", "BestSim", "WorstSim", "MixedSim"]
    for prefix in prefixes:
        for postfix in postfixes:
            probability_cols.append(f"{prefix}{postfix}")

    # Melt the DataFrame
    df_melted = df.melt(
        id_vars=['Symbol', 'Value', 'Quantity', 'Color'],
        value_vars=probability_cols,
        var_name='Probability_Type',
        value_name='Probability_Value'
    )

    df_melted = df_melted.sort_values(["Symbol", "Value", "Quantity"], ascending=[True, True, False])

    prefix_column = df_melted["Probability_Type"]
    postfix_column = df_melted["Probability_Type"]

    for prefix in prefixes:
        postfix_column = postfix_column.str.removeprefix(prefix)

    for postfix in postfixes:
        prefix_column = prefix_column.str.removesuffix(postfix)
    df_melted["prefix"] = prefix_column
    df_melted["suffix"] = postfix_column

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

    sandbox = df[["Symbol", "Quantity", "ExactlyRandomProbability", "ExactlyFocusMeanProbability", "ExactlyBestSim", "ExactlyWorstSim", "ExactlyMixedSim"]]
    generate_melted_figure(df)
    generate_composite_figure(df)


    # generate_image_and_table(generate_symbol_point_graph_and_df, "symbol_point", df, color_map)
    # generate_image_and_table(figure_and_expected_value, "symbol_point_random", df, color_map, "RandomProbability", title="Randomly collecting symbols")
    # generate_image_and_table(figure_and_expected_value, "symbol_max_prob_point", df, color_map, "FocusMaxProbability",
    #                          title="Collecting most symbols out of 3 cards")
    # generate_image_and_table(figure_and_expected_value, "symbol_min_prob_point", df, color_map, "FocusMinProbability",
    #                         title="Collecting least symbols out of 3 cards")
    # generate_image_and_table(figure_and_expected_value, "symbol_mean_prob_point", df, color_map, "FocusMeanProbability",
    #                          title="Half time collecting the most, half time the least symbols out of three cards")
