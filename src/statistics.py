import pandas as pd
import plotly.express as px

from src.symbols import create_df, Symbols

color_map = {symbol.display: symbol.color_hex for symbol in Symbols}

# df = create_df()



good_sim = pd.read_csv("simulate_5_10000_simulate_5_turn_mixed_game_less_evil.csv")
good_sim = good_sim / good_sim.sum()
good_sim.loc[12] = good_sim.loc[12:].sum()

bad_sim = pd.read_csv("simulate_5_10000_simulate_5_turn_mixed_game_more_evil.csv")
bad_sim = bad_sim / bad_sim.sum()
bad_sim.loc[12] = bad_sim.loc[12:].sum()

good_sim_column = good_sim.stack()
good_sim_column.name = "GoodSimProbability"
good_sim_table = good_sim_column.reset_index().rename(columns={'level_0': 'Quantity', 'level_1': 'Symbol'})
# df = pd.merge(df, good_sim_table)

probability_cols = ['ExactlyRandomProbability', 'ExactlyFocusMaxProbability', 'ExactlyFocusMinProbability', 'ExactlyFocusMeanProbability',
                    "GoodSimProbability"]

df_melted = df.melt(
        id_vars=['Symbol', 'Value', 'Quantity', 'Color'],
        value_vars=probability_cols,
        var_name='Probability_Type',
        value_name='Probability_Value'
    )

fig = px.line(df_melted,
              x='Probability_Value',
              y='Value',
              color='Symbol',
              color_discrete_map=color_map,
              line_dash='Probability_Type',
              symbol='Probability_Type',
              title='Value vs. Different Probabilities per Symbol',
              hover_data=['Symbol', 'Value', 'Probability_Value', "Quantity"]
              )
fig.update_traces(marker={'size': 10})
fig.show()