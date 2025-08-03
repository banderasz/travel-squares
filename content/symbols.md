Title: Symbols
Date: 2025-07-24
Circle: flamingo
Star: koala

# Calculating values of symbols
The goal is to design different, exciting type of symbols that the players try to collect or avoid. 

At the end of the game all the visible symbols earn point based on the following table:
{% include 'plotly_graphs/symbol_point_table.html' %}
{% include 'plotly_graphs/symbol_point.html' %}

The weight column describes the probability that the symbol is chosen. The table defines the $V(k)$ value of $k$ number of $x$ symbol.

The symbol values are chosen to support multiple winning strategies, there are no symbols that are always better to collect than others (with the obvious exception of negative symbols).
To see how good strategy is to collect a symbol we also need to calculate with the rarity of that symbol.

## Random collection
A card contains exactly 4 quarters, and each quarter contains at most 4 symbols, as one of the possible symbols is no symbol.
During the game the players collect 6 cards, but when placing a card they need to hide at least 1 quarter of a card, except for one card. 
This means that the maximum number visible quarters are $5 \cdot 3+4=19$, so the maximum number of symbols is $19 \cdot 4=76$.
However, the arrow symbols are copying the symbols in the quarter they are pointing to. The symbols $6.25%$ are arrow, approximately $5$, which are each copying at most 4 symbols.
Together with the arrows, a player can have maximum approximately $96$ symbols. This is the sum of the weight, so the weight gives what is the expected value of symbols a player has at the end of the game is chosen randomly.

We can calculate what is the probability to have exactly $k$ number of a symbol, which let us compare how hard is to gain points by collecting a symbol.

The total number of symbols ("nothing included"): $N=96$

The weight of $x$ symbol: $w_x$

The probability that one symbol is $x$:  $p=\frac{w_x}{N}$

The probability that one symbol is not $x$ : $q=1-p$

The probability that we have exactly $k$ number of $x$ symbol is:
$$P(k) = {N\choose k} \cdot p^k \cdot q^{N-k}$$

As the players can collect at most $12$ from a symbol, $P(12) = 1 - \sum_{k=0}^{11} P(k).$

### Value of symbols
There are multiple probability that are worth calculating:

 - **Exactly**: The probability of collecting $k$ number of symbols that results in $v$ point.
 - **At Least Quantity**: The probability of collecting at least $k$ number of symbols. $C(k) = P(X \ge k) = \sum_{i=k}^{12} P(k)$
 - **At Most Quantity**: The probability of collecting at most $k$ number of symbols. $D(k) = P(X \le k) = \sum_{i=0}^{k} P(k)$
 - **At Least Value**: The probability of collecting symbols worth at least $v=V(k)$ value. $E(k) = P(V(k) \ge v) = \sum_{k|V(k) \ge v} P(k)$
 - **At Most Value**: The probability of collecting symbols worth at most $v=V(k)$ value. $F(k) = P(V(k) \le v) = \sum_{k|V(k) \le v} P(k)$

For positive monotonic symbols - where more symbols always worth more point - the **~Quantity** and the **~Value** is the same.
For negative monotonic symbols - where more symbols always worth less point - the **At Least Quantity** and the **At Most Value** is the same, and vice versa.
For the non-monotonic symbols depending on the strategy how the symbols are collected any of these metrics can be useful.

The following graph shows the different probabilities and the gained point by randomly collecting symbols.
{% include 'plotly_graphs/symbol_point_random.html' %}
{% include 'plotly_graphs/symbol_point_random_table.html' %}

$X$ symbol is better than $Y$ symbol if it's above and to the right in the graph, meaning that it's more probable to gain the same amount of point from $X$, or with the same probability $X$ can worth more.
There is no symbol which is better than other, for example in higher probabilities the {{ STAR }} will worth more point, but with luck {{ CIRCLE }} can worth more point. {{ CIRCLE }} is always more probable to get the same points as {{ SQUARE }}, but {{ SQUARE }} can gain more points as well.

## Focused collection
During the game, the players are not randomly receiving cards, but can choose 1 out of 3. Also when placing the cards they can focus on the symbols to not hide them and that arrows are pointing to them.
To simulate the probability of collecting symbols when focusing on it, we need to calculate the probability that a chosen card will have $k$ number of symbols. Choosing 6 cards is independent of each other, so their probability can be summarized. It's hard to quarters are hidden and where the arrows are pointing, but as each one affects the same number of symbols, for simplicity's sake we ignore both. This means we underestimate the probabilities of high symbol values a bit.

Similarly to the previous random scenario, the probability that we have exactly $k$ number of $x$ symbol in one card - which has total $N_c = 16$ symbol - is:
$P(k) = {N_c\choose k} \cdot p^k \cdot q^{N_c-k}$

The probability that a card has **at most** $k$ number of $x$ symbol is $D(k) = P(X \le k) = \sum_{i=0}^{k} P(k)$.

The probability that the chosen card out of the 3 cards has **at most** $k$ number of $x$ symbols is: $P_{D}(k) = D(k)^3$, as it means that there is no card that has $\gt k$ symbols, and as independent probabilities we can the cumulative probability is the product of the separate probabilities.

The probability that the chosen card out of the 3 cards has **exactly** $k$ number of $x$ symbols are:

 - for $k=0$:  $P_{D}(0) = D(0)^3$, as at most $0$ means exactly $0$
 - for $k=1,2 \dots 12$: $P_{D}(k) = D(k)^3 - D(k-1)^3$

$P_{D}(k)$ gives us the probability that card with the most symbol out of 3 card has exactly $k$ number of $x$ symbols.

The probability that $2$ card has total $j$ number of symbols is $Q_2(j) = \sum_{k=0}^j P_D(k) \cdot P_D(j-k)$

The probability that $N$ card has total $j$ number of symbols is $Q_N(j) = \sum_{k=0}^j Q_{N-1}(k) \cdot P_D(j-k)$.

As before, a player can have at most $k=12$ symbols, so $Q_6(12) = 1 - \sum_{k=0}^{11} Q_6(k)$.

The following graph shows the different $Q_6$ probabilities:

{% include 'plotly_graphs/symbol_max_prob_point.html' %}
{% include 'plotly_graphs/symbol_max_prob_point_table.html' %}

For the monotone positive symbols ({{ CIRCLE }}, {{ SQUARE }}, {{ STAR }}, {{ TRIANGLE }}) the most interesting is the **at least** diagrams. 1 {{ TRIANGLE }} symbol actually worth less point as 0, but while trying to collect them having no {{ TRIANGLE }} has minimal probability.

## Avoiding collection
A useful metric to calculate the probability if a player tries to avoid collecting a symbol for the negative symbols, and also to verify how hard is it to get accidentally point.

The probability that a card has **at least** $k$ number of $x$ symbol is $C(k) = P(X \ge k) = \sum_{i=k}^{12} P(k)$.

The probability that the chosen card out of the 3 cards has **at least** $k$ number of $x$ symbols is: $P_{C}(k) = C(k)^3$, as it means that there is no card that has $\lt k$ symbols, and as independent probabilities we can the cumulative probability is the product of the separate probabilities.

The probability that the chosen card out of the 3 cards has **exactly** $k$ number of $x$ symbols are:

 - for $k=12$:  $P_{C}(12) = C(12)^3$, as at least $12$ means exactly $12$
 - for $k=11,10 \dots 0$: $P_{C}(k) = C(k)^3 - C(k+1)^3$

$P_{C}(k)$ gives us the probability that card with the least symbol out of 3 card has exactly $k$ number of $x$ symbols.

The probability that $2$ card has total $j$ number of symbols is $R_2(j) = \sum_{k=0}^j P_C(k) \cdot P_C(j-k)$

The probability that $N$ card has total $j$ number of symbols is $R_N(j) = \sum_{k=0}^j R_{N-1}(k) \cdot P_C(j-k)$.

The following graph shows the different $R_6$ probabilities:

{% include 'plotly_graphs/symbol_min_prob_point.html' %}
{% include 'plotly_graphs/symbol_min_prob_point_table.html' %}

For symbols where 1 symbol has a big difference ( {{ SKULL }}, {{ DIAMOND }}) most interesting is the **at least quantity** diagram. It shows that there is $13.24%$ chance to collect at least 1 {{ DIAMOND }} by "accident", and even while trying to avoid it there is $32.4%$ chance to collect at least 1 {{ SKULL }}.

## Mixed collection
Usually the player tries to collect the good symbols and avoid the bad ones, while it's opponent does the opposite for them. For most of the symbols it's a realistic simulation to assume for 1 turn the symbols are collected, in the other they are avoided.

By starting with the player's "collection phase" (for even number of cards the order doesn't matter):

The probability that $2$ card has total $j$ number of symbols is $S_2(j) = \sum_{k=0}^j P_D(k) \cdot P_C(j-k)$
The probability that $3$ card has total $j$ number of symbols is $S_3(j) = \sum_{k=0}^j S_2(k) \cdot P_D(j-k)$

The probability that $N$ card has total $j$ number of symbols is:
 
 - if j is odd: $S_N(j) = \sum_{k=0}^j S_{N-1}(k) \cdot P_D(j-k)$.
 - if j is even: $S_N(j) = \sum_{k=0}^j S_{N-1}(k) \cdot P_C(j-k)$.

The following graph shows the different $R_6$ probabilities:

{% include 'plotly_graphs/symbol_mean_prob_point.html' %}
{% include 'plotly_graphs/symbol_mean_prob_point_table.html' %}

## 

## Symbol comparison

- {{ CIRCLE }} vs {{ STAR }}: >48 {{ CIRCLE }} is better
  - RandomProbability: $<47%$ 
  - MaxProbability: $<99%$
  - MinProbability: $<0.2%$
  - MeanProbability: $<51%$
- {{ CIRCLE }} vs {{ SQUARE }}: >102 (max {{ SQUARE }}) {{ SQUARE }} is better.
  - RandomProbability: $<2.6%$ 
  - MaxProbability: $<46%$
  - MinProbability: $<0%$
  - MeanProbability: $<1%$
- {{ SQUARE }} vs {{ TRIANGLE }}:
  - RandomProbability: $43% (34) - 2.8% (100)$ {{ TRIANGLE }} is better
  - MaxProbability: $96.7% (38) - 56% (91)$ {{ TRIANGLE }} is better
  - MinProbability: $38% (4) - 1.8% (20)$ {{ TRIANGLE }} is better
  - MeanProbability: $55% (31) - 1% (103) $  {{ TRIANGLE }} is better

- RandomProbability:
  - $100% - 47%$: {{ STAR }} - {{ CIRCLE }} - {{ SQUARE }} - {{ TRIANGLE }}
  - $47% - 44%$: {{ CIRCLE }} - {{ STAR }} - {{ SQUARE }} - {{ TRIANGLE }}
  - $44% - 19.5%$: {{ CIRCLE }} - {{ STAR }} - {{ TRIANGLE }} - {{ SQUARE }}
  - $19.5% - 12%$: {{ CIRCLE }} - {{ TRIANGLE }} - {{ STAR }} -  {{ SQUARE }}
  - $12% - 2.8%$: {{ CIRCLE }} - {{ TRIANGLE }} - {{ SQUARE }} - {{ STAR }}
  - $12% - 2.8%$: {{ CIRCLE }} - {{ TRIANGLE }} - {{ SQUARE }} - {{ STAR }}
