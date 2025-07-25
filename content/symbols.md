Title: Symbols
Date: 2025-07-24
Circle: flamingo
Star: koala

# Calculating values of symbols
The goal is to design different, exciting type of symbols that the players try to collect or avoid. 

At the end of the game all the visible symbols earn point based on the following table:
{% include 'plotly_graphs/symbol_point_table.html' %}
{% include 'plotly_graphs/symbol_point.html' %}

The weight column describes the probability that the symbol is chosen.

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

The probability that a player has at least $k$ number of symbols is $C(k) = P(X \ge k) = \sum_{i=k}^{12} P(k)$.

The following graph shows the probability and the gained point by randomly selecting symbols.
{% include 'plotly_graphs/symbol_point_random.html' %}

$X$ symbol is better than $Y$ symbol if it's above and to the right in the graph, meaning that it's more probable to gain the same amount of point from $X$, or with the same probability $X$ can worth more.
There is no symbol which is better than other, for example in higher probabilities the {{ STAR }} will worth more point, but with luck {{ CIRCLE }} can worth more point. {{ CIRCLE }} is always more probable to get the same points as {{ SQUARE }}, but {{ SQUARE }} can gain more points as well.

## Focused collection
During the game, the players are not randomly receiving cards, but can choose 1 out of 3. Also when placing the cards they can focus on the symbols to not hide them and that arrows are pointing to them.
To simulate the probability of collecting symbols when focusing on it, we need to calculate the probability that a chosen card will have $k$ number of symbols. Choosing 6 cards is independent of each other, so their probability can be summarized. It's hard to quarters are hidden and where the arrows are pointing, but as each one affects the same number of symbols, for simplicity's sake we ignore both. This means we underestimate the probabilities of high symbol values a bit.

Similarly to the previous random scenario, the probability that we have exactly $k$ number of $x$ symbol in one card - which has total $N_c = 16$ symbol - is:
$P(k) = {N_c\choose k} \cdot p^k \cdot q^{N_c-k}$

The probability that a card has **at most** $k$ number of $x$ symbol is $D(k) = P(X \le k) = \sum_{i=0}^{k} P(k) $.
The probability that the chosen card out of the 3 cards has **at most** $k$ number of $x$ symbols is: $P_{D}(k) = D(k)^3$, as it means that there is no card that has $\gt k$ symbols, and as independent probabilities we can the cumulative probability is the product of the separate probabilities.
The probability that the chosen card out of the 3 cards has **exactly** $k$ number of $x$ symbols are:
 - for $k=0$:  $P_{D}(0) = D(0)^3$, as at most $0$ means exactly $0$
 - for $k=1,2 \dots 12$: $P_{D}(k) = D(k)^3 - D(k-1)^3$

$P_{D}(k)$ gives us the probability that card with the most symbol out of 3 card has exactly $k$ number of $x$ symbols.

The probability that $2$ card has total $j$ number of symbols is $Q_2(j) = \sum_{k=0}^j P(k) \cdot P(j-k)$

The probability that $N$ card has total $j$ number of symbols is $Q_N(j) = \sum_{k=0}^j Q_{N-1}(k) \cdot P(j-k)$.

As before, a player can have at most $k=12$ symbols, so $Q_6(12) = 1 - \sum_{k=0}^{11} Q_6(k)$.

{% include 'plotly_graphs/symbol_max_prob_point.html' %}
