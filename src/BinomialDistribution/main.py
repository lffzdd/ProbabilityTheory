import math
import matplotlib.pyplot as plt

p = 0.5
n = 1000

k_values = list(range(n + 1))
probs = [math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in k_values]

plt.bar(k_values, probs)
# for k, prob in zip(k_values, probs):
#     plt.text(k, prob, f"{prob:.3f}", ha="center", va="bottom", fontsize=8)
plt.xlabel("k")
plt.ylabel("P(X=k)")
plt.title(f"Binomial(n={n}, p={p})")
plt.show()
