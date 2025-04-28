import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytaj wyniki
results_df = pd.read_csv('augmentation_results.csv', index_col=0)

# Ustawienia wykresu
plt.figure(figsize=(12, 8))
sns.heatmap(results_df, annot=True, fmt=".4f", cmap="viridis", cbar=True, linewidths=0.5)

plt.title('Heatmapa wyników metryk dla różnych augmentacji', fontsize=16)
plt.xlabel('Metryki', fontsize=14)
plt.ylabel('Typ augmentacji', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Zapisz wykres do pliku
plt.savefig('augmentation_results_heatmap.png')

plt.show()
