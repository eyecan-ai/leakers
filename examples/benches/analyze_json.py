import json
import numpy as np
import pandas as pd
import rich


# Load data from BENCH
with open("L_8_3.json") as f:
    my_list = [json.loads(line) for line in f]
df = pd.DataFrame(my_list)

# GROUP BY Radius -> Zenith
grouped = df.groupby(["radius", "zenith"]).sum()

# Collect separated datapoints [radius, zenith, recall]
datapoints = []
for name, group in grouped.iterrows():
    radius, zenith = name

    tp = group["tp"]
    p = group["p"]
    recall = tp / p
    datapoints.append(
        {
            "radius": radius,
            "zenith": zenith,
            "recall": recall,
        }
    )
rich.print(datapoints)

# Plot Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(datapoints)
df = df.pivot("radius", "zenith", "recall")
sns.heatmap(df, annot=False, square=True)
plt.show()
