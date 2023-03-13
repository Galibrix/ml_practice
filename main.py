from typing import List, Tuple, TypeVar
from collections import Counter
import matplotlib.pyplot as plt
import math
import pandas as pd

Vector: list[float] = []
X = TypeVar('X')

data = pd.read_csv('Customers.csv')
# print(data.head())
# print(data.columns)

data['Annual Income ($)'] = data['Annual Income ($)'].map(lambda x: x / 1000)

males = data.loc[data['Gender'] == 'Male'].loc[:, ['Work Experience', 'Annual Income ($)', 'Gender']]
females = data.loc[data['Gender'] == 'Female'].loc[:, ['Work Experience', 'Annual Income ($)', 'Gender']]
print(len(males), len(females))
print(males, females)

under_3_years = data.loc[(data['Work Experience'] <= 3)]
under_6_years = data.loc[(data['Work Experience'] > 3) & (data['Work Experience'] <= 6)]
upper_6_years = data.loc[(data['Work Experience'] > 6)]
print(f"Mean salary with exp under 3 years - {under_3_years['Annual Income ($)'].mean()},"
      f" customer counts {len(under_3_years)}")
print(f"Mean salary with exp under 6 years - {under_6_years['Annual Income ($)'].mean()},"
      f" customer counts {len(under_6_years)}")
print(f"Mean salary with exp upper 6 years - {upper_6_years['Annual Income ($)'].mean()},"
      f" customer counts {len(upper_6_years)}")

plt.scatter(females['Work Experience'], females['Annual Income ($)'], label='Female', alpha=0.3)
plt.scatter(males['Work Experience'], males['Annual Income ($)'], label='Male', alpha=0.3)
plt.xlabel('Work Experience')
plt.ylabel('Income in thousands')
plt.legend(loc=9)
plt.show()

def distance(v: Vector, w: Vector) -> float:
    return math.sqrt(sum((v_i - w_i) ** 2 for v_i, w_i in zip(v, w)))


def split_data(df: pd.DataFrame, prob: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df.sample(frac=1)
    prob = int(len(df) * prob)
    return df.loc[:prob], data.loc[prob:]

# print(split_data(males, 0.7))

def majority_vote(genders: list[str]) -> str:
    vote_counts = Counter(genders)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        majority_vote(genders[:-1])

