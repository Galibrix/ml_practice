import random
from typing import List, Tuple, TypeVar, NamedTuple, Dict
from collections import Counter, defaultdict
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


def split_data(data: list[X], prob: float) -> Tuple[list[X], list[X]]:
    random.shuffle(data)
    prob = int(len(data) * prob)
    return data[:prob], data[prob:]

# print(split_data(males, 0.7))

def majority_vote(genders: list[str]) -> str:
    vote_counts = Counter(genders)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        majority_vote(genders[:-1])

class LabeledPoint(NamedTuple):
    gender: str
    point: Vector


def knn_classifier(k: int, labeled_points: list[LabeledPoint], new_point: Vector) -> str:
    by_distance = sorted(labeled_points, key=lambda lp: distance(lp.point, new_point))
    k_nearest_genders = [lp.gender for lp in by_distance[:k]]
    return majority_vote(k_nearest_genders)

# print(knn_classifier(5, data, [150, 6]))

points_by_gender: Dict[str, list[Vector]] = defaultdict(list)

labeled_points = [LabeledPoint(gender, [income, exp])
                  for income, exp, gender in zip(data['Annual Income ($)'],
                                                 data['Work Experience'],
                                                 data['Gender'])]

for customer in labeled_points:
    points_by_gender[customer.gender].append(customer.point)
print(points_by_gender)

metrics = ['Work experience', 'Annual Income ($)']
marks = ['x', '+']

fig, ax = plt.subplots(2, 2)

for i in range(len(metrics)):
    for j in range(len(metrics)):

        if i != j:
            ax[i][j].set_title(f'{metrics[i]} to {metrics[j]}', fontsize=8)
            ax[i][j].set_yticks([])
            ax[i][j].set_xticks([])
            for mark, (gender, points) in zip(marks, points_by_gender.items()):
                xs = [point[i] for point in points]
                ys = [point[j] for point in points]
                ax[i][j].scatter(xs, ys, marker=mark, label=gender, alpha=0.4)

        else:
            ax[i][j].annotate('Series' + str(j), (0.5, 0.5), xycoords='axes fraction', va='center', ha='center')

        if i < len(metrics)-1: ax[i][j].xaxis.set_visible(False)
        if j > 0: ax[i][j].yaxis.set_visible(False)

ax[0][len(metrics)-1].legend(loc='upper right')
plt.show()


confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
num_correct = 0

gender_train, gender_test = split_data(labeled_points, 0.7)

for customer in gender_test:
    predicted = knn_classifier(5, gender_train, customer.point)
    actual = customer.gender

    if predicted == actual:
        num_correct += 1

    confusion_matrix[(predicted, actual)] += 1
pct_correct = num_correct / len(gender_test)
print(pct_correct, confusion_matrix)

print(knn_classifier(5, labeled_points, [180, 0]))