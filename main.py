import matplotlib.pyplot as plt
import math
import pandas as pd

data = pd.read_csv('Customers.csv')
# print(data.head())
# print(data.columns)

data['Annual Income ($)'] = data['Annual Income ($)'].map(lambda x: x/1000)

males = data.loc[data['Gender'] == 'Male'].loc[:, ['Work Experience', 'Annual Income ($)', 'Gender']]
females = data.loc[data['Gender'] == 'Female'].loc[:, ['Work Experience', 'Annual Income ($)', 'Gender']]
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

plt.scatter(females['Work Experience'], females['Annual Income ($)'], label='Female')
plt.scatter(males['Work Experience'], males['Annual Income ($)'], label='Male')
plt.show()

