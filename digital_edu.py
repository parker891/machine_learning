import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv('games.csv')

print(df.winner.value_counts())
print('White appears to win', round(10001 / 20058, 2) * 100,'% of the games')
print('while Black wins', round(9107 / 20058, 2) * 100,'% of the games')
print('and', round(950 / 20058, 2) * 100, '% of the games are a draw')

open_df = df.groupby(by='opening_name').winner.value_counts()
open_df = open_df.reset_index(name='wins')
open_df = open_df.sort_values(by='wins', ascending=False)

black_wins = open_df[open_df['winner'] == 'black']
white_wins = open_df[open_df['winner'] == 'white']
black_winner = list(black_wins.head().opening_name)
white_winner = list(white_wins.head().opening_name)
winner = black_winner + white_winner
dataframes = []
for x in winner:
    temp = open_df[open_df['opening_name'] == x]
    temp['sum'] = temp.wins.sum().astype(int)
    temp['percentage'] = temp['wins'] / temp['sum']
    dataframes.append(temp)
win_prob = dataframes[0]
for x in dataframes[1:]:
    win_prob = pd.concat([win_prob, x])

x = win_prob[win_prob['winner'] == 'black'].opening_name
y = win_prob[win_prob['winner'] == 'black'].percentage
plt.figure(dpi=100)
plt.bar(x, height=y, edgecolor='black')
plt.xticks(rotation='vertical')
plt.title('Win percentage by opening for black')
plt.xlabel('Opening')
plt.ylabel('Percentage(Out of 1)')

#plt.show()

resolve = pd.DataFrame(df.victory_status.value_counts())
resolve = resolve.reset_index()
plt.figure(dpi=100)
plt.bar(x=resolve['index'], height=resolve.victory_status, edgecolor='black')

#plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

cols = ['white_rating', 'black_rating']
X = df[cols]
y = df['winner']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)


test= np.array([1463, 1500])
test = test.reshape(1, -1)
print(lr.predict(test))
print(max(lr.predict_proba(test)[0]))
print((lr.predict_proba(test)[0]))