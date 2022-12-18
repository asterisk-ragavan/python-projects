import pandas as pd
from googletrans import Translator
data = pd.read_csv("hindi.csv")

print(data)
translator = Translator()
translations = []
for column in data.columns:
    unique = data[column].unique()
    translations.append(translator.translate(unique, src='auto', dest='en'))
for i in translations.items():
    print(i)
    data.replace(translations, inplace=True)
print(data)