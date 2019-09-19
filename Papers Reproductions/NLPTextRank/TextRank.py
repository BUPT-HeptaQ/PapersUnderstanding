""
Keywords extraction based on TextRank algorithm

# note the default filter part of speech
  jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
  jieba.analyse.TextRank() build user_defined TextRank instance


import jieba.analyse as analyse
import pandas as pd
data_file = pd.read_csv("D:technology_news.csv", encoding='utf-8')
data_file = data_file.dropna()
lines = data_file.content.values.tolist()
content = " ".join(str(lines))

print("  ".join(analyse.textrank(content, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))))
print("---------------------separate line----------------")
print("  ".join(analyse.textrank(content, topK=20, withWeight=False, allowPOS=('ns', 'n'))))

