from sklearn.model_selection import train_test_split

data = []
num_example = 0
with open("data/DataSet.txt", 'r') as f:
    for line in f:
        data.append(line)

train, develop = train_test_split(data, test_size=0.2)
develop, test = train_test_split(develop, test_size=0.5)

with open("data/train.txt", 'w') as f:
    for line in train:
        f.write(line)

with open("data/develop.txt", 'w') as f:
    for line in develop:
        f.write(line)

with open("data/develop.txt", 'w') as f:
    for line in test:
        f.write(line)

