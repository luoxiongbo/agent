scores = [88, 92, 79, 93, 85]

print(scores[0])
print(scores[-1])
print(scores[0:3])
scores[2] = 80
avg = sum(scores) /len(scores)
max(scores)
for score in scores:
    print(score)
for i, score in enumerate(scores):
    print(i, score)