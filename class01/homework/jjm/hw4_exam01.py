# exam01
# 튜플형 데이터 A, B를 만들고 C를 추가한다.


data1 = ('A', 'B',)
data2 = ('C',)
data3 = data1 + data2
print(data3)

# TypeError: can only concatenate tuple (not "str") to tuple
# ("A", "B") => ("A", "B",)
