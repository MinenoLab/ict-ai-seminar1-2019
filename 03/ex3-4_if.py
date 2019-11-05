print("分子を入力してください: ", end="")
bunshi = input()
print("分母を入力してください: ", end="")
bunbo = input()

if float(bunbo) == 0:
    print("分母0を検知しました")
else:
    answer = float(bunshi) / float(bunbo)
    print("答え: " + str(answer))

