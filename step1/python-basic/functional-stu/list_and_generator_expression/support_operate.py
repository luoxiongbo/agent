list_comp = [1, 2, 3, 4]
print(list_comp[0])   # 1
print(len(list_comp)) # 4
print(50*"_")

gen_exp = (x for x in [1, 2, 3, 4])
for x in gen_exp:
    print(x)
# print(gen_exp[0])   # 报错 TypeError: 'generator' object-stu is not subscriptable
# print(len(gen_exp)) # 报错 TypeError: object of type 'generator' has no len()