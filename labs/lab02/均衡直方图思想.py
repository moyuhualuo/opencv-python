import numpy as np

# 5 x 5 随机数组，取值为 [0, 10)
scr = np.random.randint(0, 10, size=[5, 5], dtype=np.uint8)
# 查看
print(scr)
# 创建静态数组 arr 记录次数， sum_arr 为累加算数和
sum_arr = [0] * 11
arr = [0] * 10
# arr 实现
for i in range(len(scr[0])):
    for j in scr[i]:
        arr[j] += 1
# sum_arr 实现
for j in range(10):
    arr[j] = arr[j] / 25
    sum_arr[j + 1] = (sum_arr[j + 1] + sum_arr[j] + arr[j])
# 四舍五入
for i in range(1, 11):
    sum_arr[i] = int(sum_arr[i] * 8)
# 查看累加数组
print(sum_arr[1:])
# 把累加数组终值替换到scr 数组
for i in range(len(scr)):
    for j in range(len(scr[0])):
        scr[i][j] = sum_arr[scr[i][j]]
print(scr)



