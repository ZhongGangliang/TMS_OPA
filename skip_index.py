#

skipIndexs = list(range(96*101, 101*101+1)) # 去除后脑部分
skipIndexs.extend(list(range(0*101, 12*101+1))) # 去除前额部分
for i in range(0, 101):
    skipIndexs.extend(list(range(i*101+84, i*101+102))) # 去除耳朵部分
