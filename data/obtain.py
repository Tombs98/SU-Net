import os

print(os.getcwd())
class Obtain_data():
    def deg_list():
        res = open('./data_path-any/f_deg-any.txt')
        lines = res.readlines()
        t = []
        for i in lines:
            if i != '\n':
                t.append(float(i))
        # print(len(t))
        return t


    def t_list():
        print("os", os.getcwd())
        t = open('./data_path-any/t_path-any.txt')
        lines = t.readlines()
        res = []
        for i in lines:
            i = i.replace('\n', '')
            res.append(i)
        # print(len(lines))
        return res


