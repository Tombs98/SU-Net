import pathlib
import pytesseract
from PIL import Image
import datetime

starttime = datetime.datetime.now()
#加载数据列表
data_path = pathlib.Path("./data_t1")  #这里是你要加载的图片文件夹路径、每个文件夹下分各个标签的文件
all_image_paths = list(data_path.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]  # 所有图片路径的列表
print(len(all_image_paths))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def data_pro(path):
    '''
    获取图片中的参数，并将参数和图片制作 成数据集
    :param path:
    :return:
    '''

    img_list = [] #图片列表
    pitch_list = [] #俯仰角列表
    azimuth_list = [] #方位角列表
    f_path = open('f_deg.txt', 'w')
    t_path = open('t_path.txt', 'w')
    e_path = open('error.txt', 'w')
    for i in path:
        # print(i)

        flag = 0  #标记是否将数据存入
        #读取图片
        im = Image.open(i)
        # 识别文字
        string = pytesseract.image_to_string(im)
        string = string.lower()

        pitch_index = string.find('pitch:')
        azi_index = string.find('azi')
        if pitch_index == -1:
            flag = 1
        if azi_index == -1:
            flag = 1
        # azimuth = temp_list[4].split(':')[1]
        if flag == 0:
            pitch_index = pitch_index + 6
            pitch = string[pitch_index:azi_index]
            # print(pitch)
            pitch = pitch.replace(',', '.')
            pitch = pitch.replace('i', '1')
            pitch = pitch.replace(" ", "")
            if is_number(pitch):
                if float(pitch)<=360:
                    print(pitch)
                    f_path.write(pitch)
                    f_path.write('\n')
                    f_path.flush()

                    t_path.write(i)
                    t_path.write('\n')
                    t_path.flush()
                else:
                    e_path.write(pitch)
                    e_path.write('\n')
                    e_path.write('      ')
                    e_path.write(i)
                    e_path.flush()
            else:
                e_path.write(pitch)
                e_path.write('\n')
                e_path.write('      ')
                e_path.write(i)
                e_path.flush()
            # print(azimuth)
            # pitch_list.append(pitch)
            # azimuth_list.append(azimuth)
    f_path.close()
    t_path.close()
    e_path.close()
    return pitch_list, azimuth_list

p_list, a_list = data_pro(all_image_paths)

print(p_list)
endtime = datetime.datetime.now()
print("运行时间：", (endtime - starttime).seconds)