from xml.dom.minidom import parse
import xml.etree.ElementTree as et


def show_dorms(path):
    # 首先读入文件并构建一个文档树
    DOMTree = parse(path)
    collection = DOMTree.documentElement

    dorms = collection.getElementsByTagName("dorm")
    dorm_num = 0
    # 编辑宿舍数据
    for dorm in dorms:
        dorm_num += 1
        print("***宿舍***")
        print("宿舍号: %s" % dorm.getAttribute("id"))
        # 遍历学生数据
        students = dorm.getElementsByTagName("student")
        for student in students:
            print("学生学号: {0} 学生姓名：{1} 学生电话号码：{2} 备注：".format(student.getAttribute("id"),
                                                             student.getElementsByTagName("name")[0].childNodes[0].data,
                                                             student.getElementsByTagName("telephone")[0].childNodes[
                                                                 0].data), end='')
            if student.getElementsByTagName("remarks"):
                print(student.getElementsByTagName("remarks")[0].childNodes[0].data)
            else:
                print()
    print()
    return dorm_num


def change_telephone(path, student_id, value):
    # 首先读入文件并构建一个文档树
    DOMTree = parse(path)
    collection = DOMTree.documentElement

    dorms = collection.getElementsByTagName("dorm")
    for dorm in dorms:
        # 遍历学生数据
        students = dorm.getElementsByTagName("student")
        for student in students:
            if str(student.getAttribute("id")) == student_id:
                student.getElementsByTagName("telephone")[0].childNodes[0].data = value
            return 0
    return None


if __name__ == "__main__":
    path = "xml/data2.xml"
    show_dorms(path)
    change_telephone(path, '2018212603', '5555')
    show_dorms(path)
