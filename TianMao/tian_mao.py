# -*- coding: utf-8 -*-
# 天猫数据获取

# 函数说明：
# 获取Dict式的AJAX数据
# 参数列表：
# base_url[str]：查询数据的URL
# data_url[str]：获取数据的URL
# 返回值：
# dict[]：保存数据的字典
import requests
import re
import numpy as np
import math
import matplotlib.pyplot as plt
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


def headers(cookies):
    return {
        "accept": "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "zh-CN,zh;q=0.9",
        "cookie": cookies,
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36",
        "x-requested-with": "XMLHttpRequest"
    }


style = ['r.', 'm.', 'c.', 'g.', 'b.', 'y.']


# 获取数据
def get_bare_data(url, cookies):
    header = headers(cookies)
    pattern = re.compile(r'price\\">\S+')
    int_data = []
    for i in range(1, 3):
        page = "&pageNo=" + str(i)
        with requests.get(url+page, verify=False, headers=header) as response:
            data = pattern.findall(response.text)
        if not data:
            break
        for item in data:
            num = float(item[8:])
            if num != 0:
                int_data.append(num)
    return sorted(int_data)


# 集中密集度
def focus(data, style='r.'):
    data = np.array(data)
    ones = np.ones(data.shape)
    plt.plot(ones, data, style)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()


# 等宽离散化
def width(data, k):
    data_min = data[0]
    data_max = data[-1]
    slide = (data_max - data_min) / k
    value_list = []
    for i in range(k):
        value_list.append(data_min + slide * i)
    value_list.append(data_max)
    x_list = []
    y_list = []
    for item in data:
        for i in range(k):
            if value_list[i] <= item < value_list[i + 1]:
                x_list.append(item)
                y_list.append(i)
            elif i == k-1 and item == value_list[i + 1]:
                x_list.append(item)
                y_list.append(i)
    for index in range(len(x_list)):
        plt.plot(x_list[index], y_list[index], style[int(y_list[index])])
    plt.show()


# 等频离散化，参数为列表与分类值
def frequency(data, k):
    slide = math.ceil((len(data)) / k)
    index_list = []
    for i in range(k):
        index_list.append(slide*i)
    index_list.append(len(data)-1)
    x_list = []
    y_list = []
    for index in range(len(data)):
        for i in range(len(index_list)-1):
            if index_list[i] <= index < index_list[i+1]:
                x_list.append(data[index])
                y_list.append(i)
            elif index == index_list[i+1] and index == len(data)-1:
                x_list.append(data[index])
                y_list.append(i)
    for index in range(len(x_list)):
        plt.plot(x_list[index], y_list[index], style[int(y_list[index])])
    plt.show()


if __name__ == "__main__":
    # cookies需要天猫账户登录后获取，一天有效期
    cookie = "cna=CMxYGAwKFEcCAdrHz1iBA1Bd; t=f410036620eab852d92af980ac035d0a; _tb_token_=eb41ea35783f; cookie2=1b5a5a0df353f62236bec33b36a7c7ad; xlly_s=1; tk_trace=1; dnk=tb037632730; uc1=cookie16=URm48syIJ1yk0MX2J7mAAEhTuw%3D%3D&cookie14=Uoe0ZNVW5nCqVg%3D%3D&pas=0&cookie21=WqG3DMC9FxUx&existShop=false&cookie15=V32FPkk%2Fw0dUvg%3D%3D; uc3=vt3=F8dCuAMlGY5evYqmh48%3D&nk2=F5RFg5%2BvNijaIXg%3D&id2=VyyUygN0WuVbFg%3D%3D&lg2=VFC%2FuZ9ayeYq2g%3D%3D; tracknick=tb037632730; lid=tb037632730; uc4=id4=0%40VXtbY2iR4W2GoKREfj44Y6V6DT0H&nk4=0%40FY4O6pd4YC3z5T%2Bg5TOLBMHY%2FDNtoQ%3D%3D; _l_g_=Ug%3D%3D; unb=4051864925; lgc=tb037632730; cookie1=UNJSvkqUUJFnyyJqREC%2F7yKrQGUeLgTjXSELwknfWP4%3D; login=true; cookie17=VyyUygN0WuVbFg%3D%3D; _nk_=tb037632730; sgcookie=E100ICrmie4PluSAruNRJTxdDN3Y%2BSPen5mXqlyD7pvDrzz6E%2F1nqpmk%2FCnXWLLwuP4kzHrBhT784zqoEj%2BaJVUkbw%3D%3D; sg=05d; csg=a1de1f74; _m_h5_tk=b721c201fa4477220a437a1250a03f94_1609381662190; _m_h5_tk_enc=c3c68f5061b39b04341ad0ccdafb843c; enc=TJwMk8zD2C9r2OSF5T56ll7RmSJg5MD5NouegKQPf0z8YMQ9y8sQ0l0%2FDM8WRt6eywgj%2Fh7kybQLYSSQhhRKtA%3D%3D; pnm_cku822=; tfstk=cSwdBRDcFNbHVcnT0WCMNINlnkSGZ27-5Hgkei-UR1z9cV9Ri6w0HTJVO0TKBGC..; l=eBaf38kuO0YWzI18BOfwourza77O7IRxnuPzaNbMiOCPOiCp57KNWZ-BTNT9CnhVh6epR37kyyKQBeYBqIcWSh1if65fNTMmn; isg=BODgXXWuq841iRfymdd8b0Rmse6y6cSzrI5DfVrxvvuOVYB_AvpCQiBj7P1VV3yL"
    K = 5
    # 雅诗兰黛
    ysld_url = "https://esteelauder.tmall.com/i/asynSearch.htm?_ksTS=1609399765026_138&callback=jsonp139&mid=w-14579014202-0&wid=14579014202&path=/search.htm&search=y&spm=a1z10.3-b-s.w4011-14579014202.463.10785c6cCLwmQo&orderType=hotsell_desc&viewType=grid&shopId=110224300"
    ysld = get_bare_data(ysld_url, cookies=cookie)
    width(ysld, K)
    frequency(ysld, K)
    # SK-II
    sk_url = "https://skii.tmall.com/i/asynSearch.htm?_ksTS=1609402786063_56&callback=jsonp57&mid=w-14630548658-0&wid=14630548658&path=/search.htm&search=y&spm=a1z10.1-b-s.w5001-21498027762.5.21114887JO4KA8&scene=taobao_shop"
    sk = get_bare_data(sk_url, cookies=cookie)
    width(sk, K)
    frequency(sk, K)
    # 海蓝之谜
    hlzm_url = "https://lamer.tmall.com/i/asynSearch.htm?_ksTS=1609402977460_125&callback=jsonp126&mid=w-14859464013-0&wid=14859464013&path=/search.htm&search=y&spm=a1z10.3-b-s.w4011-14859464013.265.21c03663CwpkHS&scene=taobao_shop"
    hlzm = get_bare_data(hlzm_url, cookies=cookie)
    width(hlzm, K)
    frequency(hlzm, K)
    # 兰蔻
    lk_url = "https://lancome.tmall.com/i/asynSearch.htm?_ksTS=1609403127863_125&callback=jsonp126&mid=w-14640892229-0&wid=14640892229&path=/search.htm&search=y&spm=a1z10.3-b-s.w4011-14640892229.465.25d83c22myMQWt&scene=taobao_shop"
    lk = get_bare_data(lk_url, cookies=cookie)
    width(lk, K)
    frequency(lk, K)
    # 欧莱雅
    oly_url = "https://loreal.tmall.com/i/asynSearch.htm?_ksTS=1609403293827_124&callback=jsonp125&mid=w-22757614050-0&wid=22757614050&path=/category.htm&spm=a1z10.3-b-s.w4011-22757614050.346.2a8665e9BdfK0w&search=y"
    oly = get_bare_data(oly_url, cookies=cookie)
    width(oly, K)
    frequency(oly, K)
    # 御泥坊
    ynf_url = "https://yunifang.tmall.com/i/asynSearch.htm?_ksTS=1609403472865_129&callback=jsonp130&mid=w-14439323381-0&wid=14439323381&path=/category.htm&spm=a1z10.5-b-s.w4011-14439323381.554.2fba1fc9LxZJn6&scene=taobao_shop"
    ynf = get_bare_data(ynf_url, cookies=cookie)
    width(ynf, K)
    frequency(ynf, K)
    # 自然堂
    zrt_url = "https://chando.tmall.com/i/asynSearch.htm?_ksTS=1609403641788_125&callback=jsonp126&mid=w-14595760569-0&wid=14595760569&path=/search.htm&search=y&spm=a1z10.3-b-s.w4011-14595760569.58.194f6712Vv5M5o&scene=taobao_shop"
    zrt = get_bare_data(zrt_url, cookies=cookie)
    width(zrt, K)
    frequency(zrt, K)
    美宝莲
    mbl_url = "https://maybelline.tmall.com/i/asynSearch.htm?_ksTS=1609403779721_47&callback=jsonp48&mid=w-21350826969-0&wid=21350826969&path=/search.htm&spm=a1z10.1-b-s.w5001-22451368511.3.559625950jt6WR&prc=1&search=y&shopId=68295332&scene=taobao_shop"
    mbl = get_bare_data(mbl_url, cookies=cookie)
    width(mbl, K)
    frequency(mbl, K)
    # 香奈儿
    xne_url = "https://chanel.tmall.com/i/asynSearch.htm?_ksTS=1609404012855_89&callback=jsonp90&mid=w-21809493890-0&wid=21809493890&path=/search.htm&search=y&spm=a1z10.3-b-s.w4011-21809493890.72.128bd83eho8AAp"
    xne = get_bare_data(xne_url, cookies=cookie)
    width(xne, K)
    frequency(xne, K)
    # 迪奥
    da_url = "https://dior.tmall.com/i/asynSearch.htm?_ksTS=1609404102641_124&callback=jsonp125&mid=w-22803226051-0&wid=22803226051&path=/search.htm&search=y&spm=a1z10.3-b-s.w4011-22803226051.61.529c2487IY3ow5"
    da = get_bare_data(da_url, cookies=cookie)
    width(da, K)
    frequency(da, K)
    # 完美日记
    wmrj_url = "https://perfectdiary.tmall.com/i/asynSearch.htm?_ksTS=1609404158566_124&callback=jsonp125&mid=w-16857514267-0&wid=16857514267&path=/search.htm&search=y&spm=a1z10.3-b-s.w4011-16857514267.65.6abb46936fA8P4"
    wmrj = get_bare_data(wmrj_url, cookies=cookie)
    width(wmrj, K)
    frequency(wmrj, K)