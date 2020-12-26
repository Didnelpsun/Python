import requests
import re
import argparse

url = 'https://www.baidu.com'
with requests.get(url) as response:
    print(response.text)

# regex_title = 'class="j_th_tit ">(.+?)</a>'
# pattern = re.compile(regex_title)
# result = pattern.findall(response.text)
# i = 1
# for item in result:
#     print(f'[{i}]{item}')
#     i += 1


"Accept": "application/json, text/javascript, */*; q=0.01"
"Accept-Encoding": "gzip, deflate, br"
"Accept-Language": "zh-CN,zh;q=0.9"
"Connection": "keep-alive"
"Cookie": "_trs_uv=kj4gd4hn_6_h5ld; JSESSIONID=7iCamhpmDETtsD7gqF6uCz2TpywVXKaNqXwCgarZdmDIVSQlC-67!1550732730; u=1"
"Host": "data.stats.gov.cn"
"Referer": "https://data.stats.gov.cn/easyquery.htm?cn=C01&zb=A0G0X"
"Sec-Fetch-Dest": "empty"
"Sec-Fetch-Mode": "cors"
"Sec-Fetch-Site": "same-origin"
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36"
"X-Requested-With": "XMLHttpRequest"