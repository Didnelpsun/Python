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


