import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def download_file(url):
    local_filename = url.split('/')[-1]
    # 发送请求获取文件
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def scrape_and_download(url):
    # 获取网页内容
    response = requests.get(url)
    response.raise_for_status()  # 检查请求是否成功

    # 解析网页
    soup = BeautifulSoup(response.text, 'html.parser')

    # 找到所有控件的URL
    urls = []
    for link in soup.find_all('a', href=True):
        urls.append(urljoin(url, link['href']))
    for img in soup.find_all('img', src=True):
        urls.append(urljoin(url, img['src']))
    for script in soup.find_all('script', src=True):
        urls.append(urljoin(url, script['src']))
    for link in soup.find_all('link', href=True):
        urls.append(urljoin(url, link['href']))

    # 下载所有抓取到的URL
    for file_url in set(urls):  # 使用set去重
        try:
            print(f"Downloading {file_url}...")
            download_file(file_url)
        except Exception as e:
            print(f"Failed to download {file_url}: {e}")

if __name__ == "__main__":
    target_url = input("请输入要抓取的URL: ")
    scrape_and_download(target_url)

