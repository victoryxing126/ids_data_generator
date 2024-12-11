#coding=utf-8
import json, csv, uuid, os
from urllib.parse import urlparse

def parse_har_to_csv(har_data, csv_file):
    # har_data：从 .har 文件中读取的 [dict(1), dict(2), ...]
    # 打开 CSV 文件进行写入
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'action', 'attack', 'cookie', 'headers', 'host', 'id', 'payload',
            'payload_location', 'policyid', 'request_line', 'rule', 'sip', 'time', 'url'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 解析 .har 文件中的每个请求
        for entry in har_data:
            request = entry['request']
            response = entry['response']

            # 获取 cookie
            cookies = '; '.join([f"{cookie['name']}={cookie['value']}" for cookie in request.get('cookies', [])])

            # 获取 headers
            headers = {header['name']: header['value'] for header in request.get('headers', [])}

            # 获取 host
            url_parsed = urlparse(request['url'])
            host = url_parsed.netloc

            # 获取 payload 和 payload_location
            payload = None
            payload_location = None
            if request['method'] in ['POST', 'PUT', 'PATCH']:
                if 'postData' in request:
                    payload = request['postData'].get('text', '')
                    if 'params' in request['postData']:
                        payload = {param['name']: param['value'] for param in request['postData']['params']}
                    payload_location = 'body'
            elif 'queryString' in request:
                payload = {param['name']: param['value'] for param in request['queryString']}
                payload_location = 'uri'

            # 获取 request_line
            request_line = f"{request['method']} {request['url']} HTTP/{request['httpVersion']}"

            # 获取时间
            time = entry['startedDateTime']

            # 写入 CSV 行
            writer.writerow({
                'action': '',
                'attack': '',
                'cookie': cookies,
                'headers': json.dumps(headers),
                'host': host,
                'id': '',
                'payload': json.dumps(payload) if payload else '',
                'payload_location': payload_location if payload_location else '',
                'policyid': '',
                'request_line': request_line,
                'rule': '',
                'sip': '',
                'time': time,
                'url': request['url']
            })

if __name__ == '__main__':
    # 指定 .har 文件路径和输出的 CSV 文件路径
    har_file_path = '0-Data-TrainTest/www.freebuf.com.har'
    with open(har_file_path, 'r', encoding='utf-8') as f:
        har_data = json.load(f)
    #将 har 数据三等分、分别写入 train、test、val数据集
    train_entries = har_data.get("log").get("entries")[0:int(len(har_data.get('log').get('entries'))/3)]
    test_entries = har_data.get("log").get("entries")[int(len(har_data.get('log').get('entries'))/3):2*int(len(har_data.get('log').get('entries'))/3)]
    val_entries = har_data.get("log").get("entries")[2*int(len(har_data.get('log').get('entries'))/3):len(har_data.get('log').get('entries'))+1]
    parse_har_to_csv(test_entries, f'0-Data-TrainTest/test-from-har-{uuid.uuid1().__str__()}.csv')
    parse_har_to_csv(train_entries, f'0-Data-TrainTest/train-from-har-{uuid.uuid1().__str__()}.csv')
    parse_har_to_csv(val_entries, f'0-Data-TrainTest/val-from-har-{uuid.uuid1().__str__()}.csv')
    try:
        os.remove(har_file_path)
    except:
        pass
    print("Done~")

