#coding=utf-8
#MYSQL 8.0 社区版 连接信息
RESP_200 = dict(code=200, data=None, msg='OK')

# 数据库连接参数
DB_CONN_DICT = dict(
    host = '127.0.0.1', 
    port = 3306, 
    user = 'root', 
    password = '123456' ,
    database = 'sec_ai')

# 漏洞类型字典
vulnType = {'vuln': '其它攻击类型',
            'miscellaneous': '其他杂项',
            'sqli': 'sql注入攻击',
            'sql injection': 'sql注入攻击',
            'lfi': '本地文件包含',
            'local file inclusion': '本地文件包含',
            'cmdi': '命令注入攻击',
            'command injection': '命令注入攻击',
            'xss': 'XSS攻击',
            'cross site scripting': 'XSS攻击',
            'robot': '恶意爬虫',
            'rfi': '远程文件包含',
            'custom_custom': '精准防护',
            'webshell': '网站木马',
            'custom_whiteblackip': '黑白名单拦截',
            'custom_geoip': '地理访问控制拦截',
            'antitamper': '防篡改',
            'anticrawler': '反爬虫',
            'leakage': '网站信息防泄漏',
            'illegal': '非法请求'}

