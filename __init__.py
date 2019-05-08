

import functools

def log(text):
    def decorator(func):
        print (func.__name__)
        #@functools.wraps(func)
        def wrapper(*args, **kw):
            print('%s %s():' % (text, func.__name__))
            return func(*args, **kw)
        return wrapper
    return decorator
    print(decorator.__name__)

@log('execute')
def now():
    print('2015-3-25')


'''
def log(func):
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper

@log
def now():
    print('2015-3-25')
'''

'''
def log(func):
    def decorator(func1):
        def wrapper(*args, **kw):
            print('begin call')
            result = func1(*args, **kw)
            print('end call')
            return result
        return wrapper

    if isinstance(func, str):
        return decorator
    else:
        return decorator(func)

@log('test')
def f():
    print('f is invoked')
'''
'''
if __name__== '__main__':
    print (now.__name__)
    now()
'''
import asyncio

@asyncio.coroutine
def wget(host):
    print('wget %s...' % host)
    connect = asyncio.open_connection(host, 80)
    reader, writer = yield from connect
    print(reader)
    print(writer)
    header = 'GET / HTTP/1.0\r\nHost: %s\r\n\r\n' % host
    writer.write(header.encode('utf-8'))
    yield from writer.drain()
    while True:
        line = yield from reader.readline()
        if line == b'\r\n':
            break
        print('%s header > %s' % (host, line.decode('utf-8').rstrip()))
    # Ignore the body, close the socket
    writer.close()

loop = asyncio.get_event_loop()
tasks = [wget(host) for host in ['www.sina.com.cn', 'www.sohu.com', 'www.163.com']]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()