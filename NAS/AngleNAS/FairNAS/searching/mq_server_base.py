import os
import pika
from multiprocessing.pool import ThreadPool
import threading
import pickle
from functools import partial
from typing import Tuple
from queue import Queue
import time

from abc import ABCMeta,abstractmethod

import sys
sys.setrecursionlimit(100000)


import functools
import termcolor
import datetime
print=functools.partial(print,flush=True)
tostring=lambda *args:' '.join(map(str,args))
printred=lambda *args,**kwargs:termcolor.cprint(tostring(*args),color='red',flush=True,**kwargs)
printgreen=lambda *args,**kwargs:termcolor.cprint(tostring(*args),color='green',flush=True,**kwargs)
def info_prefix():
    return '[{} info]'.format(datetime.datetime(1,1,1).now())


class MessageQueueServerBase(metaclass=ABCMeta):

    _rmq_server_addr = None

    _username=None

    _request_pipe_name = None

    _response_pipe_name = None

    _eval_callback = None


    _nr_threads = None
    
    heartbeat=0

    @property
    def nr_threads(self):
        return self._nr_threads

    @nr_threads.setter
    def nr_threads(self, v):
        self._nr_threads = v

    @abstractmethod
    def eval():
        pass

    def __init__(self, rmq_server_addr:str, port:int, username:str, request_pipe_name:str, response_pipe_name:str):
        self._rmq_server_addr = rmq_server_addr
        self._port=port
        self._username=username
        self._request_pipe_name = request_pipe_name
        self._response_pipe_name = response_pipe_name

    def listen(self, reset_pipe=False):
        assert self.nr_threads is not None
        if reset_pipe:
            printgreen(info_prefix(),'Reset existing pipes.')
            print('request_pipe_name:',self._request_pipe_name)
            print('response_pipe_name:',self._response_pipe_name)
            print()
            self._clear_pipe(self._request_pipe_name)
            self._clear_pipe(self._response_pipe_name)

        threads = ThreadPool(self.nr_threads)
        threads.map(self._listen_thread, range(self.nr_threads))

    def _clear_pipe(self, pipe_name):
        conn = pika.BlockingConnection(pika.ConnectionParameters(host=self._rmq_server_addr,port=self._port,heartbeat=self.heartbeat,blocked_connection_timeout=None,virtual_host='/',credentials=pika.PlainCredentials(self._username,self._username)))
        channel = conn.channel()
        channel.queue_delete(queue=pipe_name)
        channel.close()
        conn.close()

    def _listen_thread(self, thread_idx):
        conn = pika.BlockingConnection(pika.ConnectionParameters(host=self._rmq_server_addr,port=self._port,heartbeat=self.heartbeat,blocked_connection_timeout=None,virtual_host='/',credentials=pika.PlainCredentials(self._username,self._username)))
        pika.credentials.ExternalCredentials
        channel_request = conn.channel()
        channel_request.queue_declare(queue=self._request_pipe_name)

        channel_response = conn.channel()
        channel_response.queue_declare(queue=self._response_pipe_name)

        def fail(*args,**kwargs):
            print('args:',args)
            print('kwargs:',kwargs)
            raise NotImplementedError

        channel_response.add_on_cancel_callback(fail)

        channel_request.basic_qos(prefetch_count=1)
        channel_request.basic_consume(self._request_pipe_name, partial(self._request_callback, channel_response=channel_response))

        printgreen(info_prefix(),'Listening ({})'.format(thread_idx))
        print()
        channel_request.start_consuming()

    def _request_callback(self, cur_channel, frame, properties, body, channel_response):
        data = pickle.loads(body)
        assert(len(data) == 2)

        key = data[0]
        content = data[1]

        printgreen(info_prefix(),'receive key:',key)
        print('content:',content)
        print('waiting for evaluation...')
        print()

        try:
            result = self.eval(content)
        except:
            import traceback
            traceback.print_exc()
            time.sleep(10)
            os._exit(1)
            print()
            return {'status':'uncatched error'}

        #assert isinstance(result,dict)

        printgreen(info_prefix(),'finish key:',key)
        print('content:',content)
        print('result:',result)
        print()

        del data,content

        obj = pickle.dumps((key, result))

        if cur_channel.is_closed:
            raise NotImplementedError

        printgreen(info_prefix(),'return result')

        channel_response.basic_publish(exchange='', routing_key=self._response_pipe_name, body=obj)
        cur_channel.basic_ack(delivery_tag=frame.delivery_tag)
    def run(self,threads,*,reset_pipe=False):
        self.nr_threads=threads
        self.listen(reset_pipe=reset_pipe)

class MessageQueueClientBase(metaclass=ABCMeta):

    _rmq_server_addr = None

    _username=None

    _request_pipe_name = None

    _response_pipe_name = None

    _channel_request = None

    _buffer = None

    _buffer_queue = None

    _data_idx = None

    _thread=None

    heartbeat=0


    def __init__(self, rmq_server_addr:str, port:int, username:str, request_pipe_name:str, response_pipe_name:str):
        self._rmq_server_addr = rmq_server_addr
        self._port=port
        self._username=username
        self._request_pipe_name = request_pipe_name
        self._response_pipe_name = response_pipe_name

        self._buffer = dict()
        self._buffer_queue = Queue()
        self._data_idx = 0

    def save(self):
        return {'buffer':self._buffer}
    def load(self,info):
        self._buffer=info['buffer']

    def connect(self, reset_pipe=False):
        conn = pika.BlockingConnection(pika.ConnectionParameters(host=self._rmq_server_addr,port=self._port,heartbeat=self.heartbeat,blocked_connection_timeout=None,virtual_host='/',credentials=pika.PlainCredentials(self._username,self._username)))
        self._conn=conn

        self._channel_request = conn.channel()
        if reset_pipe:
            self._channel_request.queue_delete(queue=self._request_pipe_name)
        self._channel_request.queue_declare(queue=self._request_pipe_name)

        def start_consuming():
            conn = pika.BlockingConnection(pika.ConnectionParameters(host=self._rmq_server_addr,port=self._port,heartbeat=self.heartbeat,blocked_connection_timeout=None,virtual_host='/',credentials=pika.PlainCredentials(self._username,self._username)))
            channel_response = conn.channel()
            if reset_pipe:
                channel_response.queue_declare(queue=self._response_pipe_name)
            channel_response.queue_declare(queue=self._response_pipe_name)

            channel_response.basic_consume(self._response_pipe_name, self._fetch_response_callback)
            channel_response.start_consuming()

        if self._thread is not None:
            #self._thread._stop()
            self._thread=None

        thread = threading.Thread(target=start_consuming)
        thread.start()
        self._thread=thread


    def _fetch_response_callback(self, cur_channel, frame, properties, body):
        #print('callback')

        data = pickle.loads(body)

        assert len(data) == 2

        printgreen(info_prefix(),'receive key:',data[0])
        print('result:',data[1])
        print()

        #print(id(self),type(self))
        self._buffer_queue.put(data)

        cur_channel.basic_ack(delivery_tag=frame.delivery_tag)

    @abstractmethod
    def send(self, content, *,key:str=None) -> str:
        self._data_idx += 1
        key = key or "{}-{}".format(self._data_idx, time.time())

        printgreen(info_prefix(),'send key',key)
        print('content:',content)
        print()

        obj = pickle.dumps((key, content))

        while True:
            try:
                self._channel_request.basic_publish(exchange='', routing_key=self._request_pipe_name, body=obj)
                break
            except:
                import traceback
                traceback.print_exc()
                time.sleep(10)
                print('Send failed, reconnecting >>>>>')
                print('reconnect')
                self.connect()

        return key

    def get(self, key:str, *, timeout) -> str:
        printgreen(info_prefix(),'try to get key:',key)
        if key in self._buffer:
            data = self._buffer[key]
            del self._buffer[key]
            return data
        #print ('buffer:',self._buffer)

        #print(id(self),type(self))

        begin_time=time.time()

        while True:
            #assert time.time()-begin_time<timeout
            cur_key, content = self._buffer_queue.get(timeout=timeout)
            #print('data:',cur_key,content)
            if cur_key == key:
                return content
            else:
                self._buffer[cur_key] = content
                return None
    def get_retry(self,info,*,key,timeout):
        while True:
            try:
                if key is None:
                    key=self.send(info)
                    print('new key')
                res=self.get(key,timeout=timeout);
                return res
            except:
                import traceback
                traceback.print_exc()
                time.sleep(1)
                key=None

if __name__ == '__main__':
    pass
