from types import FunctionType
import re
import requests
import os
import imageio
import json
from tqdm import tqdm
from PixivClass import PixivUser
from PixivClass import Image
import threading
from multiprocessing import Pool
import time

#爬虫时间测试: (从相同的id列表获取图片)
#单进程(单线程): 363s
#多进程(4 进程): 176s
#多进程(8 进程): 95s
#多进程(16进程): 57s
#多线程(4 线程): 239s
#多线程(8 线程): 123s
#多线程(16线程): 73

#注意: python的多线程是真实线程, 但是解释器自带锁, 只能在一个CPU上执行, 只有使用多进程才有可能实现并发
pixivHeaders = {
'method':'GET',
'cookie':'first_visit_datetime_pc=2022-07-30+21:56:08; p_ab_id=4; p_ab_id_2=8; p_ab_d_id=24300274; _fbp=fb.1.1659185776163.1855011282; __utma=235335808.723299299.1659185772.1659185778.1659185778.1; __utmc=235335808; __utmz=235335808.1659185778.1.1.utmcsr=zhidao.baidu.com|utmccn=(referral)|utmcmd=referral|utmcct=/question/551945585.html; __utmt=1; yuid_b=IUJHmSQ; _gid=GA1.2.1027644613.1659185789; _im_vid=01G97JW02M164KTW505MYQ3ZTS; PHPSESSID=13748038_jko4pfRHJRYIu9LKEtZovKMqvuo1K4HL; device_token=95fc0a113a28a9a3111c2f36b135bded; c_type=102; privacy_policy_agreement=0; privacy_policy_notification=0; a_type=0; b_type=1; __utmv=235335808.|3=plan=normal=1^5=gender=male=1^6=user_id=13748038=1^11=lang=zh=1; QSI_S_ZN_5hF4My7Ad6VNNAi=v:0:0; cto_bundle=I_OMcF9halBZeHlhSks0V0RqWlN3YU11WXc1S1dGRVo2M3d1N3lZTWtnYVJDODRzWEd6UURZajZ5UmJlRjA4WnM3MzVCUkJCU0klMkJ2ZGJsY20xM1FuNnIyZzNCMTcwZUhKczQwMmN2d0t0WmhuUDFZTm1OV3ZZazV1UmVOSjA1ZnFJM29VZjF3UGlIb2VJVjcxTEpXOTBFemxDdyUzRCUzRA; tag_view_ranking=0xsDLqCEW6~F1AKdqvivA~jEthU99Q2P~3gc3uGrU1V~KN7uxuR89w~cb91hphOyK~jGhu1L3S_w~K8qOt_bKU_~OlC0hKTA-T; categorized_tags=F1AKdqvivA~b8b4-hqot7; tags_sended=1; adr_id=pbbFDZ5croUEokQgxIpfQ1uWpIE8Q4aasA98brGSDs9r6qWM; _ga=GA1.1.723299299.1659185772; __cf_bm=zKtpRSo9jyvD31OFWcOtu3h.AZgUNnc.zA074plHiLs-1659186275-0-AdR4pgiYO1DUZAji/o16gt1Vp5eS5Yx6LJwa5SwC1GQLySRb8ovK2+BLPnQho3l9pLpml843qk7Ahql55h7L2ykPUejfzCcl9In7CYCXRUSEDbvv1wRWeJvHLoK8Ete0BFFh8EG3YzHbUm3wzyU9y+nF/GhKbI5OdgTtPcmyC1fC8bdyRyFvgIdq/Mv99Pqssw==; __utmb=235335808.9.10.1659185778; _ga_75BBYNYN9J=GS1.1.1659185771.1.1.1659186276.0; _im_uid.3929=i.7J3ajAqLSFSB5u7_4INsdQ',
'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
'referer':'https://www.pixiv.net'
}
pixivDownloadHeaders = {
'method':'GET',
'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
'referer':'https://www.pixiv.net'
}
proxies = {#在梯子底下能翻到,用的本地地址和本地端口
'http':'http://127.0.0.1:19180',
'https':'http://127.0.0.1:19180'
}

def ListDownload(IDlist, Quality = 'regular', pbar = False, path = 'E:/WormDownloadLib/PixivImage/test/', order = '-1'):
    """_summary_
    Args:
        Quality (str, optional): Including:[small, regular, origin]
        pbar (bool, optional): Whether user apply the pbar
        order (bool, optional): Allows 3 mode: no / 1 / -1.
    """
    if order == 'no':
        for imageID in IDlist:
                Tempimage = Image(imageID)
                Tempimage.ImageDownload(Quality= Quality, path= path, pbar= pbar, order= order)
                print('\n')
        return
    elif order == 1:
        index = 1
        for imageID in IDlist:
            Tempimage = Image(imageID)
            Tempimage.ImageDownload(Quality= Quality, path= path, pbar= pbar, order= str(index))
            index += 1
            print('\n')
        pass
    else:
        for imageID in IDlist:
            Tempimage = Image(imageID)
            Tempimage.ImageDownload(Quality= Quality, path= path, pbar= pbar)
            print('\n')
            pass
    return True

#处在测试中的线程对象
class ThreadTest(threading.Thread):
    
    pass
#以下是多线程代码组
#可被线程类调用的对象(含有__call__方法)
class ThreadImage(object):
    #args不能带**, 原因自明
    def __init__(self, object_:Image, Func:FunctionType, kwargs):
        self.image = object_
        self.Func = Func
        self.kwargs = kwargs
        pass
    def __call__(self):
        self.Func(self.image, **self.kwargs)
        pass    
    pass
#线程函数, 用于调用image对象的方法
def func(image: Image, **args):
    image.ImageDownload(**args)
    pass
#多线程版的图片下载函数
#对锁的要求不高, 因为线程函数处理的是不同的对象
def ListDownload_thread(IDList, threads = 4, kwargs = {'Quality':'regular', 'pbar':False, 'order':'-1', 'path': 'C:/Users/DELL/Desktop/test2/'}):
    createVar = locals()
    step = threads
    #将id列表分组, 使进程池里一组n个进程对应一组n个id
    groupList = [IDList[i:i+step] for i in range(0,len(IDList),step)]
    print('Start to Download the list.')
    for group in groupList:
        threadList = []
        for i in range(step):
            try:
                image = Image(group[i])
                createVar['thread'+str(i)] = threading.Thread(target = ThreadImage(image, func, kwargs= kwargs))
                threadList.append(createVar['thread'+str(i)])
            except IndexError as e:
                print(e)
                pass
        for thread in threadList:
            thread.start()
        for thread in threadList:
            thread.join()
            pass
    
    pass


#以下是多进程代码组
#多进程版的图片下载
def ListDownload_processes(IDList, processes= 4, args_mode = ('regular', False, '-1', 'C:/Users/DELL/Desktop/test/')):
    step = processes
    #将id列表分组, 使进程池里一组n个进程对应一组n个id
    groupList = [IDList[i:i+step] for i in range(0,len(IDList),step)]
    #tempIDList = ['100176009','97186686','100634717','100612711']
    print('Start to Download the list.')
    for group in groupList:
        pool = Pool(processes= processes)
        for i in range(processes):
            try:
                image = Image(group[i])
                pool.apply_async(image.ImageDownload, args= args_mode)
            except IndexError:
                pass
            pass
        pool.close()
        pool.join()
    print('\n\nIllusitrations are downloaded over.')
    pass

#测试代码
if __name__ == '__main__':
    
    pass












"""
#线程不能封装, 否则依然顺序执行
def Thread_single(imageID):
    image = Image(imageID)
    temp_1 = ThreadImage(image, func, args={'path':'C:/Users/DELL/Desktop/', 'pbar':True})
    thread_1 = threading.Thread(target= temp_1)
    thread_1.start()
    thread_1.join()
    pass

def test():
    image = Image('100176009')
    temp_1 = ThreadImage(image, func, args={'path':'C:/Users/DELL/Desktop/', 'pbar':True})
    thread_1 = threading.Thread(target= temp_1)
    thread_1.start()
    
    image = Image('97186686')
    temp_2 = ThreadImage(image, func, args={'path':'C:/Users/DELL/Desktop/', 'pbar':True})
    thread_2 = threading.Thread(target= temp_2)
    thread_2.start()
    
    thread_1.join()
    thread_2.join()
    pass

def test_1(IDList, processes= 4):
    pool = Pool(processes= processes)
    args_mode = ('regular', False, '-1', 'C:/Users/DELL/Desktop/')

    tempIDList = ['100176009','97186686','100634717','100612711']
    for i in range(4):
        try:
            image = Image(tempIDList[i])
        except:
            image = Image('100176009')
            pass
        pool.apply_async(image.ImageDownload, args= args_mode)
        pass
    pool.close()
    pool.join()
    
    pass
"""