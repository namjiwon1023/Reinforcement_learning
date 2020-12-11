import multiprocessing as mp
import time


def job(q):
    res = 0
    for i in range(1000):
        res += i+i**2+i**3
    # queue
    q.put(res)    # 将 需要获得的值 放进queue的队列中 相当于正常情况下的return

def multicore_pool():
    pool = mp.Pool()     # 函数定义参数（processes=自己定义的进程数）默认是使用全部的核数的 
    res1 = pool.map(job,range(10))  # 运行的函数 ， 运算的值 ， 可以在定义中使用return
    print(res1)
    res1 = pool.apply_async(job,(2,))   # 输入的数值是可以迭代的所以要添加‘，’ 这个函数 一次只能在一个进程中运行
    print(res1.get())
    multi_res = [pool.apply_async(job,(i,)) for i in range(10)]    # 迭代器的形式
    print([res1.get for res1 in multi_res])

def job1(x):
    return x*x

def shared_memory():
    value = mp.Value('d', 1)      # 'i' : int , 'd' float
    array = mp.Array('i', [1,2,3])  # 只能一维
    return array,value
def job2(v, num, l):
    l.acquire()
    for _ in range(10):
        time.sleep(0.1)
        v.value += num         # 如果共享内存的数值要进行取值运算的话 用 .value的形式
        print(v.value)
    l.release()

def multicore():
    l = mp.Lock()                 # 锁完内存之后 先进行process1 的运行 ， 在运行之后 在进行process2的运行
    v = mp.Value('i' , 0)
    p1 = mp.Process(target=job2,args=(v, 1 , l ))  # 即使 只有一个输入值 也得在数值之后 进行 ‘，’的添加
    p2 = mp.Process(target=job2,args=(v, 3 , l ))
    p1.start()  # 进程开始运行
    p2.start()
    p1.join()  #  阻断 ，在 进行join的进程运行完了之后 继续下一个进程
    p2.join()



if __name__ == '__main__':
    # q = mp.Queue()
    # # 调用的函数不需要括号
    # p1 = mp.Process(target=job,args=(q, ))  # 即使 只有一个输入值 也得在数值之后 进行 ‘，’的添加
    # p2 = mp.Process(target=job,args=(q, ))
    # p1.start()  # 进程开始运行
    # p2.start()
    # p1.join()  #  阻断 ，在 进行join的进程运行完了之后 继续下一个进程
    # p2.join()

    # res1 = q.get()
    # res2 = q.get()   # 获取 queue队列中的运算结果
    # print(res1+res2)
    # print(res1==res2)
    multicore()
    ''' queue : 多核（每一个核）运算的结果 放在队列当中 然后等到所有进程或所有核都运行完了之后 再从这个队列中取出 继续的进行下一步运算
        在 需要多核运算的函数中 不能有return的值的 取而代之的是 queue'''
    '''进程池 pool 把你所有要运行的程序都放在一个池子里 python会帮你分配进程和结果'''
    '''shared memory : 共享 内存 可以定义 进程间 共享的 数值 或 列表 '''

    ''' Simple multi-process example'''
    '''
        import torch.multiprocessing as mp
        from model import MyModel

        def train(model):
            # Construct data_loader, optimizer, etc.
            for data, labels in data_loader:
                optimizer.zero_grad()
                loss_fn(model(data), labels).backward()
                optimizer.step()  # This will update the shared parameters

        if __name__ == '__main__':
            num_processes = 4
            model = MyModel()
            # NOTE: this is required for the ``fork`` method to work
            model.share_memory()
            processes = []
            for rank in range(num_processes):
                p = mp.Process(target=train, args=(model,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()'''
    '''
        Shared optimizer, the parameters in the optimizer will shared in the multiprocessors.


        import torch


        class SharedAdam(torch.optim.Adam):
            def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                        weight_decay=0):
                super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
                # State initialization
                for group in self.param_groups:
                    for p in group['params']:
                        state = self.state[p]
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)

                        # share in memory
                        state['exp_avg'].share_memory_()
                        state['exp_avg_sq'].share_memory_()
        '''