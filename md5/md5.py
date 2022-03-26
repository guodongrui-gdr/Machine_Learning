import hashlib
import time
from multiprocessing import Process


def MD5(str1):
    hl = hashlib.md5()
    hl.update(str1.encode(encoding='utf-8'))
    return hl.hexdigest()


def run1(i):
    with open('md5.txt', 'w') as f:
        count = 15355554624
        count0 = 1
        a = time.time()
        for j in range(i + count, i + count + int(k / threads)):
            count += 1
            b = str(hex(j))[2:]
            md5 = MD5(b)
            if b == md5:
                print(md5)
                break
            if j % 100000 == 0:
                print("{:.4f}items/s".format(100000 * threads * count0 / (time.time() - a)))
                count0 += 1
                f.write(str(count) + '\n')
        f.close()


global threads, k
threads = 10
k = 10 ** 7
if __name__ == '__main__':

    process_list = []
    j = []
    a = time.time()
    for i in range(threads):
        j.append(16 ** 31 + 1 + int(i * (16 ** 32 - 16 ** 31) / threads))
        p = Process(target=run1, args=(j[i],))
        p.start()
        process_list.append(p)
    for i in process_list:
        p.join()
    t = time.time() - a
    print('cost time is:{:.2f}'.format(t))
    print('average speed is:{:.2f}items/s'.format(k / t))
