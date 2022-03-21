import hashlib
import time


def MD5(str1):
    hl = hashlib.md5()
    hl.update(str1.encode(encoding='utf-8'))
    return hl.hexdigest()


def run1(n):
    count = 0
    a = time.time()
    time_count = 1
    md5 = MD5(n)
    for i in range(5 * 10 ** 8):
        count += 1
        md5 = MD5(md5)
        if n == md5:
            print(md5)
            break
        if count % 10000 == 0:
            print("{:.4f}items/s".format(10000 * time_count / (time.time() - a)))
            time_count += 1
    print(md5)


a = time.time()
run1('1c84960824e1d1392dc9a3a87d284bee')
print("run time is:{:.2f}s".format(time.time() - a))
'''


if __name__ == '__main__':
    process_list = []
    j = []
    a = time.time()
    for i in range(threads):
        j.append(16 ** 31 + 1 + int(i * 10 ** k / threads))
        p = Process(target=run1, args=(j[i],))
        p.start()
        process_list.append(p)
    for i in process_list:
        p.join()
    print(time.time() - a)
'''
