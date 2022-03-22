import hashlib
import time


def MD5(str1):
    hl = hashlib.md5()
    hl.update(str1.encode(encoding='utf-8'))
    return hl.hexdigest()


def run1(n):
    with open('md5.txt','w') as f:
        count = 0
        a = time.time()
        time_count = 1
        md5 = MD5(n)
        f.write(md5+'\n')
        for i in range(sum):
            count += 1
            md5 = MD5(md5)

            if n == md5:
                print(md5)
                break
            if count % 100000 == 0:
                print("progress:{:.2f}%,{:.4f}items/s".format(100*count/sum,100000 * time_count / (time.time() - a)))
                f.write(md5 + '\n')
                time_count += 1
        print(md5)
        f.close()
global sum
sum=12*10**9
a = time.time()
run1('f57f2ea3634a0c4673f952d92b23f014')
time=time.time()-a
print("run time is:{:.2f}s".format(time))
print("average speed is:{:.2f}items/s".format(sum/time))
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
