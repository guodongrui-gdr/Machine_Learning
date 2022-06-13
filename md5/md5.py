import hashlib
import time
from multiprocessing import Process, Pool


def MD5(str1):
    hl = hashlib.md5()
    hl.update(str1.encode(encoding='utf-8'))
    return hl.hexdigest()


def run1(i):
    count = 32494986783
    f = open('md5.txt', 'w')
    for j in range(i + count, i + count + int(k / threads)):
        count += 1
        b = str(hex(j))[2:]
        md5 = MD5(b)
        if b == md5:
            print(md5)
            break
        if count % 10 ** 6 == 0:
            f.write(str(count) + '\n')
    f.close()
    return count


def run__pool():  # main process
    process_args = []
    for i in range(threads):
        process_args.append(16 ** 31 + 1 + int(i * (16 ** 32 - 16 ** 31) / threads))
    start_time = time.time()
    with Pool(threads) as p:
        outputs = p.map(run1, process_args)
    print(
        f'outputs:{outputs[0]}\nTimeUsed:{time.time() - start_time:.2f}\naverage speed:{k / (time.time() - start_time):.2f}')


global k, threads
threads = 16
k = 10 ** 8

if __name__ == '__main__':
    run__pool()
