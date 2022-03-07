from glob import glob
from sys import argv
from os.path import exists
from os import system

path = argv[1]

folders = glob(path + '/*')


def checkFolder(f):
    for fold in range(5):
        if not exists('%s/predictions-%d.csv.gz' % (f, fold)):
            return False
        if not exists('%s/validation-%d.csv.gz' % (f, fold)):
            return False
    return True


tc = 0
fc = 0
fcf = []
dns = []
for f in folders:
    if checkFolder(f):
        tc += 1
    else:
        fc += 1
        fcf.append(f)
        dns.append(f.split('/')[-1])
        print(f.split('/')[-1])

print('Finished: %d.' % tc)
print('Not done: %d...' % fc)

if len(argv) == 3 and argv[2] == 'resub':
    print('Resubmitting...')
    for fn, dn in zip(fcf, dns):
        system('sh run.sh %s %s' % (fn, dn))
