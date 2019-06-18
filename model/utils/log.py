import sys

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'


def printc(color, msg):
    print(color + msg + ENDC, end='')
    sys.stdout.flush()


def warning(vmsg):
    sys.stdout.write(WARNING + msg + ENDC + '\n')
    sys.stdout.flush()
