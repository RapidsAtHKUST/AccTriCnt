# import commands
import os
import time
import logging
import sys

my_splitter = '-'.join(['*' for _ in range(20)])


def append_header(file_path):
    os.system(' '.join(
        ['echo', my_splitter + time.ctime() + my_splitter, '>>', file_path]))


def get_logger(filepath, name=__name__):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=filepath)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger = logging.getLogger(name)
    return logger


# def kill_all(exec_name_lst):
#     for exec_name in exec_name_lst:
#         err_code, output = commands.getstatusoutput("ps -ef | grep " + exec_name + " | awk '{print $2}'")
#         print [err_code, 'kill all' + output]
#         for pid in output.strip().split('\n'):
#             os.system('kill -9 ' + pid)
#     time.sleep(5)


def write_split(statistics_file_path):
    with open(statistics_file_path, 'a+') as ifs:
        ifs.write(my_splitter + my_splitter + '\n')
        ifs.write(my_splitter + my_splitter + '\n')

# def signal_handler(signal, frame):
#     print 'You pressed Ctrl+C!'
#     kill_all()
#     sys.exit(0)
