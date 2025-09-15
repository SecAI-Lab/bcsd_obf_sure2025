import config
import subprocess
import os
import argparse
from os import getenv
from os import path

IDA_PATH = getenv("IDA_PATH", config.IDA_PATH)
IDA_PLUGIN = path.join(path.dirname(path.abspath(__file__)), 'IDA_dataset_creation.py')
IDA_LOG_PATH = "dataset_creation_log.txt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', required=True, type=str,
                        help = "Path to IDB directory")
    args = parser.parse_args()
    idb_dir = args.data
    failed_list = []
    failed_count = 0
    succ_count = 0

    try:
        if not path.isfile(IDA_PATH):
            print(f"[!] Error: IDA_PATH:{IDA_PATH} is not valid!")
            exit(1)

        if path.islink(idb_dir):
            print('is symbolic?')
            idb_dir = path.realpath(idb_dir)
        print(idb_dir)

        for file in os.listdir(idb_dir):
            _, ext = path.splitext(file)
            if ext == ".asm" or ext == ".nam" or ext == ".id2" or ext == ".id1" or ext == ".id0" or ext == ".til":
                continue

            file = path.join(idb_dir, file)

        
            cmd = [IDA_PATH,
                   '-A',
                   f'-L{IDA_LOG_PATH}',
                   f'-S{IDA_PLUGIN}',
                   f'-Ooutput_dir:output',
                   file]
            print(f"[+] cmd: {cmd}")

            result_file_name = file.split('/')[1]
            result_file_name = result_file_name.strip('.i64')
            result_file_name = result_file_name + ".json"
            output_path = os.path.join("output", result_file_name)
            if os.path.exists(output_path):
                continue
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            if proc.returncode == 0:
                print(f"[+] {file}: success")
                succ_count += 1
            else:
                print(f"[-] {file}: failed")
                failed_count +=1
                failed_list.append(file)


    except Exception as e:
        print(f"[!] Exception in dataset creation!\n {e}")
        exit(1)

    print(f"Failed Count: {failed_count}")
    print(f"Failed list: {failed_list}")

