import os
import subprocess

import argparse
from pathlib import Path
from config import TIGRESS_HOME # Bring in our good friend
from config import CC # Bring in our good friend
from config import COREUTILS_BINARIES, BINUTILS_BINARIES
from patch_source3 import patch_file

TIGRESS_OBFUSCATION = ("EncodeArithmetic", "AddOpaque", "EncodeBranches", "Virtualize", "Jit", "Flatten") # Some transformations may need some extra work to function properly (e.g., EncodeBranches)

def setup_env():
    os.environ["TIGRESS_HOME"] = TIGRESS_HOME
    os.getenv("TIGRESS_HOME", TIGRESS_HOME)
    og_path = os.getenv("PATH")
    if og_path:
        os.environ["PATH"] = TIGRESS_HOME + os.pathsep + og_path
        os.getenv("PATH", TIGRESS_HOME + os.pathsep + og_path)
    else:
        print("[-] Failed to Set environment!, Something wrong with PATH environment variable!")
        exit(1)


    print("[+] Setup Environment Variable!")
    print(f"PATH: {os.getenv('PATH')}")
    print(f"TIGRESS_HOME: {os.getenv('TIGRESS_HOME')}")
    print(f"CC: {CC}")

def make_build_dir(sub_dir_path, opt):
    print("\t\t[+] Setting up build dir")
    build_dir = os.path.join(sub_dir_path, f"build-{opt}")
    build_dir = Path(build_dir)
    print(f"\t\t[+][+] Build DIR = {build_dir}")
    if not build_dir.exists():
        build_dir.mkdir()
    return build_dir

def gen_obf_source(build_dir, obfuscation, link_contexts, binary, additional = [], opt = "O2"):
    obf_name = f"{binary}_{obfuscation}.c"
    #### Check if obf_name source exists
    #### This is a fix because tigress generated source seems to have issues (reason unknown)
    ### specifically, for lbracket, in function dcnpgettext_expr the declaration of msg_ctxt_id is closed within a block {}.
    ### this causes a variable not declared error during compilation. As a result, this requires manual attention.
    ### We will physically patch such errors in the source.

    source_code_loc = "../src"
    output_loc = "src"
    if "binutils" in str(build_dir):
        source_code_loc = "../binutils"
        output_loc = "binutils"

    if os.path.exists(os.path.join(build_dir, obf_name)):
        print(f"\t\t[+] Skipping as {obf_name} exists")
    else:
        if binary == "ginstall":
            binary = "install"
        elif binary == "sha1sum" or binary == "sha224sum" or binary == "sha256sum" or binary == "sha384sum" or binary == "sha512sum":
            binary = "md5sum"

        if obfuscation == "EncodeArithmetic":
            cmd = ["tigress",
               "--Compiler=gcc",
               f"--Transform={obfuscation}",
               "--Functions=%100" # all fncs
               ]

            cmd = cmd + link_contexts + [ f"{source_code_loc}/{binary}.c", f"--out={obf_name}"]

        elif obfuscation == "Flatten":
            cmd = ["tigress",
               "--Compiler=gcc",
               f"--Transform={obfuscation}",
               "--Functions=%100" # all fncs
               ]
            if binary == "od" or binary == "ptx": # Flatten mishandles goto?
                return

            cmd = cmd + link_contexts + [ f"{source_code_loc}/{binary}.c", f"--out={obf_name}"]
        elif obfuscation == "EncodeBranches":
            cmd = ["tigress",
               "--Compiler=gcc",
               f"--Transform=InitBranchFuns",
               "--Transform=InitOpaque",
               "--Functions=%100", # all fncs
               f"--Transform=AntiBranchAnalysis",
               "--AntiBranchAnalysisKinds=*",
               "--Functions=%100", # all fncs
               "--Exclude=main"
               ]

            cmd = cmd + link_contexts + [ f"{source_code_loc}/{binary}.c", f"--out={obf_name}"]

        elif obfuscation == "Virtualize":
            cmd = ["tigress",
               "--Compiler=gcc",
               f"--Transform={obfuscation}",
               "--Functions=%100" # all fncs
               ]

            if binary == "od" or binary == "ptx" or binary == "stat": # Flatten mishandles goto?
                return

            cmd = cmd + link_contexts + [ f"{source_code_loc}/{binary}.c", f"--out={obf_name}"]

        elif obfuscation == "AddOpaque":
            cmd = ["tigress",
               "--Compiler=gcc",
               "--Transform=InitEntropy",
               "--Transform=InitOpaque",
               "--InitOpaqueStructs=list",
               "--Functions=%100", # all fncs
               f"--Transform={obfuscation}",
               "--AddOpaqueKinds=junk",
               "--Functions=%100" # all fncs
               ]


            cmd = cmd + link_contexts + [ f"{source_code_loc}/{binary}.c", f"--out={obf_name}"]
        else:
            print(f"Not implemented obfuscation: {obfuscation}")
            return

        print(cmd)
        print(f"\t\t[+] Running: {' '.join(cmd)} ... ")
        output = subprocess.run(cmd, cwd = build_dir)
        print(f"\t\t[+] Finished Running cmd")
        if output.returncode != 0:
            exit(1)

        if "coreutils" in str(build_dir):
            print("\t\t[+] Attempting to patch common error")
            # patch a common error with tigress
            patch_file(os.path.join(build_dir, obf_name))

    if os.path.exists(os.path.join(build_dir, "src", obf_name.replace('.c', '.o'))):
        print(f"\t\t[+] Skipping as {obf_name.replace('.c', '.o')} exists")
    else:
        # compile (genrate *.o)
        cmd = [CC] + link_contexts + ["-g", f"-{opt}", "-MT", f"{output_loc}/{obf_name.replace('.c', '.o')}", 
                               "-MD", "-MP", "-MF", f"{obf_name.replace('.c', '')}.Tpo", "-c", "-o", f"{output_loc}/{obf_name.replace('.c', '.o')}",
                               obf_name]
        print(f"\t\t[+] Running: {' '.join(cmd)} in {build_dir} ")
        output = subprocess.run(cmd, cwd = build_dir)
        print(f"\t\t[+] Finished Running cmd")
        if output.returncode != 0:
            exit(1)

    if os.path.exists(os.path.join(build_dir, "src", obf_name.replace('.c', ''))):
        print(f"\t\t[+] Skipping as {obf_name.replace('.c', '')} exists")
    else:
        # compile (genrate binary)
        cmd = [CC, "-g", f"-{opt}", "-Wl,--as-needed", "-o", f"{output_loc}/{obf_name.replace('.c', '')}",  f"{output_loc}/{obf_name.replace('.c', '.o')}"] + additional
        print(f"\t\t[+] Running: {' '.join(cmd)} in {build_dir} ")
        output = subprocess.run(cmd, cwd = build_dir)
        print(f"\t\t[+] Finished Running cmd")
        if output.returncode != 0:
            exit(1)


def compile_manager_gnu(sub_dir_path, binary_list):
    opts = ['O0', 'O1', 'O2', 'O3']

    for opt in opts:
        build_dir = make_build_dir(sub_dir_path, opt)

        cmd = [sub_dir_path + "/configure", f"CC={CC}", f"CFLAGS='-g -{opt}'"]
        print(f"\t\t[+] Running: {' '.join(cmd)} ... ")

        # Commented out for debugging. It runs every single time! So slow! Uncomment this for later.
        #output = subprocess.run(cmd, cwd = build_dir, check=True, shell=True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        print(f"\t\t[+] Finished Running cmd")

        cmd = ["make", "-j96"]
        print(f"\t\t[+] Running: {' '.join(cmd)} ... ")
        #output = subprocess.run(cmd, cwd = build_dir, check=True, shell=True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        #output = subprocess.run(cmd, cwd = build_dir, check=True, shell=True)
        print(f"\t\t[+] Finished Running cmd")

        # Do Tigress, Obfuscated source generation
        for obfuscation in TIGRESS_OBFUSCATION:
            for binary in binary_list:
                #gen_obf_source(build_dir, obfuscation, ["-Ilib", "-Isrc", "-I.","-I../lib", "-I../src", "-I./src", "-I./lib"], binary)
                additional = ["src/libver.a", "lib/libcoreutils.a", "-ldl"]
                if "binutils" in str(build_dir):
                    additional = []
                if binary == "chcon" or binary == "mkfifo" or binary == "mknod" or binary == "stat":
                    additional = additional + ['-lselinux']
                elif binary == "chgrp" or binary == "chown":
                    additional =  ['src/chown-core.o'] + additional
                elif binary == "cp":
                    additional = ['src/copy.o', 'src/cp-hash.o'] + additional
                    additional = additional + ['-lselinux']
                elif binary == "expr" or binary == "factor":
                    additional = additional + ['-lgmp']
                elif binary == "ginstall":
                    additional = ['src/ginstall-cp-hash.o', 'src/ginstall-copy.o', 'src/ginstall-prog-fprintf.o'] + additional
                    additional = additional + ['-lselinux']
                elif binary == "groups":
                    additional =  ['src/group-list.o'] + additional
                elif binary == "id":
                    additional = additional + ['-lselinux']
                    additional =  ['src/group-list.o'] + additional
                elif binary == "kill":
                    additional =  ['src/operand2sig.o'] + additional
                elif binary == "ls":
                    additional = additional + ['-lselinux']
                    additional =  ['src/ls-ls.o'] + additional
                elif binary == "mkdir":
                    additional = additional + ['-lselinux']
                    additional =  ['src/prog-fprintf.o'] + additional
                elif binary == "mv":
                    additional = additional + ['-lselinux']
                    additional =  ['src/remove.o', 'src/cp-hash.o', 'src/copy.o'] + additional
                elif binary == "rm":
                    additional =  ['src/remove.o'] + additional
                elif binary == "rmdir":
                    additional =  ['src/prog-fprintf.o'] + additional
                elif binary == "runcon":
                    additional = additional + ['-lselinux']
                elif binary == "timeout":
                    additional =  ['src/operand2sig.o'] + additional
                elif binary == "uname":
                    additional =  ['src/uname-uname.o'] + additional
                elif binary == "su":
                    additional = additional + ['-lcrypt']



                link_flags = ["-I./lib", "-I./src", "-I../lib", "-I../src", "-I."]
                if "binutils" in str(build_dir):
                    link_flags = ["-I./binutils", "-I../binutils", "-I../include", "-I./bfd", "-I../bfd"]
                if binary == "md5sum":
                    link_flags = link_flags + ['-DHASH_ALGO_MD5']
                elif binary == "sha1sum":
                    link_flags = link_flags + ['-DHASH_ALGO_SHA1']
                elif binary == "sha224sum":
                    link_flags = link_flags + ['-DHASH_ALGO_SHA224']
                elif binary == "sha256sum":
                    link_flags = link_flags + ['-DHASH_ALGO_SHA256']
                elif binary == "sha384sum":
                    link_flags = link_flags + ['-DHASH_ALGO_SHA384']
                elif binary == "sha512sum":
                    link_flags = link_flags + ['-DHASH_ALGO_SHA512']
                gen_obf_source(build_dir, obfuscation, link_flags, binary, additional = additional, opt = opt)

def compile_manager_miniweb(sub_dir_path):

    pass

 
def compile_manager(sub_dir_path):
    if "coreutils" in sub_dir_path:
        compile_manager_gnu(sub_dir_path, COREUTILS_BINARIES)
#    elif "binutils" in sub_dir_path:
#        compile_manager_gnu(sub_dir_path, BINUTILS_BINARIES)
    elif "miniweb" in sub_dir_path:
        compile_manager_miniweb(sub_dir_path)
    else:
        print(f"\t\t[-] Currently no compile script for: {sub_dir_path}")

if __name__ == "__main__":
    print("[+] Hello, Build Script will begin now...")
    setup_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True, type=str, help="Dataset directory with packages as subdirectories")

    args = parser.parse_args()

    print("[+] Beginning to Parse through subdirectory")
    for sub_dir in os.listdir(args.directory):
        print(f"\t[+] Compiling package {sub_dir}")
        sub_dir_path = os.path.join(args.directory, sub_dir)
        sub_dir_path = os.path.abspath(sub_dir_path)
        compile_manager(sub_dir_path)

