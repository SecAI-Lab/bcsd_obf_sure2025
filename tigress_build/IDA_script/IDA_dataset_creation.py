import json
import os
import ida_loader
import ida_pro
import ida_bytes
import ida_funcs
import ida_nalt
import idc
import idautils
import ida_gdl
import capstone as cs

def capstone_setup():
    md = cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_64)
    return md


def run(output_dir_name, out_json_name):
    output_dict = dict()
    md = capstone_setup()

    if not os.path.isdir(output_dir_name):
        os.mkdir(output_dir_name)

    for f_ea in idautils.Functions():
        if idc.get_segm_name(f_ea) != ".text":
            continue
        func = ida_funcs.get_func(f_ea)
        addr_func_start, addr_func_end = func.start_ea, func.end_ea

        func_name = ida_funcs.get_func_name(f_ea)
        output_dict[func_name] = list()
        bbs = ida_gdl.FlowChart(func)
        for bb in bbs:
            addr_bb_start, addr_bb_end = bb.start_ea, bb.end_ea
            addr_instr = addr_bb_start
            while addr_instr < addr_bb_end:
                addr_instr_start, addr_instr_end = addr_instr, ida_bytes.get_item_end(addr_instr)
                addr_instr = addr_instr_end

                raw_bytes = ida_bytes.get_bytes(addr_instr_start, addr_instr_end - addr_instr_start)
                #cs_instr = [x for x in md.disasm(raw_bytes, addr_instr_start)][0]
                for cs_instr in md.disasm(raw_bytes, addr_instr_start):
                    mnemonic = cs_instr.mnemonic
                    operands = cs_instr.op_str
                    operands = operands.replace(" + ", "+")
                    operands = operands.replace(" - ", "-")
                    operands = operands.replace(",", "")
                    operands = operands.replace(" ", "_")
                    instr = f"{mnemonic}_{operands}"
                    output_dict[func_name].append(instr)
                #print(instr)
    with open(os.path.join(output_dir_name, out_json_name), "w") as f:
        json.dump(output_dict, f)



if __name__ == "__main__":
    file_name = ida_nalt.get_root_filename()
    print(f"[+] Analyzing {file_name}")
    if not ida_loader.get_plugin_options("output_dir"):
        print(f"[!] output file option is missing :(")
        ida_pro.qexit(1)

    output_dir_name = ida_loader.get_plugin_options("output_dir")
    print(f"OUTPUTFILE: {output_dir_name}")
    out_json_name = f"{file_name}.json"
    run(output_dir_name, out_json_name)

    ida_pro.qexit(0)
