import re
import sys


def normalize(text):
    # Strip leading/trailing spaces on each line, join with '\n'
    return "\n".join(line.strip() for line in text.strip().splitlines())

def patch_file(filename):
    with open(filename, 'r') as f:
        content = f.read()

    norm_content = normalize(content)

    # Pattern matches:
    # { { <up to 30 lines> char msg_ctxt_id[msgctxt_len + msgid_len] ;
    pattern = re.compile(
        r'(size_t msgctxt_len ;\n\s*'
        r'size_t tmp ;\n\s*'
        r'size_t msgid_len ;\n)'      # capture the 3 size_t lines
        r'((?:.*?\n)*?)'              # non-greedy match of lines in between
        r'(char msg_ctxt_id\[msgctxt_len \+ msgid_len\] ;)',  # the target line
        re.MULTILINE
    )
    
    def replacer(m):
        decl_block = m.group(1)
        middle = m.group(2)
        msg_line = m.group(3)
        return decl_block + msg_line + '\n' + middle

    new_content, count = pattern.subn(replacer, norm_content)

    if count > 0:
        with open(filename, 'w') as f:
            f.write(new_content)
        print(f"{count} patch(es) applied to '{filename}'.")
    else:
        print("⚠️ No matching blocks found — nothing changed.")
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <file.c>")
        sys.exit(1)
    patch_file(sys.argv[1])
