# the purpose of this program is to count the number of steps in a shortened proof from metamath.exe

# input .txt files should be: 
# 1. output of "show proof INSERT_THM_HERE" in MM; and 
# 2. "show new_proof" in MM-PA after using "prove INSERT_THM_HERE" in MM

import re

# open files
with open('./efgcpbllemb-mm.txt', 'r') as mm, open('./efgcpbllemb-mm-pa.txt', 'r') as mmpa:
        mm_counter = 0  # counts proof steps from MM file
        mmpa_counter = 0    # counts proof steps from MM-PA file
        for line in mm:
            mm_line_num = re.search("(^((|\s|\s\s|\s\s\s)[0-9]+))(=?(\s+.+?)=)",line)
            if mm_line_num is not None:
                mm_counter += 1
                final_mm_line_num = mm_line_num
        for line in mmpa:
            mmpa_line_num = re.search("(^((|\s|\s\s|\s\s\s)[0-9]+))(=?(\s+.+?)=)",line)
            if mmpa_line_num is not None:
                mmpa_counter += 1
                final_mmpa_line_num = mmpa_line_num
        print(f"The final numbered step from MM is {final_mm_line_num.group(1)} and the total number of steps in MM is {mm_counter}")
        print(f"The final numbered step from MM-PA is {final_mmpa_line_num.group(1)} and the total number of steps in MM-PA is {mmpa_counter}")