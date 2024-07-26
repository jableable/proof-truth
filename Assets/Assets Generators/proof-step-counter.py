# the purpose of this program is to count the number of steps in a shortened proof from metamath.exe

# input .txt files should be: 
# 1. output of "show proof INSERT_THM_HERE" in MM; and 
# 2. "show new_proof" in MM-PA after using "prove INSERT_THM_HERE" in MM

import re

# open file
with open('./efgcpbllemb-mm.txt', 'r') as mm, open('./efgcpbllemb-mm-pa.txt', 'r') as mmpa:
        mm_counter = 0
        mmpa_counter = 0
        for line in mm:
            mm_line_num = re.search("^[0-9]+[\s]",line)
            if mm_line_num is not None:
                mm_counter += 1
        for line in mmpa:
            mmpa_line_num = re.search("^[0-9]+[\s]",line)
            if mmpa_line_num is not None:
                mmpa_counter += 1
        print(f"The final numbered step from MM is {mm_line_num.group()}and the total number of steps in MM is {mm_counter}")
        print(f"The final numbered step from MM is {mmpa_line_num.group()}and the total number of steps in MM is {mmpa_counter}")