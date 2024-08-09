# run on var.csv to create vocab.txt with vocabulary

import re

with open('./var.csv', 'r') as input, open('../vocab.txt', 'w') as output:
        for line in input:
            new_line = re.sub(r'\$[a-z]\s|\s\$.', '', line) # get rid of $c, $v  
            unquoted_line = re.findall(r'(?<=^\").+(?=\"$)',new_line)   # get rid of outer quotes around 
                                                                        # special chars like / or (

            if len(unquoted_line)==0:   # if line had no outer quotes, there aren't duplicate double-quotes
                for char in new_line.split():
                    output.write(char+"\n") # char is written if there are no outer quotes
            
            if len(unquoted_line)>0:    #if line had outer quotes, remove duplicate double-quotes
                for char in unquoted_line[0].split():
                    deduplicated_quotes_char = re.sub(r'\"\"','\"',char)
                    output.write(deduplicated_quotes_char+"\n")
