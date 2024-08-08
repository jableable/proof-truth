# run on var.csv to create .txt with vocabulary

import re

with open('./var.csv', 'r') as input, open('../vocab.txt', 'w') as output:
        #text = input.read()
        #new_text = re.sub(r'\$[a-z]\s|\s\$.', '', text)
        for line in input:
            new_line = re.sub(r'\$[a-z]\s|\s\$.', '', line)
            output.write(new_line)