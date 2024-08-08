import re
from collections import defaultdict
import pandas as pd

def single_line_processor(line,tags,var):
    if '$p' in line:

        match = re.match(r'^(.*?\S)\s(\$.*?\$)\=\s\((.*?)\)', line)
        tag = match.group(1).strip()
        if tag in tags:
            print(f"Double Tag Error {tag}")  
            return -1       
        tags[tag] = {}
        tags[tag]['statement'] = match.group(2).strip()
        ant = match.group(3).strip().split(' ')
        if ant:
            tags[tag]['antecedent_tag'] = ant
    else:
        matches = re.findall(r'\$.*?\$\.*', line)
        if matches:
            for match in matches:
                line = line.replace(match, '')
            tag = line.strip()
            if tag == '':
                for match in matches:
                    var.add(match)
            else:
                if len(matches) == 1:
                    if tag in tags:
                        print(f"Double Tag Error {tag}")  
                        return -1         
                    tags[tag] = {}
                    tags[tag]['statement'] = matches[0]
                else:               
                    print(f"Exception1: Line does not match expected patterns - {line}")
        else:

            print(f"Exception1: Line does not match expected patterns - {line}")



def multiple_line_processor(lines,tags,sub_statement,sub_var):
    new_statement = {}
    for s in sub_statement:
        new_statement[s] = sub_statement[s]
    new_var = set()
    for v in sub_var:
        new_var.add(v)
    i = 0

    while i < len(lines):
        line = lines[i]
        if re.match(r'^\s*\${', line):              # treat ${     $} as a block
            start_indent = re.match(r'^(\s*)\${', line).group(1)
            multiline_block = []
            i += 1
            while i < len(lines) and not re.match(rf'^{start_indent}\$}}', lines[i]):
                multiline_block.append(lines[i])
                i += 1
            multiple_line_processor(multiline_block,tags,new_statement,new_var)
        else:   
            c = line.count('$')
            if c in (3,5,7,9): 
                print(f"Exception2: Line does not match expected patterns - {lines[i]}")
                continue
            if c == 1 and '$p' not in line:             #a very long statement can take several lines
                combined_line = line
                i += 1
                while i < len(lines) and '$.' not in lines[i]:
                    combined_line += lines[i]
                    i += 1
                if i < len(lines):
                    combined_line += lines[i]
                line = re.sub(r'\s+',' ',combined_line)
            else:
                line = lines[i]


            if '$p' in line:
                combined_line = line
                i += 1
                while i < len(lines) and '$.' not in lines[i]:
                    combined_line += lines[i]
                    i += 1
                combined_line += lines[i]
                combined_line = re.sub(r'\s+',' ',combined_line.strip())
                match = re.match(r'^(.*?\S)\s(\$.*?\$)\=.*?\((.*?)\)', combined_line)
                tag = match.group(1).strip()
                if tag in tags:
                    print(f"Double Tag Error {tag}")  
                    continue          
                tags[tag] = {}
                tags[tag]['statement'] = match.group(2).strip()
                ant = match.group(3).strip().split(' ')
                if ant and ant != ['']:
                    tags[tag]['antecedent_tag'] = ant
                if new_statement:
                    tags[tag]['hypothesis_s'] = new_statement
                if new_var:
                    tags[tag]['hypothesis_v'] = new_var 
            else:
                single_line_processor(line,new_statement,new_var)

        i += 1    
    #print(new_statement)


#cleaned_set.mm     test2.txt
with open('cleaned_set.mm', 'r') as file:
    lines = file.readlines()

tags = {}
var = set()
i = 0

while i < len(lines):
    line = lines[i][:-1]
    if re.match(r'^\s*\${', line):              # treat ${     $} as a block
        start_indent = re.match(r'^(\s*)\${', line).group(1)
        multiline_block = []
        i += 1
        while i < len(lines) and not re.match(rf'^{start_indent}\$}}', lines[i]):
            multiline_block.append(lines[i][:-1])                                       #[:-1] delete \n
            i += 1
        multiple_line_processor(multiline_block,tags,{},set())
    else: 
        if '$p' in line:
            combined_line = line
            i += 1
            while i < len(lines) and '$.' not in lines[i]:
                combined_line += lines[i]
                i += 1
            combined_line += lines[i]
            combined_line = re.sub(r'\s+',' ',combined_line.strip())  
            single_line_processor(combined_line,tags,var)
        else:
            c = lines[i].count('$')
            if c in (3,5,7,9): 
                print(f"Exception3: Line does not match expected patterns - {lines[i]}")
                continue
            if c == 1:             #a very long statement can take several lines
                combined_line = lines[i]
                i += 1
                while i < len(lines) and '$.' not in lines[i]:
                    combined_line += lines[i]
                    i += 1
                combined_line += lines[i]
                combined_line = re.sub(r'\s+',' ',combined_line)
                single_line_processor(combined_line,tags,var)
            else:
                single_line_processor(line,tags,var)


    i += 1

tag_list = []
for key, value in tags.items():
    value['tag'] = key
    tag_list.append(value)

#print(tag_list)

df = pd.DataFrame(tag_list)
df = df[['tag','statement','antecedent_tag','hypothesis_s','hypothesis_v']]
df.to_csv('tag.csv', index=False)
