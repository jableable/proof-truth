import re
from collections import defaultdict
import pandas as pd
from metamath import Metamath
import ast




df = pd.read_csv('tag.csv',na_values=['']).iloc[:]
tag_dict = df.to_dict(orient='index')
thrd = 0
metamath = Metamath('metamath/metamath.exe')    #path from this file to metamath.exe
print(metamath.initialize(thrd))
print(metamath.send(thrd,'read "metamath/set.mm"'))
print(metamath.send(thrd,'set height 1000000'))
print(metamath.send(thrd,'set width 1000000'))

tags = {}
error_tag = []
count = 0


for tag in tag_dict:
    tag_name = tag_dict[tag]['tag']

    if tag_name in tags:
        print(f"SameName error {tag_name}")
    if tag_dict[tag]['statement'].startswith('$p'):
        print(count,tag_name,tag_dict[tag]['statement'])
        count += 1
    else:
        continue
    try:
        output = metamath.send(thrd,'show proof '+str(tag_name) +'/lemmon')
    except:
        error_tag.append(tag_name)
        print(f"unknown error:  {tag_name}")
        thrd += 1
        print(metamath.initialize(thrd))
        print(metamath.send(thrd,'read "metamath/set.mm"'))
        print(metamath.send(thrd,'set height 1000000'))
        print(metamath.send(thrd,'set width 100000'))   
        continue
    lines = output.split('\n')[:-1]

    if pd.isna(tag_dict[tag]['hypothesis_s']):
        tag_dict[tag]['hypothesis_s'] = "{}"
    tag_dict[tag]['hypothesis_s'] = ast.literal_eval(tag_dict[tag]['hypothesis_s'])

    tags[tag_name] = {}
    tags[tag_name]['tag'] = tag_name
    tags[tag_name]['proof'] = {}
    tags[tag_name]['node'] = {}
    tags[tag_name]['hpt'] = {}
    proof_raw = []
    for line in lines:
        line = line.strip()
        match = re.match(r'(\d+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\$.*)', line)
        if match:
            step = match.group(1)
            hpt = match.group(3)
            node_tag = match.group(5)
            statement = match.group(9)
            tags[tag_name]['proof'][int(step)] = statement
            if node_tag not in tags[tag_name]['node']:  
                tags[tag_name]['node'][node_tag] ={}
            tags[tag_name]['node'][node_tag][int(step)] = [int(h) for h in hpt.split(',')]
        else:
            match = re.match(r'(\d+)(\s+)(\S+)(\s+)(\S+)(\s+)(\$.*)', line)
            if match:
                if match.group(5).startswith("@"):
                    step = match.group(1)
                    if match.group(3) in tag_dict[tag]['hypothesis_s']:
                        print(1/0)
                        if match.group(3) in tags[tag_name]['hpt']:
                            tags[tag_name]['hpt'][match.group(3)].append(int(step))
                        else:
                            tags[tag_name]['hpt'][match.group(3)] = [int(step)]
                    statement = match.group(7)
                    tags[tag_name]['proof'][int(step)] = statement
                else:
                    step = match.group(1)
                    hpt = match.group(3)
                    node_tag = match.group(5)
                    statement = match.group(7)
                    tags[tag_name]['proof'][int(step)] = statement
                    if node_tag not in tags[tag_name]['node']:  
                        tags[tag_name]['node'][node_tag] ={}
                    tags[tag_name]['node'][node_tag][int(step)] = [int(h) for h in hpt.split(',')]
            else:
                match = re.match(r'(\d+)(\s+)(\S+)(\s+)(\$.*)', line)
                if not match:
                    print(f"Matching Error {line}")
                    error_tag.append(tag_name)         
                    continue
                step = match.group(1)
                if match.group(3).startswith("@"):
                    statement = match.group(3)
                else:
                    label = match.group(3)
                    if label in tag_dict[tag]['hypothesis_s']:
                        if match.group(3) in tags[tag_name]['hpt']:
                            tags[tag_name]['hpt'][match.group(3)].append(int(step))
                        else:
                            tags[tag_name]['hpt'][match.group(3)] = [int(step)]
                    statement = match.group(5)
                tags[tag_name]['proof'][int(step)] = statement


    if error_tag and tag_name == error_tag[-1]:
        continue


print(f"Error Tag: {error_tag}")
df_out = pd.DataFrame(tags.values())
df_out.to_csv('tag_proof.csv', index=False)
    

