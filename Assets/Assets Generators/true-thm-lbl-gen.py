#removing comments makes set.mm easier to parse; numbering theorems makes matching up with website easier
import re

#remove comments from set.mm
with open('../set.mm', 'r') as input, open('./set_with_no_comments.txt', 'w') as output:
        text = input.read()
        new_text = re.sub(r'(\$\([\s\S]*?\$\))', '', text)
        output.write(new_text)

#remove blank lines from previously generated file
with open('./set_with_no_comments.txt', 'r') as input, open('./set_with_no_comments_or_blank_lines.txt', 'w') as output:        
    for line in input:  
        if not line.isspace():
            output.write(line)

#number theorems from previously generated file
j=1
with open('./set_with_no_comments_or_blank_lines.txt', 'r') as input, open('./numbered_set.txt', 'w') as output:     
    for line in input:  #number theorems
        new_line = ""
        for i, char in enumerate(line):
            if line[i] in ["a","p"]:
                if line[i-1] == "$":
                    new_line += line[i]+str(j)
                    j += 1
                    continue
            new_line += char
        output.write(new_line)
                      
#check counts for sanity (should agree with website, e.g. 45330)            
with open('./numbered_set.txt', 'r') as input:
    text = input.read()
    thms= re.findall("\$a|\$p",text)             
    print("There are", len(thms), "theorems in numbered_set.txt")

#extract labels of theorems
with open('./numbered_set.txt', 'r') as input, open('../true_labels.txt', 'w') as output:     
    for line in input:
        label = re.search("\S+(?=\s\$a|\s\$p)",line)
        if label is not None:
            output.write(label.group()+"\n")
    output.write("$e"+"\n") # extra assumption label
    output.write("$a") # extra axiom label