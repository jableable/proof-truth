# the purpose of this file is to find the longest statement used in all proofs
# this number is used to determine length of embedding vector in statement-embedding.py

import pickle
with open('../Assets/proof_graph.pkl', 'rb') as file:
    raw_data = pickle.load(file)

max = 0

# iterate over all proof steps to find max
for thm in raw_data:
    for num, step in enumerate(raw_data[thm]['x']):

        step_len = len(step[1]['statement'].split())
        if step_len > max:
            rel_step_num = step[1]["num"]
            max = step_len
            thm_num = thm
            step_num = num

print(f"longest statement is {max} chars long")
print(f"it occurs in thm. {thm_num} in proof_graph.pkl at absolute step {step_num} (step {rel_step_num} in MM.exe)")
print(f"this statement is used in the proof of {raw_data[thm_num]['graph_features'][1]}; here's the statement of it:")
print(len(raw_data[thm_num]['x'][step_num][1]['statement']))