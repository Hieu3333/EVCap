import json
import os
with open('eval_log/nocaps/overall_generated_captions.json','r', encoding='utf-8') as file:
    data = json.load(file)
indomain = []
neardomain = []
outdomain = []

i = 0
n = 0
o = 0

for item in data:
    if item['split'] == 'in-domain':
        indomain.append(item)
        i+=1
    if item['split'] == 'out-domain':
        outdomain.append(item)
        o+=1
    if item['split'] == 'near-domain':
        neardomain.append(item)
        n+=1

print(f"in {i}")
print(f"out {o}")
print(f"near {n}")
print(f"sum {i+o+n}")

out_path = 'eval_log/nocaps'
if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)


with open(os.path.join(out_path, f'indomain_generated_captions.json'), 'w') as outfile:
    json.dump(indomain, outfile, indent = 4)
with open(os.path.join(out_path, f'neardomain_generated_captions.json'), 'w') as outfile:
    json.dump(neardomain, outfile, indent = 4)
with open(os.path.join(out_path, f'outdomain_generated_captions.json'), 'w') as outfile:
    json.dump(outdomain, outfile, indent = 4)

