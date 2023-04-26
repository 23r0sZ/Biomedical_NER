import re

text = "accuracy:  99.00%; precision:  87.79%; recall:  94.80%; FB1:  91.13%\nChemical: precision:  87.79%; recall:  94.80%; FB1:  91.16%  5774"

fb1_pattern = r"FB1:\s+(\d+\.\d+)%"
fb1_match = re.search(fb1_pattern, text)

if fb1_match:
    fb1_score = fb1_match.group(1)
    print("FB1 score:", fb1_score)
else:
    print("No FB1 score found.")