import re
import json

all_labels = set()

def process_resume(in_path, out_path):
  with open(in_path, "r") as fp:
    data = fp.read().strip()
  data = re.split("\n\n", data)
  fp = open(out_path, "w", encoding="utf-8")
  for i in range(len(data)):
    details = data[i].split("\n")
    text = []
    labels = []
    output = {}
    output["id"] = i - 1
    for detail in details:
      tmp = detail.split(" ")
      word = tmp[0]
      label = tmp[-1]
      if "-" in label:
        rel_label = label.split("-")
        if len(rel_label) == 2:
          all_labels.add(rel_label[-1])
      text.append(word)
      if "S-" in label:
        label = label.replace("S-", "B-")
      elif "M-" in label:
        label = label.replace("M-", "I-")
      elif "E-" in label:
        label = label.replace("E-", "I-")
      labels.append(label)
    output["text"] = text
    if text == [""] or text == [" "]:
      continue
    output["labels"] = labels
    assert len(text) == len(labels)
    fp.write(json.dumps(output, ensure_ascii=False) + "\n")
  fp.close()

process_resume("data/resume/raw_data/train.char.bmes", "data/resume/mid_data/train.txt")
process_resume("data/resume/raw_data/dev.char.bmes", "data/resume/mid_data/valid.txt")
process_resume("data/resume/raw_data/test.char.bmes", "data/resume/mid_data/test.txt")

with open("data/resume/mid_data/labels.txt", "w") as fp:
  fp.write("\n".join(list(all_labels)))