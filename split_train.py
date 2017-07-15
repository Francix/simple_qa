# split the qa_pair file into train set and valid set

def split():
  fd = open("qa_pairs", "r")
  lines = fd.readlines()
  fd.close()
  total_lines = len(lines)
  train_lines = []
  valid_lines = []
  for i in range(total_lines):
    if((i % (total_lines / 20000) == 0) and (i > 0)):
      valid_lines.append(lines[i])
    else:
      train_lines.append(lines[i])
  fd = open("train", "w")
  for l in train_lines:
    fd.write(l)
  fd.close()
  fd = open("valid", "w")
  for l in valid_lines:
    fd.write(l)
  fd.close()
  print("%d train sentences, %d valid sentences, ratio = %f" 
    % ( len(train_lines), 
    len(valid_lines), 
    float(len(valid_lines)) / float(len(train_lines)) ) )
  return 

def main():
  split()
  return 

if __name__ == "__main__":
  main()
