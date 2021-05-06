import os
from glob import glob
from distutils.dir_util import copy_tree

base = '/media/data/dirty_pix_v3/validation_RAW/'
root = base + 'RAW_human_ISO8000_EXP10000'
target = base + 'RAW_synset_ISO8000_EXP10000'

human_labels = glob(os.path.join(root,'*/'))
human_labels = [label.split('/')[-2] for label in human_labels]
print(human_labels)

human_to_synset = {}
with open('raw_metadata.txt', 'r') as synset_human_file:
    for line in synset_human_file:
	synset = line[:9]
	human = line[9:].strip().lower()
        for label in human_labels:
            for match in human.split(','):
                if label.strip() == match.strip().lower():
                   human_to_synset[label] = synset

print(human_to_synset)
missing = False
for h in human_labels:
   if h not in human_to_synset:
    print h
    missing = True
if missing:
    print("Missing synsets!")
else:
    print("All synsets mapped!")

#print len(human_labels)
#print len(human_to_synset)

all_dirs = glob(os.path.join(root,'*/'))
for subdir in all_dirs:
    no_imgs = len(glob(os.path.join(subdir, '*.dng')))
    if not no_imgs:
        print(subdir + " is empty")
        continue

    subdir = subdir[len(root)+1:-1]
    print(subdir)

    if subdir not in human_to_synset:
        print("Skipping %s"%subdir)
        continue

    print("Copying %d files from class %s"%(no_imgs, subdir))

    synset = human_to_synset[subdir]
    new_dir = os.path.join(target, synset)
    old_dir = os.path.join(root,subdir)
    copy_tree(old_dir, new_dir)
