import os
import sys

os.system("git add *.py")
os.system("git add *.sbatch")

os.system("git commit -m \"%s\"" % sys.argv[1])
