import os

for i in os.listdir("./data/images/images"):
    new_dir = os.path.join("./data/images/images", i)
    for j in os.listdir(new_dir):
        if j.startswith("."):
            print(j)
            full_path = os.path.join(new_dir, j)
            os.remove(full_path)