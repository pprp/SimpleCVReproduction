import yaml 

f = open("configs/test.yml", "r")

y = yaml.load(f, Loader=yaml.FullLoader)

print(y)