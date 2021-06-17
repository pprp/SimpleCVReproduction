import json

f1 = open("acc_track_part1.json", "r")

d1 = json.load(f1)

f2 = open("acc_track_part2.json", "r")

d2 = json.load(f2)

f3 = open("acc_track_part3.json", "r")

d3 = json.load(f3)


d_a = {**d1,**d2,**d3}

f_a = open("results.json","w")

json.dump(d_a, f_a)