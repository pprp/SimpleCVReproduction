import math
import json
import scipy.stats


def pearson(vector1, vector2):
    n = len(vector1)
    # simple sums
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))
    # sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    # sum up the products
    p_sum = sum([vector1[i]*vector2[i] for i in range(n)])
    # 分子num，分母den
    num = p_sum - (sum1*sum2/n)
    den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den


def kendalltau(vector1, vector2):
    tau, p_value = scipy.stats.kendalltau(vector1, vector2)
    return tau


if __name__ == "__main__":
    json1_path = "eval/eval-final.json"
    json_target = "data/benchmark.json"

    f1 = open(json1_path, "r")
    f2 = open(json_target, "r")

    f1_dict = json.load(f1)
    f2_dict = json.load(f2)

    f1_list = []
    f2_list = []

    for k, v in f1_dict.items():
        f1_list.append(v["acc"])

    for k, v in f2_dict.items():
        f2_list.append(float(v["acc"]))

    print(f1_list, f2_list)

    print("person:", pearson(f1_list, f2_list))
    print("kendall", kendalltau(f1_list, f2_list))

    f1.close()
    f2.close()
