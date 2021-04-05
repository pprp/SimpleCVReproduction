import random
import copy

def cal_diff(l_a,l_b):
    count=0
    res=[]
    for idx, item in enumerate(l_a):
        if item!=l_b[idx]:
            count+=1
            res.append(idx)
    return count,res



def check_exist(l,off_list):
    if l in off_list:
        return True
    else:
        return False


def cross(list_a,list_b,res):
    size = len(list_a)
    if len(res)==2:
        low=res[0]
        high=low
    else:
        low = random.randint(0, res[-1])
        high = random.randint(res[0], size - 1)
    if low > high:
        low,high=high,low

    list_a[low:high+1], list_b[low:high+1] \
        = list_b[low:high+1], list_a[low:high+1]

    return list_a, list_b


def mutate(arch, prob):
    block_candidate = []
    for i in range(2, 18):
        temp_candidate = list(range(i + 1, 18)) + [0]
        block_candidate.append(temp_candidate)
    op1_candidate = [1, 2, 3, 4, 5, 6]
    op2_candidate = [0, 1, 2, 3, 4, 5, 6]
    for idx, item in enumerate(arch):
        if random.random() < prob:
            print('Mutate Started!')
            if idx % 3 == 0:
                print('mutate layer2 index:')
                block_index=idx//3
                candidate=copy.deepcopy(block_candidate)
                candidate[block_index].remove(item)
                if candidate[block_index]:
                    mutate=random.choice(candidate[block_index])
                    arch[idx]=mutate
                else:
                    print('no candidate for layer2 index')

            elif idx % 3 == 1:
                print('mutate op1:')
                candidate_1=copy.deepcopy(op1_candidate)
                candidate_1.remove(item)
                if candidate_1:
                    mutate=random.choice(candidate_1)
                    arch[idx]=mutate
                else:
                    print('no candidate for op1')

            elif idx % 3 == 2:
                print('mutate op2:')
                if idx-2!=0:
                    candidate_2=copy.deepcopy(op2_candidate)
                    candidate_2.remove(item)
                    if candidate_2:
                        mutate=random.choice(candidate_2)
                        arch[idx]=mutate
                    else:
                        print('no candidate for op2')
                else:
                    print('layer2 index is 0, op2 ignored!')

            else:
                print('error!')
    return arch



def next_generation(coder, xargs, sorted_offspring, arch_set):
    formulation, derive=[],[]
    for item in arch_set:
        if item not in formulation:
            formulation.append(item)
    indice = xargs.popl_size//3
    offspring_1 = sorted_offspring[:indice]
    offspring_2 = sorted_offspring[indice:indice*2]
    offspring_3 = sorted_offspring[indice*2:]

    for child1, child2 in zip(offspring_1[::2], offspring_1[1::2]):
        arch1=child1[:-1]
        arch2=child2[:-1]
        count1 ,res1 = cal_diff(arch1 , arch2)
        if count1 > 1 and random.random() < xargs.incross_prob:
            count_exist1, count_all1=0,0
            flag1=0
            print('inside spring_1 cross:')
            print('child1 before cross:', child1[:-1],'  ', child1[-1])
            print('child2 before cross:', child2[:-1],'  ', child2[-1])
            cross1,cross2=cross(arch1, arch2, res1)
            print('child1 after cross:', cross1)
            print('child2 after cross:', cross2)
            while not coder.check_valid(cross1) or not coder.check_valid(cross2) or check_exist(cross1,formulation) or check_exist(cross2,formulation):
                print('cross failed! cross again!')
                if not coder.check_valid(cross1):
                    print('child1 not valid !')
                if not coder.check_valid(cross2):
                    print('child2 not valid !')
                if check_exist(cross1,formulation) or check_exist(cross2,formulation):
                    count_exist1+=1
                    if check_exist(cross1,formulation):
                        print('child1 exist in formulation!')
                    if check_exist(cross2,formulation):
                        print('child2 exist in formulation!')
                cross1,cross2=cross(arch1,arch2, res1)
                print('child1 cross again:', cross1)
                print('child2 cross again:', cross2)
                count_all1+=1
                if count_exist1>20 or count_all1>80:
                    flag1=1
                    print('Too many fails, quit crossover!')
                    break
            if flag1==0:
                formulation.append(cross1)
                formulation.append(cross2)
                derive.append(cross1)
                derive.append(cross2)
                offspring_1.remove(child1)
                offspring_1.remove(child2)


    for child3, child4 in zip(offspring_1, offspring_2):
        arch3 = child3[:-1]
        arch4 = child4[:-1]
        count2, res2 = cal_diff(arch3, arch4)
        if count2 > 1 and random.random() < xargs.cross_prob:
            count_exist2, count_all2 = 0, 0
            flag2 = 0
            print('between spring_1 and spring_2 cross:')
            print('child3 before cross:', child3[:-1], '  ', child3[-1])
            print('child4 before cross:', child4[:-1], '  ', child4[-1])
            cross3, cross4 = cross(arch3, arch4, res2)
            print('child3 after cross:', cross3)
            print('child4 after cross:', cross4)
            while not coder.check_valid(cross3) or not coder.check_valid(cross4) or check_exist(cross3,formulation) or check_exist(cross4, formulation):
                print('cross failed! cross again!')
                if not coder.check_valid(cross3):
                    print('child3 not valid !')
                if not coder.check_valid(cross4):
                    print('child4 not valid !')
                if check_exist(cross3, formulation) or check_exist(cross4, formulation):
                    count_exist2 += 1
                    if check_exist(cross3, formulation):
                        print('child3 exist in formulation!')
                    if check_exist(cross4, formulation):
                        print('child4 exist in formulation!')
                cross3, cross4 = cross(arch3, arch4, res2)
                print('child3 cross again:', cross3)
                print('child4 cross again:', cross4)
                count_all2 += 1
                if count_exist2 > 20 or count_all2 > 80:
                    flag2 = 1
                    print('Too many fails, quit crossover!')
                    break
            if flag2 == 0:
                formulation.append(cross3)
                formulation.append(cross4)
                derive.append(cross3)
                derive.append(cross4)
                offspring_1.remove(child3)
                offspring_2.remove(child4)




    for child5, child6 in zip(offspring_2[::2], offspring_2[1::2]):
        arch5 = child5[:-1]
        arch6 = child6[:-1]
        count3, res3 = cal_diff(arch5, arch6)
        if  count3 > 1 and random.random() < xargs.incross_prob:
            count_exist3, count_all3 = 0, 0
            flag3 = 0
            print('inside spring_2 cross:')
            print('child5 before cross:', child5[:-1], '  ', child5[-1])
            print('child6 before cross:', child6[:-1], '  ', child6[-1])
            cross5, cross6 = cross(arch5, arch6, res3)
            print('child5 after cross:', cross5)
            print('child6 after cross:', cross6)
            while not coder.check_valid(cross5) or not coder.check_valid(cross6) or check_exist(cross5,formulation) or check_exist(cross6, formulation):
                print('cross failed! cross again!')
                if not coder.check_valid(cross5):
                    print('child5 not valid !')
                if not coder.check_valid(cross6):
                    print('child6 not valid !')
                if check_exist(cross5, formulation) or check_exist(cross6, formulation):
                    count_exist3 += 1
                    if check_exist(cross5, formulation):
                        print('child5 exist in formulation!')
                    if check_exist(cross6, formulation):
                        print('child6 exist in formulation!')
                cross5, cross6 = cross(arch5, arch6, res3)
                print('child5 cross again:', cross5)
                print('child6 cross again:', cross6)
                count_all3 += 1
                if count_exist3 > 20 or count_all3 > 80:
                    flag3 = 1
                    print('Too many fails, quit crossover!')
                    break
            if flag3 == 0:
                formulation.append(cross5)
                formulation.append(cross6)
                derive.append(cross5)
                derive.append(cross6)
                offspring_2.remove(child5)
                offspring_2.remove(child6)



    for child7, child8 in zip(offspring_1, offspring_3):
        arch7 = child7[:-1]
        arch8 = child8[:-1]
        count4, res4 = cal_diff(arch7, arch8)
        if  count4 > 1 and random.random() < xargs.cross_prob:
            count_exist4, count_all4 = 0, 0
            flag4 = 0
            print('between spring_1 and spring_3 cross:')
            print('child7 before cross:', child7[:-1], '  ', child7[-1])
            print('child8 before cross:', child8[:-1], '  ', child8[-1])
            cross7, cross8 = cross(arch7, arch8, res4)
            print('child7 after cross:', cross7)
            print('child8 after cross:', cross8)
            while not coder.check_valid(cross7) or not coder.check_valid(cross8) or check_exist(cross7,formulation) or check_exist(cross8, formulation):
                print('cross failed! cross again!')
                if not coder.check_valid(cross7):
                    print('child7 not valid !')
                if not coder.check_valid(cross8):
                    print('child8 not valid !')
                if check_exist(cross7, formulation) or check_exist(cross7, formulation):
                    count_exist4 += 1
                    if check_exist(cross7, formulation):
                        print('child7 exist in formulation!')
                    if check_exist(cross8, formulation):
                        print('child8 exist in formulation!')
                cross7, cross8 = cross(arch7, arch8, res4)
                print('child7 cross again:', cross7)
                print('child8 cross again:', cross8)
                count_all4 += 1
                if count_exist4 > 20 or count_all4 > 80:
                    flag4 = 1
                    print('Too many fails, quit crossover!')
                    break
            if flag4 == 0:
                formulation.append(cross7)
                formulation.append(cross8)
                derive.append(cross7)
                derive.append(cross8)
                offspring_1.remove(child7)
                offspring_3.remove(child8)


    for child9, child10 in zip(offspring_2, offspring_3):
        arch9 = child9[:-1]
        arch10 = child10[:-1]
        count5, res5 = cal_diff(arch9, arch10)
        if  count5 > 1 and random.random() < xargs.cross_prob:
            count_exist5, count_all5 = 0, 0
            flag5 = 0
            print('between spring_2 and spring_3 cross:')
            print('child9 before cross:', child9[:-1], '  ', child9[-1])
            print('child10 before cross:', child10[:-1], '  ', child10[-1])
            cross9, cross10 = cross(arch9, arch10, res5)
            print('child9 after cross:', cross9)
            print('child10 after cross:', cross10)
            while not coder.check_valid(cross9) or not coder.check_valid(cross10) or check_exist(cross9,formulation) or check_exist(cross10, formulation):
                print('cross failed! cross again!')
                if not coder.check_valid(cross9):
                    print('child9 not valid !')
                if not coder.check_valid(cross10):
                    print('child10 not valid !')
                if check_exist(cross9, formulation) or check_exist(cross10, formulation):
                    count_exist5 += 1
                    if check_exist(cross9, formulation):
                        print('child9 exist in formulation!')
                    if check_exist(cross10, formulation):
                        print('child10 exist in formulation!')
                cross9, cross10 = cross(arch9, arch10, res5)
                print('child9 cross again:', cross9)
                print('child10 cross again:', cross10)
                count_all5 += 1
                if count_exist5 > 20 or count_all5 > 80:
                    flag5 = 1
                    print('Too many fails, quit crossover!')
                    break
            if flag5 == 0:
                formulation.append(cross9)
                formulation.append(cross10)
                derive.append(cross9)
                derive.append(cross10)
                offspring_2.remove(child9)
                offspring_3.remove(child10)

    offspring_final=offspring_1+offspring_2+offspring_3
    print('for mutant len :',len(offspring_final))

    for mutant in offspring_final:
        arch=mutant[:-1]
        print('mutate:')
        print('before:', arch,'  ', mutant[-1])
        m_arch=mutate(arch,xargs.m_prob)
        print('after:', m_arch)
        while not coder.check_valid(m_arch) or check_exist(m_arch,formulation):
            print('mutate failed! mutate again!')
            if not coder.check_valid(m_arch):
                print('mutant not valid!')
            if check_exist(m_arch,formulation):
                print('mutant exist in formulation!')
            m_arch=mutate(arch,xargs.m_prob)
            print('mutate again:', m_arch)
        formulation.append(m_arch)
        derive.append(m_arch)

    print('arch_set len:',len(arch_set))
    print('derive len:',len(derive))
    print('formulation len:', len(formulation))

    derive_set=[]
    for de in derive:
        if de not in derive_set:
            derive_set.append(de)
        else:
            print('repeat exists in derive!')
            print(de)
    if derive==derive_set:
        print('no repeat')
    else:
        print('Error happened in derive set and derive!')

    return derive
