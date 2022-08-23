import csv
import itertools
import copy

with open('transactions5.csv', 'r') as f:
    reader = csv.reader(f)
    transactions = list(reader)

print("Print all the input transaction: ")
print(transactions)
print('\n')


min_support = input("Enter the min support: ")
min_confidence = input("Enter the min confidence: ")
min_sup = float(min_support)
min_conf = float(min_confidence)


def apriori(transactions, min_support, min_confidence):
    if min_support <= 0:
        raise ValueError('minimum support must be > 0')

    transactions_len = len(transactions)
    print("The total number of transactions is : %d" % transactions_len)
    sup = min_support * transactions_len
    print("The min_sup is: %d" % sup)
    print('\n')

    freq_itemsets = []
    alldict = {}

    """

    Calculate the frequence itemset

    """

    # Calculate the support one item
    dict1 = {}
    for tran in transactions:
        for item in tran:
            if item in dict1:
                dict1[item] += 1
            else:
                dict1[item] = 1

    # Remove the itemsets that are less than the min_sup
    del_list1 = []
    dict1_copy = dict1.copy()
    for key in dict1_copy:
        if dict1_copy[key] < sup:
            del_list1.append(key)
            del(dict1[key])
    print("Print out the items that are not satisfied: ")
    print(del_list1)

    satisfy_list1 = []
    for key in dict1:
        satisfy_list1.append(key)
    print("Print out the satisfy_list1: ")
    print(satisfy_list1)
    print("Print the new dict1: ")
    print(dict1)
    print('\n')
    alldict.update(dict1)

    # Do combination to expand to more itemsets
    result_list = list(itertools.combinations(satisfy_list1, 2))
    print("Print the combinations result_list: ")
    print(result_list)
    print('\n')

    # Calculate 2 ~ n items
    dictionary = {}
    for tran in transactions:
        for tp in result_list:
            allin = True
            for k in range(0, len(tp)):
                if tp[k] not in tran:
                    allin = False
                    break
            if allin is True:
                if tuple(sorted(tp)) in dictionary:
                    dictionary[tuple(sorted(tp))] += 1
                else:
                    dictionary[tuple(sorted(tp))] = 1
    print("Print out the dictionary: ")
    print(dictionary)
    print('\n')

    del_list = []
    dict_copy = dictionary.copy()
    for key in dict_copy:
        if dict_copy[key] < sup:
            del_list.append(key)
            del(dictionary[key])

    alldict.update(dictionary)
    print("Print out the items that are not satisfied: ")
    print(del_list)
    print("Print the new dictionary: ")
    print(dictionary)
    print('\n')

    satisfy_list = []
    for key in dictionary:
        satisfy_list.append(key)
    print("Print out the satisfy_list: ")
    print(satisfy_list)

    # Frequence itemsets should be more than one item
    freq_itemsets = copy.copy(satisfy_list)


    while len(dictionary) > 2:
        comb_list = list(itertools.combinations(satisfy_list, 2))
        print("Comb_list: ")
        print(comb_list)

        set_list = []
        for tp in comb_list:
            s = set(tp[0]).union(tp[1])
            if s not in set_list:
                set_list.append(s)

        print("Print the set_list")
        print(set_list)

        result_list = []
        for k in range(0, len(set_list)):
            result_list.append(tuple(set_list[k]))

        print("Print the combinations result_list: ")
        print(result_list)
        print('\n')


        dictionary = {}
        for tran in transactions:
            for tp in result_list:
                allin = True
                for k in range(0, len(tp)):
                    if tp[k] not in tran:
                        allin = False
                        break
                if allin is True:
                    if tuple(sorted(tp)) in dictionary:
                        dictionary[tuple(sorted(tp))] += 1
                    else:
                        dictionary[tuple(sorted(tp))] = 1
        print("Print out the dictionary: ")
        print(dictionary)
        print('\n')


        del_list = []
        dict_copy = dictionary.copy()
        for key in dict_copy:
            if dict_copy[key] < sup:
                del_list.append(key)
                del(dictionary[key])

        alldict.update(dictionary)
        print("Print out the items that are not satisfied: ")
        print(del_list)
        print("Print the new dictionary: ")
        print(dictionary)

        satisfy_list = []
        for key in dictionary:
            satisfy_list.append(key)
        print("Print out the satisfy_list: ")
        print(satisfy_list)

        freq_itemsets = freq_itemsets + satisfy_list

    # End of the while loop

    # Print out the frequence itemsets and all dictionary
    print('\n')
    print("print the freq_itemsets")
    print(freq_itemsets)
    print('\n')
    print("Print the alldict: ")
    print(alldict)
    print('\n')


    """

    Generate association rules

    and

    Calculate the confidence

    """

    print("Print the association_rules: ")
    for tp in freq_itemsets:
        subsets_ls = list_subsets(list(tp))
        tp2st = set(tp)
        for s in subsets_ls:
            if (len(s) != len(tp2st)):
                # confifence: fractor = P(X,Z), demo = P(X)
                # support: fractor = P(X,Z), demo = totoal
                if len(s) == 1:
                    deno_conf = alldict[list(s)[0]]
                    deno_sup = alldict[list(s)[0]]
                else:
                    deno_conf = alldict[tuple(sorted(tuple(s)))]
                    deno_sup = alldict[tuple(sorted(tuple(s)))]
                frac_conf = alldict[tp]
                confidence = frac_conf / deno_conf
                support = frac_conf / transactions_len
                if confidence >= min_confidence:
                    print(s, '-->', tp2st.difference(s), ' support: ', support, ' confidence: ', confidence)

# End of function apriori

def list_subsets(in_list):
    '''
    Returns a list of subsets in type set
    '''
    result = []
    list_subsets_r(in_list, [], 0, result)
    return result


def list_subsets_r(in_list, state, start, result):
    for i in range(start, len(in_list)):
        state.append(in_list[i])
        result.append(set(state))
        list_subsets_r(in_list, state, i + 1, result)
        state.pop()
apriori(transactions, min_sup, min_conf)

