import json
import numpy as np
import math


def retrieve_list():
    my_list = [line.rstrip('\n') for line in open('../data/train.txt')]
    return_list = []
    for rows in my_list:
        a = rows.split(" ")
        return_list.append(a)
    return return_list


new_list = retrieve_list()
attribute_dict = {
    'RISK': new_list[0],
    'AGE': new_list[1],
    'CRED_HIS': new_list[2],
    'INCOME': new_list[3],
    'RACE': new_list[4],
    'HEALTH': new_list[5]
}


def entropy_S(test_dict):
    """
    Calculate entropy(S).

    return: entropy(S)
    """
    label_class = np.asarray(test_dict['RISK'])
    unique_elements, counts_elements = np.unique(
        label_class, return_counts=True)
    main_entropy = 0
    for num in range(len(counts_elements)):
        ent_label = counts_elements[num]/len(label_class)
        main_entropy -= (ent_label) * math.log2(ent_label)
    return main_entropy


def get_attr_gain(attr_name, file_attribute):
    """
    Take attribute name and training/ test datasets .

    return: Gain value for specified attribute
    """
    label_class = np.asarray(file_attribute['RISK'])
    attr = np.asarray(file_attribute['{}'.format(attr_name)])
    u2, c2 = np.unique(attr, return_counts=True)
    index = []
    for d in u2:
        disc = []
        for idx, elem in enumerate(attr):
            if d == elem:
                disc.append(idx)
        index.append(disc)
    entropy_given = 0
    for idd in index:
        assess = []
        for ee in idd:
            assess.append(label_class[ee])
        u1, c1 = np.unique(assess, return_counts=True)
        entropy = 0
        for num in c1:
            ent_label = num/len(idd)
            entropy -= (ent_label) * math.log2(ent_label)
        entropy_given += (len(idd)/len(label_class)) * entropy
    main_entropy = entropy_S(file_attribute)
    Gain = main_entropy - entropy_given
    return Gain


def get_index(unique_list, attr):
    """
    So here, this function takes the unique values from
    you dataset e.g [1,2,3] and returns a list of indexes
    of values 1,2,3 .

    return: list of lists containing indexes of unique
    attribute values
    """
    index = []
    white = []
    for d in unique_list:
        disc = []
        vl = []
        for idx, elem in enumerate(attr):
            if d == elem:
                disc.append(idx)
                vl.append(elem)
        index.append(disc)
        white.append(vl)
    return index


def get_attribute(index, test_attribute, attribute):
    """
    Once we determine a test attribute or parent attribute.
    Group the dataset based on that attribute, remove that
    attribute from the remaining dataset.

    return: list of attributes split between uniques values
    of a test attribute or parent attribute.
    """
    the_list = list()
    for ids in index:
        new_dict = {}
        for key in attribute:
            loss = []
            if key == test_attribute:
                continue
            for val in ids:
                loss.append(np.asarray(attribute[key])[val])
            new_dict[key] = loss
        the_list.append(new_dict)
    return the_list


def get_max_gain(attribute):
    """
    Call function that calculates the gain.
    Get a dict of gains, retrieve the maximum
    gain values and its attribute.

    return: Test attribute and other parent attributes in the tree.
    """
    growth = {}
    for key in attribute:
        if key == 'RISK':
            continue
        gain = get_attr_gain(key, attribute)
        growth[key] = gain
    return max(growth, key=growth.get)


def grow():
    # compare the gains and determine the test attribute
    test_attribute = get_max_gain(attribute_dict)
    node_1 = np.asarray(attribute_dict[test_attribute])
    u2, c2 = np.unique(node_1, return_counts=True)
    root_node_index = get_index(u2, node_1)
    root_sub_attrs = get_attribute(
        root_node_index, test_attribute, attribute_dict)
    my_dict = []
    my_dict.append(test_attribute)
    sub_val, sub_root = get_gain(u2, root_sub_attrs)
    for key in sub_val:
        if type(sub_val[key]) is int:
            continue
        if len(np.unique(sub_root[key]['RISK'])) == 1:
            sub_val[str(key)] = int(np.unique(sub_root[key]['RISK']).tolist()[0])
        header = sub_val[key]
        if type(header) is list:
            print(header)
            continue
        header = sub_val[key]
        sub_node = np.asarray(sub_root[key][header])
        u3 = np.unique(sub_node)
        sub_node_index = get_index(u3, sub_node)
        sub_attrs = get_attribute(sub_node_index, header, sub_root[key])
        sub_val2, sub_root_2 = get_gain(u3, sub_attrs)
        sub_val[str(key)] = [header, sub_val2]
        sub_unique2 = np.unique(sub_root_2[key]['RISK'])
        for key2 in sub_val2:
            if type(sub_val2[key2]) is int:
                continue
            if len(np.unique(sub_root_2[key2]['RISK'])) == 1:
                sub_val2[str(key2)] = int(np.unique(sub_root_2[key2]['RISK']).tolist()[0])
            header2 = sub_val2[key2]
            if type(header2) is int:
                continue
            sub_node2 = np.asarray(sub_root_2[key2][header2])
            u4 = np.unique(sub_node2)
            sub_node_index_2 = get_index(u4, sub_node2)
            sub_attrs2 = get_attribute(
                sub_node_index_2, header2, sub_root_2[key2])
            sub_val3, sub_root_3 = get_gain(u4, sub_attrs2)
            for nn in sub_attrs2:
                velve = np.unique(nn['RISK'])
                if len(velve) == 1:
                    sub_val3[str(key2)] = int(velve.tolist()[0])
            sub_val2[str(key2)] = [header2, sub_val3]
            for key3 in sub_val3:
                if type(sub_val3[key3]) is int:
                    continue
                if len(np.unique(sub_root_3[key3]['RISK'])) == 1:
                    sub_val3[str(key3)] = int(np.unique(sub_root_3[key3]['RISK']).tolist()[0])
                header3 = sub_val3[key3]
                if type(header3) is int:
                    continue
                # print(header3)
                sub_node3 = np.asarray(sub_root_3[key3][header3])
                u5 = np.unique(sub_node3)
                sub_node_index_3 = get_index(u5, sub_node3)
                sub_attrs3 = get_attribute(
                    sub_node_index_3, header3, sub_root_3[key3])
                sub_val4, sub_root_4 = get_gain(u5, sub_attrs3)
                for nn2 in sub_attrs3:
                    velve2 = np.unique(nn2['RISK'])
                    if len(velve2) == 1:
                        sub_val4[str(key3)] = int(velve2.tolist()[0])
                sub_val3[str(key3)] = [header3, sub_val4]
                for key4 in sub_val4:
                    if len(np.unique(sub_root_4[key4]['RISK'])) == 1:
                        sub_val4[str(key4)] = int(np.unique(sub_root_4[key4]['RISK']).tolist()[0])
                    header4 = sub_val4[key4]
                    if type(header4) is int:
                        continue
                    sub_node4 = np.asarray(sub_root_4[key4][header4])
                    u6 = np.unique(sub_node4)
                    sub_node_index_4 = get_index(u6, sub_node4)
                    sub_attrs4 = get_attribute(
                        sub_node_index_4, header4, sub_root_4[key4])
                    ht = []
                    for ff in sub_attrs4:
                        for key in ff:
                            a = np.unique(ff[key])
                            ht.append([a[0]])
                    df = {}
                    for i in range(len(u6)):
                        df[u6[i]] = int(ht[i][0])
                    sub_val4[str(key4)] = [header4, df]
    my_dict.append(sub_val)
    with open("../data/train_decision_tree.txt", 'w') as f:
        json.dump(my_dict, f)


def get_gain(unique, attr):
    sub_root = {}
    for i in range(len(unique)):
        sub_root[unique[i]] = attr[i]
    sub_val = {}
    for sub_key in sub_root:
        growth = {}
        ent_s = entropy_S(sub_root[sub_key])
        exp_class = np.asarray(sub_root[sub_key]['RISK'])
        for key in sub_root[sub_key]:
            if key == 'RISK':
                continue
            attr = np.asarray(sub_root[sub_key][key])
            u3, c3 = np.unique(attr, return_counts=True)
            id_list = get_index(u3, attr)
            fall = []
            entropy_given = 0
            for nana in id_list:
                ray = []
                for jup in nana:
                    ray.append(exp_class[jup])
                fall.append(ray)
            entry = 0
            for ds in range(len(u3)):
                u4, c4 = np.unique(fall[ds], return_counts=True)
                for count in c4:
                    entry -= (count/len(fall[ds])) * \
                        math.log2(count/len(fall[ds]))
                entropy_given += (len(fall[ds])/len(exp_class)) * entry
            gain = ent_s - entropy_given
            growth[key] = gain
        sub_val[str(sub_key)] = max(growth, key=growth.get)
    return sub_val, sub_root


def main():
    grow()


if __name__ == "__main__":
    main()
