import json
import numpy as np
import math


def retrieve_list():
    my_list = [line.rstrip('\n') for line in open('../data/test.txt')]
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
    # Get class label
    label_class = np.asarray(test_dict['RISK'])
    unique_elements, counts_elements = np.unique(
        label_class, return_counts=True)
    main_entropy = 0
    for num in range(len(counts_elements)):
        ent_label = counts_elements[num]/len(label_class)
        main_entropy -= (ent_label) * math.log2(ent_label)
    return main_entropy


def get_age_splitting_value(test_dict):
    label_class = np.asarray(test_dict['RISK'])
    age_label_class = np.asarray(test_dict['AGE'])
    unique_elements, counts_elements = np.unique(
        age_label_class, return_counts=True)
    less_attr_index = []
    greater_attr_index = []
    for i in unique_elements:
        less_list = []
        greater_list = []
        for idx, el in enumerate(age_label_class):
            if int(el) <= int(i):
                less_list.append(idx)
            if int(el) > int(i):
                greater_list.append(idx)
        less_attr_index.append(less_list)
        greater_attr_index.append(greater_list)

    pol = []
    for ii in range(len(less_attr_index)):
        risk_assess = []
        for a in less_attr_index[ii]:
            risk_assess.append(label_class[a])
        risk_assess
        u1, c1 = np.unique(risk_assess, return_counts=True)
        less_entropy = 0
        for num in range(len(c1)):
            ent_label = c1[num]/len(less_attr_index[ii])
            less_entropy -= (ent_label) * math.log2(ent_label)

        grisk_assess = []
        for a in greater_attr_index[ii]:
            grisk_assess.append(label_class[a])
        u2, c2 = np.unique(grisk_assess, return_counts=True)
        great_entropy = 0
        for num in range(len(c2)):
            ent_label = c2[num]/len(greater_attr_index[ii])
            great_entropy -= (ent_label) * math.log2(ent_label)

        #print("Less entropy", less_entropy)
        #print("Greater entropy", great_entropy)
        entropy_given = (len(less_attr_index[ii])/len(label_class)) * less_entropy + (
            len(greater_attr_index[ii])/len(label_class)) * great_entropy
        #print("Given AGE {} entropy : {}".format(int(ii)+1, entropy_given))
        main_entropy = entropy_S(test_dict)
        Gain = main_entropy - entropy_given
        #print("Gain ", Gain)
        # print("")
        pol.append(Gain)

    best_splitting_value = pol.index(max(pol)) + 1
    return best_splitting_value, max(pol)

# This does not seem to work for RACE. will debug.


def get_attr_gain(attr_name, attribute_dict):
    label_class = np.asarray(attribute_dict['RISK']).astype(int)
    attr = np.asarray(attribute_dict['{}'.format(attr_name)]).astype(int)
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
    main_entropy = entropy_S(attribute_dict)
    Gain = main_entropy - entropy_given
    return Gain


def get_index(unique_list, attr):
    index = []
    for d in unique_list:
        disc = []
        for idx, elem in enumerate(attr):
            if d == elem:
                disc.append(idx)
        index.append(disc)
    return index


def get_attribute(index, test_attribute, attribute):
    the_list = list()
    for ids in index:
        new_dict = {}
        for key in attribute:
            loss = []
            if key == test_attribute:
                continue
            for val in ids:
                loss.append(np.asarray(attribute[key]).astype(int)[val])
            new_dict[key] = loss
        the_list.append(new_dict)
    return the_list


def get_max_gain(attribute_dict):
    growth = {}
    for key in attribute_dict:
        if key == 'RISK':
            continue
        gain = get_attr_gain(key, attribute_dict)
        growth[key] = gain
    return max(growth, key=growth.get)


def grow():
    # compare the gains and determine the test attribute
    test_attribute = get_max_gain(attribute_dict)
    node_1 = np.asarray(attribute_dict[test_attribute]).astype(int)
    u2, c2 = np.unique(node_1, return_counts=True)
    root_node_index = get_index(u2, node_1)
    root_sub_attrs = get_attribute(
        root_node_index, test_attribute, attribute_dict)
    print(test_attribute)
    print("-------------------------------------------")
    sub_root = {}
    my_dict = []
    my_dict.append(test_attribute)
    for i in range(len(u2)):
        sub_root[u2[i]] = root_sub_attrs[i]
    sub_val = {}
    for sub_key in sub_root:
        # Check if Risk has uniform values
        sub_unique = np.unique(sub_root[sub_key]['RISK'])
        if len(sub_unique) == 1:
            label = sub_unique[0]
        else:
            sub_heading = get_max_gain(sub_root[sub_key])
            sub_node = np.asarray(sub_root[sub_key][sub_heading]).astype(int)
            u3 = np.unique(sub_node)
            sub_node_index = get_index(u3, sub_node)
            sub_attrs = get_attribute(
                sub_node_index, sub_heading, sub_root[sub_key])
            sub_val[sub_key] = sub_heading
            sub_root_2 = {}
            for i in range(len(u3)):
                sub_root_2[u3[i]] = sub_attrs[i]
            sub_val2 = {}
            for s_key in sub_root_2:
                sub_unique2 = np.unique(sub_root_2[s_key]['RISK'])
                if len(sub_unique2) == 1:
                    label = sub_unique2[0]
                else:
                    sub_heading2 = get_max_gain(sub_root_2[s_key])
                    sub_node_2 = np.asarray(
                        sub_root_2[s_key][sub_heading2]).astype(int)
                    u4 = np.unique(sub_node_2)
                    sub_node_index2 = get_index(u4, sub_node_2)
                    sub_attrs2 = get_attribute(
                        sub_node_index2, sub_heading2, sub_root_2[s_key])
                    sub_val2[s_key] = sub_heading2
                    sub_val[sub_key] = [sub_heading, sub_val2]
                    print("-------------------------------------------")
                    print(sub_heading2)
                    print(sub_root_2[s_key]['RISK'])
                    #sub_val2[s_key] = sub_heading2
                    # print(sub_val2)
                    #sub_val[sub_key] = [sub_heading, sub_val2]
                    #print(sub_attrs2)
    my_dict.append(sub_val)
    print(my_dict)
    """
                    print(sub_attrs2)
                    sub_root_3 = {}
                    for ii in range(len(u4)):
                        sub_root_3[u4[ii]] = sub_attrs2[ii]
                    for s_key in sub_root_3:
                        sub_unique3 = np.unique(sub_root_3[s_key]['RISK'])
                        if len(sub_unique3) == 1:
                            label = sub_unique3[0]
                        else:
                            sub_heading3 = get_max_gain(sub_root_3[s_key])
                            sub_node_3 = np.asarray(
                                sub_root_3[s_key][sub_heading3]).astype(int)
                            u5 = np.unique(sub_node_3)
                            sub_node_index3 = get_index(u4, sub_node_3)
                            sub_attrs3 = get_attribute(
                                sub_node_index3, sub_heading3, sub_root_3[s_key])
                            print("-------------------------------------------")
                            print(sub_heading3)
                            print(sub_attrs3)
                            sub_root_4 = {}
                            for ii in range(len(u5)):
                                sub_root_4[u5[ii]] = sub_attrs3[ii]
                            for s_key in sub_root_4:
                                sub_unique4 = np.unique(
                                    sub_root_4[s_key]['RISK'])
                                if len(sub_unique4) == 1:
                                    label = sub_unique4[0]
                                else:
                                    sub_heading4 = get_max_gain(
                                        sub_root_4[s_key])
                                    sub_node_4 = np.asarray(
                                        sub_root_4[s_key][sub_heading4]).astype(int)
                                    u6 = np.unique(sub_node_4)
                                    sub_node_index4 = get_index(u5, sub_node_4)
                                    sub_attrs4 = get_attribute(
                                        sub_node_index4, sub_heading4, sub_root_4[s_key])
                                    print(
                                        "-------------------------------------------")
                                    print(sub_heading4)
                                    print(sub_attrs4)
                                    sub_root_5 = {}
                                    for ii in range(len(u5)):
                                        sub_root_5[u5[ii]] = sub_attrs4[ii]
                                    for s_key in sub_root_5:
                                        sub_unique5 = np.unique(
                                            sub_root_5[s_key]['RISK'])
                                        if len(sub_unique5) == 1:
                                            label = sub_unique5[0]
                                        else:
                                            try:
                                                sub_heading5 = get_max_gain(
                                                    sub_root_5[s_key])
                                                sub_node_5 = np.asarray(
                                                    sub_root_5[s_key][sub_heading5]).astype(int)
                                                u6 = np.unique(sub_node_5)
                                                sub_node_index5 = get_index(u5, sub_node_5)
                                                sub_attrs5 = get_attribute(
                                                    sub_node_index5, sub_heading5, sub_root_5[s_key])
                                                print("-------------------------------------------")
                                                print(sub_heading5)
                                                print(sub_attrs5)
                                            except ValueError:
                                                print("Empty sequesnce")
    """


def main():
    grow()


if __name__ == "__main__":
    main()
