import json
import numpy as np
import math


def retrieve_list():
    my_list = [line.rstrip('\n') for line in open('train.txt')]
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

def get_attr_gain(attr_name, attribute_dict):
    """
    Take attribute name and training/ test datasets .

    return: Gain value for specified attribute
    """
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
    """
    So here, this function takes the unique values from
    you dataset e.g [1,2,3] and returns a list of indexes 
    of values 1,2,3 .

    return: list of lists containing indexes of unique 
    attribute values
    """
    index = []
    for d in unique_list:
        disc = []
        for idx, elem in enumerate(attr):
            if d == elem:
                disc.append(idx)
        index.append(disc)
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
                loss.append(np.asarray(attribute[key]).astype(int)[val])
            new_dict[key] = loss
        the_list.append(new_dict)
    return the_list


def get_max_gain(attribute_dict):
    """
    Call function that calculates the gain.
    Get a dict of gains, retrieve the maximum
    gain values and its attribute. 

    return: Test attribute and other parent attributes in the tree.
    """
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
    #print(test_attribute)
    #print("-------------------------------------------")
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
            #print(sub_heading)
            sub_node = np.asarray(sub_root[sub_key][sub_heading]).astype(int)
            u3 = np.unique(sub_node)
            sub_node_index = get_index(u3, sub_node)
            sub_attrs = get_attribute(
                sub_node_index, sub_heading, sub_root[sub_key])
            sub_val[str(sub_key)] = sub_heading
            sub_root_2 = {}
            #print(sub_attrs)
            for i in range(len(u3)):
                sub_root_2[u3[i]] = sub_attrs[i]
            sub_val2 = {}
            for s_key in sub_root_2:
                sub_unique2 = np.unique(sub_root_2[s_key]['RISK'])
                if len(sub_unique2) == 1 or len(sub_root_2[s_key]['RISK']) == 1 or len(sub_root_2[s_key]['RISK']) == 0 :
                    label = sub_unique2.tolist()
                    print('done here')
                else:
                    sub_heading2 = get_max_gain(sub_root_2[s_key])
                    #print(sub_root_2[s_key]['RISK'])
                    sub_node_2 = np.asarray(
                        sub_root_2[s_key][sub_heading2]).astype(int)
                    u4 = np.unique(sub_node_2)
                    sub_node_index2 = get_index(u4, sub_node_2)
                    sub_attrs2 = get_attribute(
                        sub_node_index2, sub_heading2, sub_root_2[s_key])
                    sub_val2[str(s_key)] = sub_heading2
                    sub_val[str(sub_key)] = [sub_heading, sub_val2]
                    #print("-------------------------------------------")
                    #print(sub_attrs2)          
                    sub_root_3 = {}
                    for ii in range(len(u4)):
                        sub_root_3[u4[ii]] = sub_attrs2[ii]
                    sub_val3 = {}
                    for s_key2 in sub_root_3:
                        #print(s_key2)
                        sub_unique3 = np.unique(sub_root_3[s_key2]['RISK'])
                        sub_heading3 = get_max_gain(sub_root_3[s_key2])
                        #sub_val3[s_key2] = sub_heading3
                        #sub_val2[s_key] = [sub_heading2, sub_val3]

                        if len(sub_unique3) == 1 or len(sub_root_3[s_key2]['RISK']) == 1 or len(sub_root_3[s_key2]['RISK']) == 0:
                            label = sub_unique3.tolist()
                            sub_val3[str(s_key2)] = sub_heading3
                            sub_val2[str(s_key)] = label
                        else:
                            sub_node_3 = np.asarray(
                                sub_root_3[s_key2][sub_heading3]).astype(int)
                            u5 = np.unique(sub_node_3)
                            sub_node_index3 = get_index(u5, sub_node_3)
                            sub_attrs3 = get_attribute(
                                sub_node_index3, sub_heading3, sub_root_3[s_key2])
                            #print("-------------------------------------------")
                            #print(sub_heading3)
                            #print(sub_attrs3)
                            sub_val3[str(s_key2)] = sub_heading3
                            sub_val2[str(s_key)] = [sub_heading2, sub_val3]
                            sub_root_4 = {}
                            for ii in range(len(u5)):
                                sub_root_4[u5[ii]] = sub_attrs3[ii]
                            sub_val4 = {}
                            for s_key3 in sub_root_4:
                                sub_unique4 = np.unique(sub_root_4[s_key3]['RISK'])
                                sub_heading4 = get_max_gain(sub_root_4[s_key3])
                                if len(sub_unique4) == 1 or len(sub_root_4[s_key3]['RISK']) == 1 or len(sub_root_4[s_key3]['RISK']) == 0:
                                    label = sub_unique4.tolist()
                                    sub_val4[str(s_key3)] = sub_heading4
                                    sub_val3[str(s_key2)] = label
                                else:
                                    sub_node_4 = np.asarray(
                                        sub_root_4[s_key3][sub_heading4]).astype(int)
                                    u6 = np.unique(sub_node_4)
                                    sub_node_index4 = get_index(u6, sub_node_4)
                                    sub_attrs4 = get_attribute(
                                        sub_node_index4, sub_heading4, sub_root_4[s_key3])
                                    sub_val4[str(s_key3)] = sub_heading4
                                    sub_val3[str(s_key2)] = [sub_heading3, sub_val4]
                                    sub_root_5 = {}
                                    for ii in range(len(u6)):
                                        sub_root_5[u6[ii]] = sub_attrs4[ii]
                                    sub_val5 = {}
                                    for s_key4 in sub_root_5:
                                        sub_unique5 = np.unique(sub_root_5[s_key4]['RISK'])
                                        if len(sub_unique5) == 1 or len(sub_root_5[s_key4]['RISK']) == 1 or len(sub_root_5[s_key4]['RISK']) == 0:
                                            label = sub_unique5.tolist()
                                            sub_val4[str(s_key3)] = label
                                        #   TO BE CONTINUED!!!
                                        elif len(sub_root_5[s_key4]) == 1:
                                            sub_val4[str(s_key3)] = [max(sub_root_5[s_key4]['RISK'])]

    my_dict.append(sub_val)
    print(my_dict)
    with open("nao.txt", 'w') as f:
        json.dump(str(my_dict), f)

def main():
    grow()


if __name__ == "__main__":
    main()
