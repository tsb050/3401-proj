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


def main():
    S = entropy_S(attribute_dict)
    age, gain = get_age_splitting_value(attribute_dict)
    cred_gain = get_attr_gain('INCOME', attribute_dict)
    print('entropy(S): {}'.format(S))
    print("Split value Age: {} at Gain value {}".format(age, gain))
    print("Gain for CRED_HIS: {}".format(cred_gain))


if __name__ == "__main__":
    main()
