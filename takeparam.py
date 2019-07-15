import json




def takeparam(file, param):
    exist_param = {}
    nonexist_param = []
    with open(file, 'r')as infile:
        val_file = json.load(infile)
        for i in param:
            if i in val_file.keys():
                pair_param_val = dict.fromkeys([i], val_file[i])
                exist_param.update(pair_param_val)
            else:
                nonexist_param.append(i)
    if len(nonexist_param)>0:
        raise KeyError(nonexist_param)
    else:
        return exist_param

