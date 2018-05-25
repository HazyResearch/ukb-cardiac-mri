from inspect import getsourcelines
import numpy as np

def find_dependencies(L_names, primitive_names):
    LFs = []
    for lf in L_names:
        LFs.append(getsourcelines(lf)[0])  

    L_deps = []
    for lf_idx, lf in enumerate(LFs):
        L_dep = []
        for line in lf:
            if len(line.strip()) > 0:
                if line.lstrip()[:3] == "def":
                    parameters = line[line.find("(")+1:line.rfind(")")].split(",")
                    for param in parameters:
                        L_dep.append(primitive_names.index(param.strip()))
        L_deps.append(L_dep)    
    return L_deps

def discretize_primitives(L_names):
    LFs = []
    for lf in L_names:
        LFs.append(getsourcelines(lf)[0])  

    # extract all conditions from labeling functions
    lfs_primitives = {}
    primitives = {}
    primitives_ordered = []
    for lf_idx, lf in enumerate(LFs):
        for line in lf:
            if len(line.strip()) > 0:
                if (line.lstrip()[:2] == "if" or line.lstrip()[:4] == "elif") and line.rstrip()[-1] == ":":
                    p_cond = line[line.find("if")+2:-1].lstrip()    #TODO(pabajaj): handle multiple and or conditions
                    p_name, p_cond, p_thresh = p_cond.split()
                    if p_name not in primitives:
                        primitives_ordered.append(p_name)
                        primitives[p_name] = []
                    if (p_name, p_cond, p_thresh) not in primitives[p_name]:
                        primitives[p_name].append((p_name, p_cond, p_thresh))

    code = []
    for p_idx, p_name in enumerate(primitives_ordered):
        p_idx_str = str(p_idx)
        p_max = len(primitives[p_name])
        p_data = primitives[p_name][0]         

        code.append("P.discrete_primitive_mtx[i,"+p_idx_str+"] = "+str(p_max)+" if P.primitive_mtx[i,"+p_idx_str+"] "+p_data[1]+" "+p_data[2].strip(':')+" else 0")
        for p_data in primitives[p_name][1:]:
            p_max -= 1
            code.append("P.discrete_primitive_mtx[i,"+p_idx_str+"] = "+str(p_max)+" if P.primitive_mtx[i,"+p_idx_str+"] "+p_data[1]+" "+p_data[2].strip(':')+" else P.discrete_primitive_mtx[i,"+p_idx_str+"]")
    return code