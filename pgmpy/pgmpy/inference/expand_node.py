from scipy.optimize import linprog

#modifies mplp object according to fixing
#fixing is a dictionary of {node: value}

def mod_fixing(mplp,fixing):
    

def LP_expand(mplp): 
    (nodes,edges,_) = mplp.get_factors_by_type()

    var_to_index = {}
    index = 0
    for node in nodes:
        var_to_index[frozenset([node])] = (index,index+1)
        index += 2

    edge_offset = 2*len(nodes)
    max_offset = 0
    for edge in edges:
        edge_copy = set(edge)
        var_to_index[edge] = (edge_offset,edge_offset+3)  
        max_offset = max(max_offset,edge_offset+3)
        edge_offset += 4

    obj = [0]*(max_offset+1)
    factor_dict = mplp.factor_dict

    for node in nodes:
        node = frozenset([node])
        (start,end) = var_to_index[node]
        values = factor_dict[node].values 
        obj[start] = values[0] 
        obj[end] = values[1]

    for edge in edges:
        (start,end) = var_to_index[edge]
        values = []
        for val in factor_dict[edge].values:
            values += list(val)    
        for i in range(4):
            obj[start+i] = values[i]

    #note minimization 
    obj = [-1*elem for elem in obj]
    total_var = len(obj)
    lhs_eq = []
    rhs_eq = []

    index = -1
    for node in nodes:
        eq = [0]*total_var
        node = frozenset([node])
        (start,end) = var_to_index[node]
        for i in range(start,end+1):
            eq[i] = 1 
        lhs_eq.append(eq)
        rhs_eq.append(1)

    for edge in edges:
        (start,end) = var_to_index[edge]
        constraint = [[1,1,0,0],
                     [0,0,1,1],
                     [1,0,1,0],
                     [0,1,0,1]]

        for i in range(4):
            eq = [0]*total_var
            eq[start:end+1] = constraint[i] 
            edge_order = factor_dict[edge].variables
            if i < 2:
                marg = edge_order[0]
                marg_index = var_to_index[frozenset([marg])]
                eq[marg_index[i%2]] = -1
            else: 
                marg = edge_order[1]
                marg_index = var_to_index[frozenset([marg])]
                eq[marg_index[i%2]] = -1
            lhs_eq.append(eq)
            rhs_eq.append(0)

    bnd = [(0,float("inf")) for i in range(total_var)]
    opt = linprog(c=obj,
                   A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd,
                    method="revised simplex")

    #decode
    print('opt.fun: ', opt.fun)
    solution = opt.x
    decode = {}
    for node in nodes: 
        print('node: ', node)
        (start,end) = var_to_index[frozenset([node])]
        sol = solution[start:end+1]
        print('values: ', sol)
        decode[node] = np.argmax(np.array(sol))


    return feasible, objective, solution

#(obj,decode) = degree_two(mplp)