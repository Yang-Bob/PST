import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ortools.graph import pywrapgraph

def Match(mu_f_s, mu_f_q):
    b = mu_f_q.size(0)
    dis = torch.bmm(mu_f_s, mu_f_q.permute(0, 2, 1))
    value, indices = torch.max(dis, dim=2)

    for i in range(b):
        x = indices[i]
        if i == 0:
            mu_f_q_t = mu_f_q[i, x].unsqueeze(dim=0)
        else:
            mu_f_q_temp = mu_f_q[i, x].unsqueeze(dim=0)
            mu_f_q_t = torch.cat([mu_f_q_t, mu_f_q_temp], dim=0)

    return mu_f_s, mu_f_q_t

def MCMFMatch(mu_f_s, mu_f_q, cap_end=1):
    b = mu_f_q.size(0)
    dis = torch.bmm(mu_f_s, mu_f_q.permute(0, 2, 1))
    Cost = 1 - dis
    Cost = Cost * 1000
    Cost = Cost.int()
    for i in range(b):
        cost = Cost[i].cpu().numpy().tolist()
        x = MCMF(cost, cap_end)
        if i == 0:
            mu_f_q_t = mu_f_q[i, x].unsqueeze(dim=0)
        else:
            mu_f_q_temp = mu_f_q[i, x].unsqueeze(dim=0)
            mu_f_q_t = torch.cat([mu_f_q_t, mu_f_q_temp], dim=0)

    return mu_f_s, mu_f_q_t

def MCMF(Cost, cap_end=1):
    #Cost = Cost.tolist()
    half_node_num = len(Cost)
    all_node_num = half_node_num*2
    Start = 0
    End = all_node_num +1

    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    # Add each arc.
    for i in range(half_node_num):
        node_id = i+1
        min_cost_flow.AddArcWithCapacityAndUnitCost(Start, node_id,
                                                    1, 0)
        min_cost_flow.AddArcWithCapacityAndUnitCost(node_id+half_node_num, End,
                                                    cap_end, 0)
    for i in range(0, half_node_num):
        for j in range(0, half_node_num):
            node_id_s = i+1
            node_id_q = j+1+half_node_num
            min_cost_flow.AddArcWithCapacityAndUnitCost(node_id_s, node_id_q, 1, Cost[i][j])
        # Add node supplies.

    min_cost_flow.SetNodeSupply(0, half_node_num)
    min_cost_flow.SetNodeSupply(all_node_num+1, -1*half_node_num)
    # Find the minimum cost flow between node 0 and node 4.
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        order =np.array(range(half_node_num))
        for i in range(half_node_num*2,min_cost_flow.NumArcs()):
            if min_cost_flow.Flow(i)==1:
                order[min_cost_flow.Tail(i)-1] = min_cost_flow.Head(i)-half_node_num-1
        return order
    else:
        print('There was an issue with the min cost flow input.')
        return 0
