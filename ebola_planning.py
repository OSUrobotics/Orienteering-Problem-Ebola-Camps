#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 6 15:25:33 2018

@author: sritee
"""

#Orienteering problem with Miller-Tucker-Zemlin formulation

import cvxpy as c
import numpy as np
import sys
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(1)

num_nodes=10 #number of nodes in the Orienteering
try_times=range(20,100,20) #the time horizons which we try out

for total_time in try_times: #try it for different time horizons
    #ttotal_time=100
    
    cost_matrix=np.random.randint(1,15,(num_nodes,num_nodes)) #this will generate a random cost matrix.
    cost_matrix=cost_matrix + cost_matrix.T #ensure symmetry of the matrix
    
    score_vector=np.random.randint(1,5,(num_nodes)) #this will generate a random score matrix.
    score_vector[0]=0 #since the 0th node, start node, has no value!
    
    
    
    #sample matrix is given below
    
    #cost_matrix=np.array([[1,5,4,3],[3,1,8,2],[5,3,1,9],[6,4,3,4]]) #the true least cost is 12, this is used as a check.
    np.fill_diagonal(cost_matrix,1000) #make sure we don't travel from node to same node, by having high cost.
    
    
    x=c.Variable((num_nodes,num_nodes),boolean=True) #x_ij is 1, if we travel from i to j in the tour.
    
    u=c.Variable(num_nodes) #variables in subtour elimination constraints
    
    
    cost=c.trace(c.matmul(cost_matrix.T,x)) #total cost of the tour
    profit=c.sum(c.matmul(x,score_vector))
    
    ones_arr=np.ones([num_nodes]) #array for ones
      
    constraints=[]
    
    #now, let us make sure each node is visited only once, and we leave only once from that node.
    
    constraints.append(c.sum(x[0,:])==1)  #we leave from the first node
    constraints.append(c.sum(x[:,0])==1) #we come back to the first node
    
    constraints.append(c.matmul(x.T,ones_arr)<=1)  #max one connection outgoing and incoming
    constraints.append(c.matmul(x,ones_arr)<=1)
    
    for i in range(num_nodes):
        constraints.append(c.sum(x[:,i])==c.sum(x[i,:]))
    #let us add the time constraints
    
    constraints.append(cost<=total_time)
    #Let us add the subtour elimination constraints (Miller-Tucker-Zemlin similar formulation)
    
    for i in range(1,num_nodes):
        for j in range(1,num_nodes):
            if i!=j:
                constraints.append((u[i]-u[j]+num_nodes*x[i,j]-num_nodes+1<=0))
            else:
                continue
    
    prob=c.Problem(c.Maximize(profit),constraints)
    
    prob.solve()
    
    if np.any(x.value==None): #no feasible solution found!
        
        print('Feasible solution not found, lower your time constraint!') 
        sys.exit()
    
    #print(x.value.astype('int32'))
    #print(cost_matrix)
    
    
    tour=[0] #.the final tour we have found will be stored here. Initialize with start node.
    verified_cost=0 #builds up the cost of the tour, independently from the gurobi solver. We use this as a sanity check.
    
    now_node=0 #Initialize at start node
    
    g=nx.DiGraph()
    
    ##g.add_nodes_from(range(1,num_nodes+1))
    for k in range(num_nodes):
        g.add_node(k+1,value=score_vector[k]) #1 based indexing
    
    while(1): #till we reach end node
         
        next_node=np.argmax(x.value[now_node,:]) #where  we go from node i
        g.add_edge(now_node+1,next_node+1,weight=int(cost_matrix[now_node,next_node])) #1 based indexing graph
        verified_cost=verified_cost+cost_matrix[now_node,next_node] #build up the cost
        tour.append(next_node) #for 1 based indexing
        now_node=next_node
        if next_node==0: #we have looped again
            break
    
    print('The maximum profit tour found is ',end=" ")
    for idx,k in enumerate(tour):
        if idx!=len(tour)-1:
            print(k+1,end=" -> ")
        else:
            print(k+1)
    
    
    print('Profit of tour found by the solver is {}, cost of it is {}, and cost computed by us for verification is {} '.format(round(prob.value,2),cost.value,round(verified_cost,2)))
    print('Time taken to solve the problem instance after formulation is {} seconds'.format(round(prob.solver_stats.solve_time,2)))
    
    color_map=['red']*num_nodes
    color_map[0]='green'
    
    pos = nx.circular_layout(g)
    nodeval=nx.get_node_attributes(g,'value')
    nx.draw_circular(g,with_labels=True,node_color=color_map,node_size=1000,labels=nodeval)
    labels = nx.get_edge_attributes(g,'weight')
    nx.draw_networkx_edge_labels(g,pos,edge_labels=labels, width=20, edge_color='b')
    plt.savefig('foo.png')
    plt.title('Time horizon {}'.format(total_time))
    plt.show()
#print(cost_matrix)
#print(x.value)
#print(score_vector)
