#branch and bound mplp
#The question is how adding cuts changes the dynamics of branch and cut
#I do not know, so first i must build a branch and bound off the basic mplp iterration
#I do know the basic mplp iteration tends to converge to the marginal polytope solution 
#I do know mplp is substantially faster than the lp simplex solve owing to sparsity
#I do know adding additional constraints does something...
#How much that additional performance boost matters for branch and cut remains to be seen.

from queue import PriorityQueue, Queue, LifoQueue
import numpy as np
from scipy.optimize import linprog
import copy
from pgmpy.models import MarkovNetwork
from pgmpy.inference import Mplp
from pgmpy.factors.discrete import DiscreteFactor

class Node(object):
    def __init__(self, instance, fixing=None, metadata = None):
        self.instance = instance
        self.fixing = fixing
        self.metadata = metadata

class NodeFIFOQueue(object):
	def __init__(self):
		self.nodes = Queue()
		self.priorities = []

	def append(self, node, prioriy=1.0):
		self.nodes.put(node)
		self.priorities.append(prioriy)

	def sample(self):
		return self.nodes.get()

	def __len__(self):
		return self.nodes.qsize()

class NodeExpander(object):
    def __init__(self):
        pass

    def expandnode(self, node):
        # return expanded result and modify the node
        raise NotImplementedError


class MPLPexpander(NodeExpander):
    def __init__(self):
        NodeExpander.__init__(self)

    def expandnode(self, node):
        #A, b, c = node.A, node.b, node.c
        #feasible, objective, solution = GurobiIntSolve2(A, b, c)
        #instance is mplp object 
        mplp = node.instance
        fixing = node.fixing
        metadata = node.metadata

        #for now ignore fixing and metadata
        output = mplp.map_query(tighten_triplet = False)
        print('output: ', output)
        (objective, solution, fractional) = mplp.get_solution()
        print('in function solution: ', solution)
        feasible = True
        return feasible, objective, solution, fractional

    def modify(self,node,fixing):
    	#for now we have only one fixing--to be modified later
    	if len(fixing) == 1:
    		for fix in fixing:
    			pivot = fix
    			val = fixing[fix]
    	else: 
    		raise NotImplementedError
    	print('pivot: ', pivot)
    	print('val: ', val)
    	mplp = copy.deepcopy(node.instance)
    	#model modify 
    	model = mplp.model
    	#list of factors corresponding to neighbors of pivot
    	#TODO: change to set
    	neighbor_nodes = []
    	neighbor_edges = []
    	for factor in model.get_factors(pivot):
    		scope = copy.copy(factor.scope())
    		if len(scope) == 2:
    			neighbor_edges.append(factor)
    			scope.remove(pivot)
    			neighbor_nodes.append(mplp.factor_dict[frozenset(scope[0])])
    	pivot_factor = mplp.factor_dict[frozenset(pivot)]
    	#pivot factor is never modified in place
    	inter_val = pivot_factor.reduce([(pivot,val)],inplace=False)
    	pivot_val = (1/len(neighbor_nodes))*inter_val.values
    	#print('pivot_val: ', pivot_val)
    	for edge in neighbor_edges:
    		#new variable not part of the model 
    		red_edge = edge.reduce([(pivot,val)], inplace=False)
    		#print('reduced edge: ', red_edge)
    		red_edge.sum(pivot_val*red_edge.identity_factor())
    		#print('edge plus pivot val: ', red_edge)
    		#modified the neighbor nodes in place because they are not removed
    		for nnode in neighbor_nodes: 
    			if nnode.variables[0] == red_edge.variables[0]:
    				#print('add potential: ', edge)
    				#print('to node: ', nnode)
    				nnode.sum(red_edge, inplace=True)
    				#print('add red_edge to node: ', nnode)
    				break

    	for node in neighbor_nodes:
    		print('post mod neighbor node: ', node)
    	#TODO: remove stray factors in the model
    	for factor in model.get_factors(pivot): 
    		model.remove_factors(factor)

    	print('post mod model')
    	for factor in model.get_factors():
    		print('factor: ', factor)	
    	#for factor in model.get_factors():
    	#		if factor.scope() 
    	#intersection set variables modify 
    	mplp.intersection_set_variables = set()
    	for edge in model.edges():
    		mplp.intersection_set_variables.update([frozenset({edge[0]}), frozenset({edge[1]})])
    	cluster_copy = copy.deepcopy(mplp.cluster_set)
    	for cluster in cluster_copy:
    		print('cluster variables: ', cluster_copy[cluster].cluster_variables)
    		if pivot in cluster_copy[cluster].cluster_variables:
    			print('removal')
    			mplp.cluster_set.pop(cluster)
    	for cluster in mplp.cluster_set:
    		print('modified cluster set: ', mplp.cluster_set[cluster].variables)
    	#TODO: sum messages and objective new are not triplet enabled
    	mplp.sum_messages = {}
    	#initialize to node factors
    	for factor in model.get_factors():
    		scope = factor.scope()
    		if len(scope) == 1:
    			#never change these two lines syntactically
    			card = factor.get_cardinality(list(scope))[scope[0]]
    			mplp.sum_messages[frozenset(scope)] = DiscreteFactor(scope,[card],np.zeros(card))
    	#add in messages necessary because dual feasible messages are nonzero
    	for factor in model.get_factors():
    		scope = frozenset(factor.scope())
    		if len(scope) == 2:
    			sending_cluster = mplp.cluster_set[scope]
    			for intersection in sending_cluster.intersection_sets_for_cluster_c:
    				mplp.sum_messages[intersection] += sending_cluster.message_from_cluster[intersection]
    	#objective_new to replace objective
    	#will include node and edge objectives 
    	mplp.objective_new = {}
    	for factor in model.get_factors():
    		scope = factor.scope()
    		if len(scope) == 1:
    			mplp.objective_new[frozenset(scope)] = factor + mplp.sum_messages[frozenset(scope)]
    	mplp.dual_lp = sum(
    		[np.amax(mplp.objective_new[obj].values) for obj in mplp.objective_new]
    	)
    	mplp.factor_dict = {}
    	for factor in model.get_factors():
    		mplp.factor_dict[frozenset(factor.scope())] = factor
    	mplp.initial_model = copy.deepcopy(mplp.model)

#node should include a graph, some fixings, and metadata
#metada: for LP should include 
#metadata: for mplp could include warm start, additional messages 
#should include all constraints including triplets which is bigger than the graph
#node then calls the solver in mode
#mode: eitheer LP or mplp

class BB(object):
	#instance is an mplp object 
	def __init__(self,instance):
	 	self.instance = instance

	def run(self):  
		node = Node(self.instance)
		nodelist = NodeFIFOQueue()

		# create a list to keep track of fractional solution
		# to form the lower bound on the objective
		fractionalsolutions = []
		childrennodes = []
		expanded = []

		# create initial best obj and solution
		BestObjective = -np.inf
		BestSolution = None

		nodelist.append(node)

		# book keepinng
		timecount = 0
		ratios = []
		optimalitygap = []

		expander = MPLPexpander()


		#main loop
		iteration = 0
		while len(nodelist) >= 1:
			iteration += 1
			print('iteration: ', iteration)
			# pop a node
			node = nodelist.sample()
			feasible, objective, solution, fractional = expander.expandnode(node)
			if feasible:
				assert objective is not None


			# check if the popped node is the child node of some parent node
			for idx in range(len(childrennodes)):
				if childrennodes[idx][0] == node:
					expanded[idx][0] = 1
					if expanded[idx][1] == 1:
						# pop the corresponding child node
						childrennodes.pop(idx)
						expanded.pop(idx)
						fractionalsolutions.pop(idx)
					break
				elif childrennodes[idx][1] == node:
					expanded[idx][1] = 1
					if expanded[idx][0] == 1:
						# pop the corresponding child node
						childrennodes.pop(idx)
						expanded.pop(idx)
						fractionalsolutions.pop(idx)
					break
			
			# check cases
			#logic: objective (relaxation) is a lower bound on any integral solution of the children of the branch
			#however, there is an integral solution that is smaller than objective, hence prune node. 
			#logic is reversed for maximization in mplp-- what to do?
			#as a matter of principle bb should be for convex optimization 
			if feasible and objective < BestObjective:
				print('prune')
				# prune the node
				pass 
			elif not feasible:
				print('not feasible')
				# prune the node
				pass
			elif True:
				#right now the logic is inconsistent, the loop never terminates
		 		#because I round so the solution is always feasible/integral
		 		# check if better than current best
		 		if objective >= BestObjective:
		 			BestSolution = solution
		 			BestObjective = objective
		 		# the solution is not integer
				# need to branch
				# now we choose branching randomly
				# we choose branching based on how fraction variables are
				#note the fractional variable is a node potential and is not a probability 
				#i.e does not sum to 1-- unclear then this is essentially a heuristic on a heuristic
		 		prob = []
		 		for var in fractional:
		 			prob.append(fractional[var][1]/sum(fractional[var]))
		 		rounded = []
		 		for item in solution:
		 			rounded.append(solution[item])
		 		print('fractional: ', fractional)
		 		print('solution: ', solution)
		 		index = np.argmax(np.abs(np.array(rounded) - np.array(prob)))
		 		var_list = [set(var).pop() for var in fractional]
		 		print('var_list: ', var_list)
		 		#TODO: index is a number not the name of the 'node' 
		 		print(index)
		 		print('branching')
		 		#debugging line 
		 		index = 0
		 		fixing = {var_list[index]:0}
		 		mod_instance = expander.modify(node,fixing)
		 		node1 = Node(mod_instance)
		 		fixing = {var_list[index]:1}
		 		mod_instance = expander.modify(node,fixing)
		 		node2 = Node(mod_instance)
			else:
				raise NotImplementedError

				# # add the corresponding constraints and create nodes
				# lower_constraint = np.zeros(A.shape[1])
				# lower_constraint[index] = 1.0
		 		# lower = np.floor(solution[index])
		 		# Alower = np.vstack((A, lower_constraint))
		 		# blower = np.append(b, lower)
		 		# node1 = Node(Alower, blower, c, IPsolution)

		 		# upper_constraint = np.zeros(A.shape[1])
		 		# upper_constraint[index] = -1.0
				# upper = -np.ceil(solution[index])
		 		# Aupper = np.vstack((A, upper_constraint))
		 		# bupper = np.append(b, upper)
		 		# node2 = Node(Aupper, bupper, c, IPsolution)

		 		# # add nodes to the queue
		 		# nodelist.append(node1)
		 		# nodelist.append(node2)

		 		# # record the newly added child nodes and the fractional solution
		 		# fractionalsolutions.append(objective)
		 		# childrennodes.append([node1, node2])
		 		# expanded.append([0, 0])

			print('hello')

