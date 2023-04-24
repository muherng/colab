import copy
import itertools as it

import numpy as np
import networkx as nx

from pgmpy.inference import Inference
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor

import time 
import random

class Mplp(Inference):
    """
    Class for performing approximate inference using Max-Product Linear Programming method.
    We derive message passing updates that result in monotone decrease of the dual of the
    MAP LP Relaxation.
    Parameters
    ----------
    model: MarkovNetwork for which inference is to be performed.
    Examples
    --------
    >>> import numpy as np
    >>> from pgmpy.models import MarkovNetwork
    >>> from pgmpy.inference import Mplp
    >>> from pgmpy.factors.discrete import DiscreteFactor
    >>> student = MarkovNetwork()
    >>> student.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('E', 'F')])
    >>> factor_a = DiscreteFactor(['A'], cardinality=[2], values=np.array([0.54577, 1.8323]))
    >>> factor_b = DiscreteFactor(['B'], cardinality=[2], values=np.array([0.93894, 1.065]))
    >>> factor_c = DiscreteFactor(['C'], cardinality=[2], values=np.array([0.89205, 1.121]))
    >>> factor_d = DiscreteFactor(['D'], cardinality=[2], values=np.array([0.56292, 1.7765]))
    >>> factor_e = DiscreteFactor(['E'], cardinality=[2], values=np.array([0.47117, 2.1224]))
    >>> factor_f = DiscreteFactor(['F'], cardinality=[2], values=np.array([1.5093, 0.66257]))
    >>> factor_a_b = DiscreteFactor(['A', 'B'], cardinality=[2, 2],
    ...                             values=np.array([1.3207, 0.75717, 0.75717, 1.3207]))
    >>> factor_b_c = DiscreteFactor(['B', 'C'], cardinality=[2, 2],
    ...                             values=np.array([0.00024189, 4134.2, 4134.2, 0.00024189]))
    >>> factor_c_d = DiscreteFactor(['C', 'D'], cardinality=[2, 2],
    ...                             values=np.array([0.0043227, 231.34, 231.34, 0.0043227]))
    >>> factor_d_e = DiscreteFactor(['E', 'F'], cardinality=[2, 2],
    ...                             values=np.array([31.228, 0.032023, 0.032023, 31.228]))
    >>> student.add_factors(factor_a, factor_b, factor_c, factor_d, factor_e, factor_f, factor_a_b,
    ...                     factor_b_c, factor_c_d, factor_d_e)
    >>> mplp = Mplp(student)
    """

    def __init__(self, model):
        if not isinstance(model, MarkovNetwork):
            raise TypeError("Only MarkovNetwork is supported")

        super(Mplp, self).__init__(model)
        self._initialize_structures()

        # S = \{c \cap c^{'} : c, c^{'} \in C, c \cap c^{'} \neq \emptyset\}
        self.intersection_set_variables = set()
        # We generate the Intersections of all the pairwise edges taken one at a time to form S
        for edge in model.edges():
            self.intersection_set_variables.update(
               [frozenset({edge[0]}), frozenset({edge[1]})]
            )
            

        # The corresponding optimization problem = \min_{\delta}{dual_lp(\delta)} where:
        # dual_lp(\delta) = \sum_{i \in V}{max_{x_i}(Objective[nodes])} + \sum_{f /in F}{max_{x_f}(Objective[factors])
        # Objective[nodes] = \theta_i(x_i) + \sum_{f \mid i \in f}{\delta_{fi}(x_i)}
        # Objective[factors] = \theta_f(x_f) - \sum_{i \in f}{\delta_{fi}(x_i)}
        # In a way Objective stores the corresponding optimization problem for all the nodes and the factors.

        # Form Objective and cluster_set in the form of a dictionary.
        self.objective = {}
        self.cluster_set = {}
        for factor in model.get_factors():
            scope = frozenset(factor.scope())
            self.objective[scope] = factor
            # For every factor consisting of more that a single node, we initialize a cluster.
            if len(scope) > 1:
                self.cluster_set[scope] = self.Cluster(
                    self.intersection_set_variables, factor
                )

        #the sum of all messages \sum_{k \in N(i)} lambda_(ik)->i(x_i)
        self.sum_messages = {}
        #initialize to node factors
        for factor in model.get_factors():
            scope = factor.scope()
            if len(scope) == 1:
                #never change these two lines syntactically
                card = factor.get_cardinality(list(scope))[scope[0]]
                self.sum_messages[frozenset(scope)] = DiscreteFactor(scope,[card],np.zeros(card))

        #add in messages necessary because dual feasible messages are nonzero
        for factor in model.get_factors():
            scope = frozenset(factor.scope())
            if len(scope) == 2:
                sending_cluster = self.cluster_set[scope]
                for intersection in sending_cluster.intersection_sets_for_cluster_c:
                    self.sum_messages[intersection] += sending_cluster.message_from_cluster[intersection]

        #objective_new to replace objective
        #will include node and edge objectives 
        self.objective_new = {}
        for factor in model.get_factors():
            scope = factor.scope()
            if len(scope) == 1:
                self.objective_new[frozenset(scope)] = factor + self.sum_messages[frozenset(scope)]

        # dual_lp(\delta) is the dual linear program
        self.dual_lp = sum(
            [np.amax(self.objective_new[obj].values) for obj in self.objective_new]
        )
        print('dual_lp: ', self.dual_lp)

        # Best integral value of the primal objective is stored here
        self.best_int_objective = 0

        # Assignment of the nodes that results in the "maximum" integral value of the primal objective
        self.best_assignment = {}
        # This sets the minimum width between the dual objective decrements. Default value = 0.0002. This can be
        # changed in the map_query() method.
        self.dual_threshold = 0.0002
        # This sets the threshold for the integrality gap below which we say that the solution is satisfactory.
        # Default value = 0.0002. This can be changed in the map_query() method.
        self.integrality_gap_threshold = 0.0002

        #flag for the updates before adding triplets
        self.triplet_mode = False

        #possibly the same as cluster_potential if cluster_potential never changes (TODO)
        self.factor_dict = {}
        for factor in model.get_factors():
            self.factor_dict[frozenset(factor.scope())] = factor

        self.initial_model = copy.deepcopy(self.model)

        #print('self.factor_dict: ', self.factor_dict)

    class Cluster(object):
        """
        Inner class for representing a cluster.
        A cluster is a subset of variables.
        Parameters
        ----------
        set_of_variables: tuple
            This is the set of variables that form the cluster.
        intersection_set_variables: set containing frozensets.
            collection of intersection of all pairs of cluster variables. For eg: \{\{C_1 \cap C_2\}, \{C_2 \cap C_3\}, \{C_3 \cap C_1\} \} for clusters C_1, C_2 & C_3.
        cluster_potential: DiscreteFactor
            Each cluster has an initial probability distribution provided beforehand.
        """

        def __init__(self, intersection_set_variables, cluster_potential):
            """
            Initialization of the current cluster
            """

            # The variables with which the cluster is made of.
            self.cluster_variables = frozenset(cluster_potential.scope())
            # The cluster potentials must be specified before only.
            self.cluster_potential = copy.deepcopy(cluster_potential)
            # Generate intersection sets for this cluster; S(c)
            #small routine for setting intersect_for_cluster_c as a set then a list
            #strongly consider doing this in one chunk of code 
            intersect_for_cluster = set()
            for intersect in intersection_set_variables:
                inter_cluster = intersect.intersection(self.cluster_variables)
                if inter_cluster:
                    intersect_for_cluster.update([inter_cluster])

            self.intersection_sets_for_cluster_c = [
                inter for inter in intersect_for_cluster
            ]

            # Initialize messages from this cluster to its respective intersection sets
            if len(self.cluster_variables) == 2:
                #initialize random dual feasible messages  
                val_a = np.zeros((2,2))
                val_b = np.zeros((2,2))
                for i in range(2):
                    for j in range(2):
                        upper = cluster_potential.values[i][j]
                        interval = np.random.uniform(0,upper)
                        val_a[i][j] = interval
                        val_b[i][j] = upper - interval
                #val_a = 1/2 * cluster_potential.values
                #val_b = 1/2 * cluster_potential.values
                arg_a = DiscreteFactor(cluster_potential.scope(), cluster_potential.cardinality, val_a)
                arg_b = DiscreteFactor(cluster_potential.scope(), cluster_potential.cardinality, val_b)
                arg_list = [arg_a, arg_b]
                self.message_from_cluster = {}
                index = -1
                for intersection in self.intersection_sets_for_cluster_c:
                    index += 1
                    arg = arg_list[index].maximize(list(self.cluster_variables - intersection), inplace=False)
                    self.message_from_cluster[intersection] = arg
                    #TODO debug
            if len(self.cluster_variables) == 3:
                self.message_from_cluster = {}
                for intersection in self.intersection_sets_for_cluster_c:
                    present_variables_card = cluster_potential.get_cardinality(
                        intersection
                    )
                    present_variables_card = [
                        present_variables_card[var] for var in intersection
                    ]
                    # We need to create a new factor whose messages are blank
                    # Edit: creat message that is dual feasible equal to theta_ij/2
                    empty_triplet_edge = DiscreteFactor(
                        intersection, present_variables_card, np.zeros(np.prod(present_variables_card))
                    )
                    self.message_from_cluster[frozenset(intersection)] = empty_triplet_edge

    def _update_message(self, sending_cluster):
        #edge to edge and edge to node messages sent simultaneously
        #first compute edge to node update
        #we assemble theta_i, theta_j, theta_ij, lambda_{c -> ij}, lambda_i^{-j}(x_i)

        theta = {}
        edge = frozenset(sending_cluster.cluster_variables)
        theta[edge] = self.factor_dict[edge]
        #print('theta[edge]: ', theta[edge])
        objective_node = {}
        objective_cluster = theta[edge]
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            theta[current_intersect] = self.factor_dict[frozenset(current_intersect)]
            message_minus = self.sum_messages[current_intersect] \
                + -1*sending_cluster.message_from_cluster[current_intersect] 
            objective_node[current_intersect] = message_minus + theta[current_intersect]
            objective_cluster += objective_node[current_intersect]
        
        updated_results = []
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            phi = objective_cluster.maximize(
                list(sending_cluster.cluster_variables - current_intersect),
                inplace=False,
            )
            phi *= 1 / len(sending_cluster.cluster_variables)
            # Step. 4) Subtract \delta_i^{-f}
            updated_results.append(
                phi + -1 * objective_node[current_intersect]
            )

        index = -1
        #copy of the factor  
        cluster_potential = copy.deepcopy(sending_cluster.cluster_potential)
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            index += 1
            message_old = copy.deepcopy(sending_cluster.message_from_cluster[current_intersect])
            message_new = updated_results[index]
            sending_cluster.message_from_cluster[current_intersect] = message_new
            self.sum_messages[current_intersect] += -1*message_old + message_new
            self.objective_new[current_intersect] = theta[current_intersect] + self.sum_messages[current_intersect]
        return 0

    def _update_message_edge_edge(self, sending_cluster):
        #edge to node messages sent simultaneously
        #first compute edge to node update
        #we assemble theta_i, theta_j, theta_ij, lambda_{c -> ij}, lambda_i^{-j}(x_i)
        #edge to node updates
        theta = {}
        edge = frozenset(sending_cluster.cluster_variables)
        theta[edge] = self.factor_dict[edge]
        objective_cluster = theta[edge]
        objective_node = {}
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            theta[current_intersect] = self.factor_dict[frozenset(current_intersect)]
            message_minus = self.sum_messages[current_intersect] \
                + -1*sending_cluster.message_from_cluster[current_intersect] 
            objective_node[current_intersect] = message_minus + theta[current_intersect] 
            objective_cluster += objective_node[current_intersect]
        
        objective_cluster += self.sum_messages[edge]
        
        updated_results = []
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            phi = objective_cluster.maximize(
                list(sending_cluster.cluster_variables - current_intersect),
                inplace=False,
            )
            phi *= 1 / 2
            # Step. 4) Subtract \delta_i^{-f}
            updated_results.append(
                phi + -1 * objective_node[current_intersect]
            )


        index = -1
        #update messages, sum messages into both nodes, and edge objective
        cluster_potential = copy.deepcopy(sending_cluster.cluster_potential)
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            index += 1
            message_old = copy.deepcopy(sending_cluster.message_from_cluster[current_intersect])
            message_new = updated_results[index]
            sending_cluster.message_from_cluster[current_intersect] = message_new
            self.sum_messages[current_intersect] += -1*message_old + message_new
            self.objective_new[current_intersect] = theta[current_intersect] + self.sum_messages[current_intersect]

        #update edge objective
        edge_objective = theta[edge]
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            edge_objective += -1*sending_cluster.message_from_cluster[current_intersect]
        edge_objective += self.sum_messages[edge]
        self.objective_new[edge] = edge_objective          

        return 0

    def _update_message_triplet_edge(self, sending_cluster):
        
        triplet = frozenset(sending_cluster.cluster_variables)
        cluster_potential = sending_cluster.cluster_potential
        cardinalities = cluster_potential.cardinality

        #defined to be lambda_e'->e'(x_e') + \sum_{c' \neq c and e' \in c'} lambda_{c'->e'}(x_e')
        objective_cluster = DiscreteFactor(triplet,cardinalities,np.zeros(np.prod(cardinalities)))  
        edge_terms = {}
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            edge_terms[frozenset(current_intersect)] = self.factor_dict[frozenset(current_intersect)]
            edge_cluster = self.cluster_set[frozenset(current_intersect)]
            for node in current_intersect:
                edge_terms[frozenset(current_intersect)] += -1*edge_cluster.message_from_cluster[frozenset([node])]
            message_minus = self.sum_messages[frozenset(current_intersect)]\
                             + -1*sending_cluster.message_from_cluster[frozenset(current_intersect)] 
            edge_terms[frozenset(current_intersect)] += message_minus
            objective_cluster += edge_terms[frozenset(current_intersect)]

        updated_results = []
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            phi = objective_cluster.maximize(
                list(sending_cluster.cluster_variables - current_intersect),
                inplace=False,
            )
            phi *= 1/3
            # Step. 4) Subtract \delta_i^{-f}
            updated_results.append(
                phi + -1*edge_terms[frozenset(current_intersect)]
            )

        index = -1
        #copy of the factor  
        cluster_potential = copy.deepcopy(sending_cluster.cluster_potential)
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            index += 1
            message_old = copy.deepcopy(sending_cluster.message_from_cluster[current_intersect])
            message_new = updated_results[index]
            sending_cluster.message_from_cluster[current_intersect] = message_new  

            self.sum_messages[current_intersect] += -1*message_old + message_new
            self.objective_new[current_intersect] += -1*message_old + message_new
            #Note the order of the signs, this is tricky, consider changing
            self.objective_new[triplet] += message_old + -1*message_new

        return 0


    def _local_decode(self):
        """
        Finds the index of the maximum values for all the single node dual objectives.
        Reference:
        code presented by Sontag in 2012 here: http://cs.nyu.edu/~dsontag/code/README_v2.html
        """
        # The current assignment of the single node factors is stored in the form of a dictionary
        decoded_result_assignment = {
            node: np.argmax(self.objective_new[node].values)
            for node in self.objective_new
            if len(node) == 1
        }
        # Use the original cluster_potentials of each factor to find the primal integral value.
        # 1. For single node factors
        integer_value = sum(
            [
                self.factors[variable][0].values[
                    decoded_result_assignment[frozenset([variable])]
                ]
                for variable in self.variables
            ]
        )

        # 2. For clusters
        for cluster_key in self.cluster_set:
            if len(cluster_key) == 2:
                cluster = self.cluster_set[cluster_key]
                index = [
                    tuple([variable, decoded_result_assignment[frozenset([variable])]])
                    for variable in cluster.cluster_variables
                ]
                integer_value += cluster.cluster_potential.reduce(
                    index, inplace=False
                ).values

        # Check if this is the best assignment till now
        if self.best_int_objective < integer_value:
            self.best_int_objective = integer_value
            self.best_assignment = decoded_result_assignment

        print('best_int_objective: ', self.best_int_objective)

    #TODO: this function has not been implemented successfully  
    def check_feasibility(self):
        dual_dict = {}
        for factor in self.model.get_factors():
            scope = factor.scope()
            if len(scope) == 1:
                dual_dict[frozenset(scope)] = DiscreteFactor(frozenset(scope), [2], np.zeros(2))
            if len(scope) == 2:
                dual_dict[frozenset(scope)] = DiscreteFactor(frozenset(scope), [2,2], np.zeros(4))
        for factor in self.model.get_factors():
            scope = factor.scope()
            if len(scope) >= 2:
                sending_cluster = self.cluster_set[frozenset(scope)]
                theta_constraint = DiscreteFactor(frozenset(scope),[2,2],np.zeros(4))
                message_list = []
                for var in scope:
                    #nefarious bug fix 
                    var = [var]
                    message = sending_cluster.message_from_cluster[frozenset(var)]
                    theta_constraint += message
                    dual_dict[frozenset(var)] += message
                    message_list.append(message)
        for factor in self.model.get_factors():
            scope = factor.scope()
            if len(scope) == 1:
                if dual_dict[frozenset(scope)] != self.objective_new[frozenset(scope)]:
                    print('dual_dict[frozenset(scope)]: ', dual_dict[frozenset(scope)])
                    print('self.objective_new[frozenset(scope)] :', self.objective_new[frozenset(scope)])
                    raise ValueError('OBJECTIVE NEW IS FAULTY')

        return 0

    def _is_converged(self, dual_threshold=None, integrality_gap_threshold=None):
        """
        This method checks the integrality gap to ensure either:
            * we have found a near to exact solution or
            * stuck on a local minima.
        Parameters
        ----------
        dual_threshold: double
                        This sets the minimum width between the dual objective decrements. If the decrement is lesser
                        than the threshold, then that means we have stuck on a local minima.
        integrality_gap_threshold: double
                                   This sets the threshold for the integrality gap below which we say that the solution
                                   is satisfactory.
        References
        ----------
        code presented by Sontag in 2012 here: http://cs.nyu.edu/~dsontag/code/README_v2.html
        """
        # Find the new objective after the message updates
        #for obj in self.objective:
        #    print('self.objective[obj].values: ', self.objective[obj].values)

        new_dual_lp = sum(
            [np.amax(self.objective_new[obj].values) for obj in self.objective_new]
        )

        print('dual objective previous: ', self.dual_lp)
        print('dual objective new: ', new_dual_lp)
        # Update the dual_gap as the difference between the dual objective of the previous and the current iteration.
        self.dual_gap = abs(self.dual_lp - new_dual_lp)

        # Update the integrality_gap as the difference between our best result vs the dual objective of the lp.
        self.integrality_gap = abs(self.dual_lp - self.best_int_objective)
        print('integrality gap: ', self.integrality_gap)
        # As the decrement of the dual_lp gets very low, we assume that we might have stuck in a local minima.
        if dual_threshold and self.dual_gap < dual_threshold:
            return True
        # Check the threshold for the integrality gap
        elif (
            integrality_gap_threshold
            and self.integrality_gap < integrality_gap_threshold
        ):
            return True
        else:
            self.dual_lp = new_dual_lp
            return False

    def find_triangles(self,inter_size=1):
        """
        Finds all the triangles present in the given model
        Examples
        --------
        >>> from pgmpy.models import MarkovNetwork
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.inference import Mplp
        >>> mm = MarkovNetwork()
        >>> mm.add_nodes_from(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
        >>> mm.add_edges_from([('x1', 'x3'), ('x1', 'x4'), ('x2', 'x4'),
        ...                    ('x2', 'x5'), ('x3', 'x6'), ('x4', 'x6'),
        ...                    ('x4', 'x7'), ('x5', 'x7')])
        >>> phi = [DiscreteFactor(edge, [2, 2], np.random.rand(4)) for edge in mm.edges()]
        >>> mm.add_factors(*phi)
        >>> mplp = Mplp(mm)
        >>> mplp.find_triangles()
        """
        #this enumeration will not include triplets with no existing edges between them
        (nodes,edges,model_triplets) = self.get_factors_by_type()
        triplets = set()
        for edge in edges:
            #edge is of type frozenset frozenset{i,j}
            for node in nodes:
                #node is of type string 
                if node not in set(edge):
                    tri = set()
                    tri = tri | edge | set([node])
                    if len(tri) != 3:
                        raise ValueError('NOT A TRIPLET')
                    size = 0 
                    intersects = []
                    for pair in it.combinations(tri,2):
                        if set(pair) in edges:
                            size += 1
                            intersects.append(set(pair))
                    if size >= inter_size and frozenset(tri) not in model_triplets:
                        triplets.add(frozenset(tri))

        return list(triplets)  
    
    #replace get_edges with this function
    #node is a set of strings 
    #edges and triplets are sets of frozensets i.e frozenset({i,j,k}) 
    def get_factors_by_type(self): 
        nodes = set()
        edges = set()
        triplets = set()
        for factor in self.model.get_factors():
            scope = factor.scope()
            if len(scope) == 1:
                nodes.add(scope[0])
            if len(scope) == 2:
                edges.add(frozenset(scope))
            if len(scope) == 3:
                triplets.add(frozenset(scope))
        return (nodes,edges,triplets)

    #return set of edges
    def get_edges(self):
        edges = set()
        for factor in self.model.get_factors():
            scope = factor.scope()
            if len(scope) == 2:
                edges.add(frozenset(scope))
        return edges

    def _update_triangles(self, triangles_list):
        """
        From a set of variables forming a triangle in the model, we form the corresponding Clusters.
        These clusters are then appended to the code.
        Parameters
        ----------
        triangle_list : list
                        The list of variables forming the triangles to be updated. It is of the form of
                        [['var_5', 'var_8', 'var_7'], ['var_4', 'var_5', 'var_7']]
        """
        print('update triangles')
        edges = self.get_edges()
        new_intersection_set = []
        for triangle_vars in triangles_list:
            cardinalities = [self.cardinality[variable] for variable in triangle_vars]
            triplet_intersections = []
            
            #add in all three edges in triplet into intersection sets 
            for pair in it.combinations(triangle_vars, 2):
                triplet_intersections.append(pair)

            current_intersection_set = [
                frozenset(intersect) for intersect in triplet_intersections
            ]
            current_factor = DiscreteFactor(
                triangle_vars, cardinalities, np.zeros(np.prod(cardinalities))
            )
            self.cluster_set[frozenset(triangle_vars)] = self.Cluster(
                current_intersection_set, current_factor
            )

            # add new factor and cluster for triplet
            #consider adding to factor dict, but so far not used
            self.model.factors.append(current_factor)
            self.objective_new[frozenset(triangle_vars)] = DiscreteFactor(list(triangle_vars),
                cardinalities,np.zeros(np.prod(cardinalities)))

            #add factor and cluster for ghost edges
            for intersect in triplet_intersections:
                if frozenset(intersect) not in edges:
                    print('intersect: ', intersect)
                    card = [self.cardinality[variable] for variable in intersect] 
                    new_edge = DiscreteFactor(list(intersect), card, np.zeros(np.prod(card)))
                    self.model.factors.append(new_edge)
                    self.factor_dict[frozenset(intersect)] = new_edge
                    edge_inter = [frozenset([var]) for var in intersect]
                    self.cluster_set[frozenset(intersect)] = self.Cluster(
                            edge_inter, new_edge 
                        )
                    self.objective_new[frozenset(intersect)] = DiscreteFactor(list(intersect), 
                        card, np.zeros(np.prod(card))) 
                    self.sum_messages[frozenset(intersect)] =  DiscreteFactor(list(intersect), 
                        card, np.zeros(np.prod(card)))


    def _get_triplet_scores(self, triangles_list):
        """
        Returns the score of each of the triplets found in the current model
        Parameters
        ---------
        triangles_list: list
                        The list of variables forming the triangles to be updated. It is of the form of
                        [['var_5', 'var_8', 'var_7'], ['var_4', 'var_5', 'var_7']]
        Return: {frozenset({'var_8', 'var_5', 'var_7'}): 5.024, frozenset({'var_5', 'var_4', 'var_7'}): 10.23}
        """
        triplet_scores = {}
        for triplet in triangles_list:
            if len(triplet) != 3:
                print(triplet)
                raise ValueError('GET TRIPLET SCORES NOT TRIPLET')
            # Find the intersection sets of the current triplet
            edges = self.get_edges()
            triplet_intersections = []
            
            for pair in it.combinations(triplet, 2):
                if frozenset(pair) in edges:
                    triplet_intersections.append(pair)

            #Independent maximization
            #independent maximization is over the edges in the triplet that currently 
            #exist in the model, the ghost edge has objective zero and makes no difference
            #sum over three edges and the triplet objective 
            #subtle point: triplet objective is always zero before adding triplet so it is omitted.  
            ind_max = sum(
                [
                    np.amax(self.objective_new[frozenset(intersect)].values)
                    for intersect in triplet_intersections
                ]
            )


            # Joint maximization
            joint_max = self.objective_new[frozenset(triplet_intersections[0])]
            if len(triplet_intersections) >= 1: 
                for intersect in triplet_intersections[1:]:
                    joint_max += self.objective_new[frozenset(intersect)]
            joint_max = np.amax(joint_max.values)
            # score = Independent maximization solution - Joint maximization solution
            score = ind_max - joint_max
            if len(triplet) != 3:
                print(triplet)
                raise ValueError('END OF FUNCTION GET TRIPLET SCORES NOT TRIPLET')
            triplet_scores[frozenset(triplet)] = score

        return triplet_scores

    def _run_mplp(self, no_iterations):
        """
        Updates messages until either Mplp converges or if it doesn't converge; halts after no_iterations.
        Parameters
        --------
        no_iterations:  integer
                        Number of maximum iterations that we want MPLP to run.
        """
        for niter in range(no_iterations):
            # We take the clusters in the order they were added in the model and update messages for all factors whose
            # scope is greater than 1
            print('iteration: ', niter)
            for factor in self.model.get_factors():
                scope = factor.scope()
                if len(scope) == 2:
                    edge = frozenset(factor.scope())
                    sending_cluster = self.cluster_set[edge]
                    if self.triplet_mode:
                        start = time.time()
                        self._update_message_edge_edge(sending_cluster)
                        end = time.time()
                        #print('time edge: ', end - start)
                    else :
                        start = time.time()
                        self._update_message(sending_cluster)
                        end = time.time()
                        #print('time edge: ', end - start)
                if len(scope) == 3:
                    triplet = frozenset(factor.scope())
                    sending_cluster = self.cluster_set[triplet]
                    start = time.time()
                    self._update_message_triplet_edge(sending_cluster)
                    end = time.time()
                    #print('time triplet to edge: ', end - start)
            # Find an integral solution by locally maximizing the single node beliefs
            self._local_decode()
            #if niter % 10 == 0:
                 #for node in self.objective_new:
                 #   if len(node) == 1: 
                 #       print(self.objective_new[node].values)
            # If mplp converges to a global/local optima, we break.
            if (
                self._is_converged(self.dual_threshold, self.integrality_gap_threshold)
                and niter >= 16
            ):
                break

    #helper function for changing mode from max product to add triplet mode.  
    def triplet_setup(self):
        #sum messages from triplet to edge initialize to zero for each edge
        for factor in self.model.get_factors():  
            scope = factor.scope()
            if len(scope) == 2:
                card = factor.get_cardinality(list(scope))[scope[0]]
                self.sum_messages[frozenset(scope)] = DiscreteFactor(scope,[card,card],np.zeros(card**2)) 

        #objective initialize for each edge 
        for factor in self.model.get_factors():
            scope = factor.scope()
            if len(scope) == 2:
                edge_cluster = self.cluster_set[frozenset(scope)]
                theta = edge_cluster.cluster_potential
                for intersect in edge_cluster.intersection_sets_for_cluster_c:
                    theta = theta + -1*edge_cluster.message_from_cluster[intersect]
                #no need to add sum triplet messages into edge because they are zero
                self.objective_new[frozenset(scope)] = theta

    def mode_random(self,triangles):
        
        triplet_scores = self._get_triplet_scores(triangles)
        import random
        add_triplets = list(set(random.choices(triplet_scores,k=max_triplets)))
        
        return add_triplets

    def mode_eval(self, triangles, max_triplets): 
        # Evaluate scores for each of the triplets found above
        add_triplets = []
        triplet_scores = self._get_triplet_scores(triangles)
        print('Eval Mode')
        sorted_scores = sorted(triplet_scores.items(), key=lambda x:x[1])
        for item in sorted_scores:
            if item[1] > 0.01:
                print(item[1])
        #sorted_scores = sorted(triplet_scores, key=triplet_scores.get) 
        for triplet_number in range(max_triplets):
            (addition,dec) = sorted_scores.pop()
            add_triplets.append(addition)
            print('addition: ', (addition,dec))
            print('shortest path: ', self.paths(addition))

        return add_triplets 

    def mode_greedy(self, triangles,sorted_scores,max_triplets):
        add_triplets = []
        for triplet_number in range(len(sorted_scores)):
            # At once, we can add at most 5 triplets
            if triplet_number >= max_triplets:
                print('break triplet number')
                break
            addition = sorted_scores.pop()
            if len(addition) != 3:
                raise ValueError("pop not triplet")
            size = 0
            edges = self.get_edges()
            for pair in it.combinations(addition,2):
                if frozenset(pair) in edges:
                    size += 1
            if size < 1:
                raise ValueError('tighten triplet less than one intersects')
            add_triplets.append(addition)

        return add_triplets 

    def mode_mcmc(self):
        raise NotImplementedError
        #self.get_edge_pairs() 
        #import random

    def mode_subsample(self,triangles,sorted_scores, max_triplets, fraction): 
        add_triplets = []
        (_,_,trip_exist) = self.get_factors_by_type()
        import random
        subsample = int(len(triangles)*fraction)
        tri_sub = list(random.choices(triangles,k=subsample))
        triplet_scores = self._get_triplet_scores(tri_sub)
        sorted_scores = sorted(triplet_scores.items(), key=lambda x:x[1])
        for triplet_number in range(max_triplets):
            (addition,dec) = sorted_scores.pop()
            #TODO: check for redundancy
            if addition in trip_exist:
                continue
            add_triplets.append(addition)
        return add_triplets

    def _tighten_triplet(self, max_iterations, later_iter, max_triplets, prolong, mode):
        """
        This method finds all the triplets that are eligible and adds them iteratively in the bunch of max_triplets
        Parameters
        ----------
        max_iterations: integer
                        Maximum number of times we tighten the relaxation
        later_iter: integer
                    Number of maximum iterations that we want MPLP to run. This is lesser than the initial number
                    of iterations.
        max_triplets: integer
                      Maximum number of triplets that can be added at most in one iteration.
        prolong: bool
                It sets the continuation of tightening after all the triplets are exhausted
        mode: string 
                'cycle' mode adds the cycle triplets.  'greedy' mode adds the triplets in a greedy fashion and fails.  
        """
        # Find all the triplets that are possible in the present model
        #note that the dual objective could increase because we add in edge objective 
        #of the form theta_ij - delta(j -> i) - \delta(i -> j) 
        #which is not guaranteed to be zero.  
        #triplet setup runs only once when switching from max product to triplet mode
        self.triplet_setup()
        start = time.time()
        triangles = self.find_triangles()
        print('length triangles: ', len(triangles))
        # Evaluate scores for each of the triplets found above
        triplet_scores = self._get_triplet_scores(triangles)
        end = time.time()
        print('TIME FOR SORT: ', end-start)
        # Arrange the keys on the basis of increasing order of th3e values of the dict. triplet_scores
        sorted_scores = sorted(triplet_scores, key=triplet_scores.get) 
        print("SORTED SCORES")
        all_score = []
        for val in triplet_scores.values():
            all_score.append(val)
        all_score.sort()
        #print(all_score)
        #find cycles only once when flipping from mplp to triplet mode
        for niter in range(max_iterations):
            if self._is_converged(
                integrality_gap_threshold=self.integrality_gap_threshold
            ):
                break 
            add_triplets = []
            triangles = self.find_triangles()
            print('length triangles: ', len(triangles))
            if mode == 'mcmc': 
                #generate k triangles
                #pick an edge per triangle to move
                #switch with weighted probability
                #not implemented 
                raise NotImplementedError
                #add_triplets = self.mode_mcmc(triangles)
            if mode == 'subsample':
                add_triplets = self.mode_subsample(triangles,sorted_scores, max_triplets, 0.1)
            if mode == 'random':
                add_triplets = self.mode_random(triangles,max_triplets)
            if mode == 'eval':
                add_triplets = self.mode_eval(triangles, max_triplets)
            if mode == 'greedy-hack': 
                add_triplets = self.mode_greedy(triangles,sorted_scores, max_triplets)
            if mode == 'cycle': 
                add_triplets = []
                cycles = self.cycles()
                tri_cycles = self.cycle_to_triangle(cycles)
                for tri in tri_cycles:
                    add_triplets += tri
                add_triplets = self.remove_duplicates(add_triplets)
            # Update the eligible triplets to tighten the relaxation
            print('add_triplets: ', add_triplets)
            self._update_triangles(add_triplets)
            # Run MPLP for a maximum of later_iter times.
            self._run_mplp(later_iter)

    def paths(self,triplet):
        (nodes,edges,_) = self.get_factors_by_type()
        nodes = list(nodes)
        #print(nodes)
        edges = list(edges)
        edges = [tuple(edge) for edge in edges]
        import networkx as nx
        G = nx.Graph() 
        G.add_nodes_from(nodes) 
        G.add_edges_from(edges)
        triplet = list(triplet)
        pairs = []
        for val1 in triplet:
            for val2 in triplet:
                if val1 != val2: 
                    pairs.append(len(nx.shortest_path(G,str(val1), str(val2))))
    
        return max(pairs)


    def remove_duplicates(self,triplet_list):
        (nodes,edges,triplets) = self.get_factors_by_type()
        return [tri for tri in triplet_list if tri not in triplets]

    def cycles(self):
        #find gap achieved by adding cycle
        (nodes,edges,_) = self.get_factors_by_type()
        nodes = list(nodes)
        edges = list(edges)
        edges = [tuple(edge) for edge in edges]
        import networkx as nx
        G = nx.Graph() 
        G.add_nodes_from(nodes) 
        G.add_edges_from(edges)
        cycles = nx.cycle_basis(G)

        return cycles

    def cycle_to_triangle(self,cycles): 
        #list of list of triangle frozensets
        cycle_tri = []
        for cycle in cycles:
            #arbitrary root
            root = random.choice(cycle)
            re_root = []
            for i in range(len(cycle)):
                elem = cycle[i]
                if elem == root:
                    re_root += cycle[i:]
                    re_root += cycle[:i]
            #list of sets
            tri_set = []
            index = -1
            if len(re_root) >= 3:
                #first = True
                for i in range(2,len(re_root)):
                    triangle = set()
                    triangle.add(root)
                    triangle.update([re_root[i-1],re_root[i]])
                    tri_set.append(frozenset(triangle))
            cycle_tri.append(tri_set) 

        return cycle_tri


    def get_integrality_gap(self):
        
        return self.integrality_gap

    def get_solution(self):
        #returns a tuple of (rounded solution, fractional solution)
        fractional = {
            node: self.objective_new[node].values
            for node in self.objective_new
            if len(node) == 1
        }
        return (self.dual_lp, self.best_assignment, fractional) 

    def query(self):
        raise NotImplementedError("map_query() is the only query method available.")

    def map_query(
        self,
        init_iter=100,
        later_iter=20,
        dual_threshold=0.005,
        integrality_gap_threshold=0.005,
        tighten_triplet=True,
        max_triplets=30,
        max_iterations=100,
        prolong=False,
        mode='cycle'
    ):
        """
        MAP query method using Max Product LP method.
        This returns the best assignment of the nodes in the form of a dictionary.
        Parameters
        ----------
        init_iter: integer
                   Number of maximum iterations that we want MPLP to run for the first time.
        later_iter: integer
                    Number of maximum iterations that we want MPLP to run for later iterations
        dual_threshold: double
                        This sets the minimum width between the dual objective decrements. If the decrement is lesser
                        than the threshold, then that means we have stuck on a local minima.
        integrality_gap_threshold: double
                                   This sets the threshold for the integrality gap below which we say that the solution
                                   is satisfactory.
        tighten_triplet: bool
                         set whether to use triplets as clusters or not.
        max_triplets: integer
                      Set the maximum number of triplets that can be added at once.
        max_iterations: integer
                        Maximum number of times we tighten the relaxation. Used only when tighten_triplet is set True.
        prolong: bool
                 If set False: The moment we exhaust of all the triplets the tightening stops.
                 If set True: The tightening will be performed max_iterations number of times irrespective of the triplets.
        References
        ----------
        Section 3.3: The Dual Algorithm; Tightening LP Relaxation for MAP using Message Passing (2008)
        By Sontag Et al.
        Examples
        --------
        >>> from pgmpy.models import MarkovNetwork
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.inference import Mplp
        >>> import numpy as np
        >>> student = MarkovNetwork()
        >>> student.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('E', 'F')])
        >>> factor_a = DiscreteFactor(['A'], cardinality=[2], values=np.array([0.54577, 1.8323]))
        >>> factor_b = DiscreteFactor(['B'], cardinality=[2], values=np.array([0.93894, 1.065]))
        >>> factor_c = DiscreteFactor(['C'], cardinality=[2], values=np.array([0.89205, 1.121]))
        >>> factor_d = DiscreteFactor(['D'], cardinality=[2], values=np.array([0.56292, 1.7765]))
        >>> factor_e = DiscreteFactor(['E'], cardinality=[2], values=np.array([0.47117, 2.1224]))
        >>> factor_f = DiscreteFactor(['F'], cardinality=[2], values=np.array([1.5093, 0.66257]))
        >>> factor_a_b = DiscreteFactor(['A', 'B'], cardinality=[2, 2],
        ...                             values=np.array([1.3207, 0.75717, 0.75717, 1.3207]))
        >>> factor_b_c = DiscreteFactor(['B', 'C'], cardinality=[2, 2],
        ...                             values=np.array([0.00024189, 4134.2, 4134.2, 0.0002418]))
        >>> factor_c_d = DiscreteFactor(['C', 'D'], cardinality=[2, 2],
        ...                             values=np.array([0.0043227, 231.34, 231.34, 0.0043227]))
        >>> factor_d_e = DiscreteFactor(['E', 'F'], cardinality=[2, 2],
        ...                             values=np.array([31.228, 0.032023, 0.032023, 31.228]))
        >>> student.add_factors(factor_a, factor_b, factor_c, factor_d, factor_e, factor_f,
        ...                     factor_a_b, factor_b_c, factor_c_d, factor_d_e)
        >>> mplp = Mplp(student)
        >>> result = mplp.map_query()
        >>> result
        {'B': 0.93894, 'C': 1.121, 'A': 1.8323, 'F': 1.5093, 'D': 1.7765, 'E': 2.12239}
        """
        print('correct pgmpy')
        self.dual_threshold = dual_threshold
        self.integrality_gap_threshold = integrality_gap_threshold
        # Run MPLP initially for a maximum of init_iter times.
        self._run_mplp(init_iter)
        # If triplets are to be used for the tightening, we proceed as follows
        if tighten_triplet:
            self.triplet_mode = True
            print('tightening triplets')
            self._tighten_triplet(max_iterations, later_iter, max_triplets, prolong, mode)
        return {list(key)[0]: val for key, val in self.best_assignment.items()}
