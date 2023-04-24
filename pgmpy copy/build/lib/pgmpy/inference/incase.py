import copy
import itertools as it

import numpy as np
import networkx as nx

from pgmpy.inference import Inference
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor


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
        
        # this line takes quadratic time 
        # for edge_pair in it.combinations(model.edges(), 2): 
        #     #print('edge pairs exist')
        #     #print('frozenset(edge_pair[0]): ', frozenset(edge_pair[0]))
        #     flag =  bool(frozenset(edge_pair[0]) & frozenset(edge_pair[1]) & self.intersection_set_variables != set())
        #     print('variable added: ', flag)
        #     self.intersection_set_variables.add(
        #         frozenset(edge_pair[0]) & frozenset(edge_pair[1])
        #     )
            

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

        self.active_messages = {}
        for edge in model.edges():
            self.active_messages[frozenset(edge)] = False

        #possibly the same as cluster_potential if cluster_potential never changes (TODO)
        self.factor_dict = {}
        for factor in model.get_factors():
            self.factor_dict[frozenset(factor.scope())] = factor

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

            #print('self.intersection_sets_for_cluster_c: ', self.intersection_sets_for_cluster_c)
            # self.intersection_sets_for_cluster_c = [
            #     intersect.intersection(self.cluster_variables)
            #     for intersect in intersection_set_variables
            #     if intersect.intersection(self.cluster_variables)
            # ]

            # Initialize messages from this cluster to its respective intersection sets
            if len(self.cluster_variables) == 2:
                self.message_from_cluster = {}
                for intersection in self.intersection_sets_for_cluster_c:
                    arg = copy.deepcopy(cluster_potential)
                    arg = arg.maximize(list(self.cluster_variables - intersection), inplace=False)
                    self.message_from_cluster[intersection] = 1/2*arg
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
                    #self.message_from_cluster[intersection] = DiscreteFactor(
                    #    present_variables,
                    #    present_variables_card,
                    #    np.zeros(np.prod(present_variables_card)),
                    #)
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
            #print('current_intersect: ', current_intersect)
            #print('sum_messages: ', self.sum_messages[current_intersect])
            #print('message_from_cluster: ', sending_cluster.message_from_cluster[current_intersect])
            #print('theta: ', theta[current_intersect])
            #print('message_minus: ', message_minus[current_intersect])
            #print('objective_cluster: ', objective_cluster)
        
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
            #print('message_old: ', message_old)
            #print('message_new: ', message_new)
            #print('sum_messages: ', self.sum_messages[current_intersect])
            #print('self.objective_new[current_intersect]: ', self.objective_new[current_intersect])

        return 0

    def _update_message_edge_edge(self, sending_cluster):
        #edge to edge and edge to node messages sent simultaneously
        #first compute edge to node update
        #we assemble theta_i, theta_j, theta_ij, lambda_{c -> ij}, lambda_i^{-j}(x_i)
        #edge to node updates
        old_dual_lp = sum(
            [np.amax(self.objective_new[obj].values) for obj in self.objective_new]
        )
        objective_old = copy.deepcopy(self.objective_new)


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
            #print('current_intersect: ', current_intersect)
            #print('sum_messages: ', self.sum_messages[current_intersect])
            #print('message_from_cluster: ', sending_cluster.message_from_cluster[current_intersect])
            #print('theta: ', theta[current_intersect])
            #print('message_minus: ', message_minus[current_intersect])
            #print('objective_cluster: ', objective_cluster)

        objective_edge_edge = objective_cluster
        #add in triplet to edge messages
        objective_edge_node = objective_cluster + self.sum_messages[edge]
        
        updated_results = []
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            phi_edge_node = objective_edge_node.maximize(
                list(sending_cluster.cluster_variables - current_intersect),
                inplace=False,
            )
            phi_edge_node *= 1 / 3
            # Step. 4) Subtract \delta_i^{-f}
            updated_results.append(
                phi_edge_node + -1 * objective_node[current_intersect]
            )


        index = -1
        #copy of the factor  
        cluster_potential = copy.deepcopy(sending_cluster.cluster_potential)
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            #print('current intersect suspicious: ', current_intersect)
            index += 1
            message_old = copy.deepcopy(sending_cluster.message_from_cluster[current_intersect])
            message_new = updated_results[index]
            sending_cluster.message_from_cluster[current_intersect] = message_new
            self.sum_messages[current_intersect] += -1*message_old + message_new
            self.objective_new[current_intersect] = theta[current_intersect] + self.sum_messages[current_intersect]
            #print('message_old: ', message_old)
            #print('current_intersect: ', current_intersect)
            #print('sending_cluster.message_from_cluster[current_intersect]: ', message_new)
            #print('sum_messages: ', self.sum_messages[current_intersect])
            #print('self.objective_new[current_intersect]: ', self.objective_new[current_intersect])

        #edge to edge update 
        message_old = copy.deepcopy(sending_cluster.message_from_cluster[edge])
        #print('(1/3)*objective_edge_edge: ', (1/3)*objective_edge_edge)
        #print('(2/3)*self.sum_messages[edge]: ', (2/3)*self.sum_messages[edge])
        message_new = (1/3)*objective_edge_edge + (-2/3)*self.sum_messages[edge]
        #print('message_new: ', message_new)

        sending_cluster.message_from_cluster[edge] = message_new 
        #self.sum_messages[edge] is sum of triplet to edge messages
        #is not updated during edge to edge
        #self.sum_messages[edge] += -1*message_old + message_new
        #print('self.sum_messages[edge]: ', self.sum_messages[edge])
        self.objective_new[edge] = message_new + self.sum_messages[edge]
        #print('self.objective_new[edge]: ', self.objective_new[edge])
        #self.objective_new[edge] += -1*message_old + message_new 
        #print('sending_cluster.message_from_cluster[edge]: ', sending_cluster.message_from_cluster[edge])
        #new_dual_lp = sum(
        #    [np.amax(self.objective_new[obj].values) for obj in self.objective_new]
        #)
        #print('edge-edge new_dual_lp: ', new_dual_lp)
        new_dual_lp = sum(
            [np.amax(self.objective_new[obj].values) for obj in self.objective_new]
        )
        if new_dual_lp > old_dual_lp +0.001:
            print('edge: ', edge)
            print('objective_edge_edge: ', objective_edge_edge)
            print('objective_edge_node: ', objective_edge_node)
            print('sending_cluster.message_from_cluster[edge]: ', sending_cluster.message_from_cluster[edge])
            for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
                print('sending_cluster.message_from_cluster[current_intersect]: ', 
                    sending_cluster.message_from_cluster[current_intersect])
                print('objective_new[current_intersect]: ', self.objective_new[current_intersect])
                print('objective_old[current_intersect]: ', objective_old[current_intersect])
            print('objective_new[edge]: ', self.objective_new[edge])
            print('objective_old[edge]: ', objective_old[edge])
            print('old_dual_lp: ', old_dual_lp)
            print('new_dual_lp: ', new_dual_lp)
            raise ValueError("DUAL NO DECREASE")

        return 0

    def _update_message_triplet_edge(self, sending_cluster):
        old_dual_lp = sum(
            [np.amax(self.objective_new[obj].values) for obj in self.objective_new]
        )

        triplet = frozenset(sending_cluster.cluster_variables)
        cluster_potential = sending_cluster.cluster_potential
        cardinalities = cluster_potential.cardinality
        #defined to be lambda_e'->e'(x_e') + \sum_{c' \neq c and e' \in c'} lambda_{c'->e'}(x_e')
        objective_cluster = DiscreteFactor(triplet,cardinalities,np.zeros(np.prod(cardinalities)))  
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            edge_edge = self.cluster_set[current_intersect].message_from_cluster[current_intersect] 
            #print('edge_edge: ', edge_edge)
            #print('self.objective_new[current_intersect]: ', self.objective_new[current_intersect])
            #print('current_intersect: ', current_intersect)
            #print('self.objective_new[current_intersect]: ', self.objective_new[current_intersect])
            #print('sending_cluster.message_from_cluster[current_intersect]: ', sending_cluster.message_from_cluster[current_intersect])
            update = self.objective_new[current_intersect] + -1*sending_cluster.message_from_cluster[current_intersect]
            #print('update: ', update)
            #print('before: ', objective_cluster)
            objective_cluster += self.objective_new[current_intersect] + \
                                -1*sending_cluster.message_from_cluster[current_intersect]
            #print('objective_cluster: ', objective_cluster)
        
        #print('objective_cluster: ', objective_cluster)
        updated_results = []
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            #print('current_intersect: ', current_intersect)
            phi_triplet_edge = objective_cluster.maximize(
                list(sending_cluster.cluster_variables - current_intersect),
                inplace=False,
            )
            #print('phi_triplet_edge before: ', phi_triplet_edge)
            phi_triplet_edge *= 1/3
            
            #print('phi_triplet_edge: ', phi_triplet_edge)
            # Step. 4) Subtract \delta_i^{-f}
            updated_results.append(
                phi_triplet_edge + -1 * (self.objective_new[current_intersect] +\
                    -1*sending_cluster.message_from_cluster[current_intersect])
            )

        objective_copy = copy.deepcopy(self.objective_new)

        index = -1
        #copy of the factor  
        cluster_potential = copy.deepcopy(sending_cluster.cluster_potential)
        for current_intersect in sending_cluster.intersection_sets_for_cluster_c:
            index += 1
            message_old = copy.deepcopy(sending_cluster.message_from_cluster[current_intersect])
            message_new = updated_results[index]
            #print('message_old: ', message_old)
            #print('message_new: ', message_new)
            sending_cluster.message_from_cluster[current_intersect] = message_new  
            self.sum_messages[current_intersect] += -1*message_old + message_new
            self.objective_new[current_intersect] += -1*message_old + message_new


        for obj in self.objective_new:
            #print('obj: ', obj)
            #print('old self.objective_new[obj].values: ', objective_copy[obj].values)
            #print('new self.objective_new[obj].values: ', self.objective_new[obj].values)
            old_max = np.amax(objective_copy[obj].values)
            new_max = np.amax(self.objective_new[obj].values)
            #print('old_max: ', old_max)
            #print('new_max: ', new_max)
        #new_dual_lp = sum(
        #    [np.amax(self.objective_new[obj].values) for obj in self.objective_new]
        #)
        #print('triplet-edge new_dual_lp: ', new_dual_lp)
        new_dual_lp = sum(
            [np.amax(self.objective_new[obj].values) for obj in self.objective_new]
        )
        if new_dual_lp > old_dual_lp+ 0.001:
            print('old_dual_lp: ', old_dual_lp)
            print('new_dual_lp: ', new_dual_lp)
            raise ValueError("DUAL NO DECREASE")
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
        #for node in self.objective_new:
        #    if len(node) == 1:
        #        print('node: ', node)
        #        print('self.objective_new[node]:', self.objective_new[node])

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
        # for obj in self.objective_new:
        #     print('obj: ', obj)
        #     print('self.objective_new[obj]: ', self.objective_new[obj])
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

    def find_triangles(self):
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
        #Simply a bug, maximal cliques do not necessarily include triplets 
        #print('nx.find_cliques(self.model): ', nx.find_cliques(self.model))
        #print('list(filter(lambda x: len(x) == 3, nx.find_cliques(self.model))): ', 
        #    list(filter(lambda x: len(x) == 3, nx.find_cliques(self.model))))
        #TODO: this is extremely inefficient this is a hack 
        return list(filter(lambda x: len(x) == 3, nx.enumerate_all_cliques(self.model)))

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
        #print('update triangles')
        new_intersection_set = []
        for triangle_vars in triangles_list:
            #print('triangle vars: ', triangle_vars)
            cardinalities = [self.cardinality[variable] for variable in triangle_vars]
            current_intersection_set = [
                frozenset(intersect) for intersect in it.combinations(triangle_vars, 2)
            ]
            current_factor = DiscreteFactor(
                triangle_vars, cardinalities, np.zeros(np.prod(cardinalities))
            )
            self.cluster_set[frozenset(triangle_vars)] = self.Cluster(
                current_intersection_set, current_factor
            )

            triplet = self.cluster_set[frozenset(triangle_vars)]
            # add new factors
            self.model.factors.append(current_factor)
            # add new intersection sets
            #I think this line does nothing TODO
            #new_intersection_set.extend(current_intersection_set)
            # add new factors in objective
            # new objective drops this line
            #self.objective[frozenset(triangle_vars)] = current_factor

            #add edge to edge messages
            for intersect in it.combinations(triangle_vars, 2):
                if not self.active_messages[frozenset(intersect)]:
                    sending_cluster = self.cluster_set[frozenset(intersect)]
                    card = [self.cardinality[variable] for variable in intersect] 
                    empty_edge_edge = DiscreteFactor(list(intersect), card, np.zeros(np.prod(card)))
                    sending_cluster.message_from_cluster[frozenset(intersect)] = empty_edge_edge  
                    self.active_messages[frozenset(intersect)] = True 


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

            # Find the intersection sets of the current triplet
            triplet_intersections = [
                intersect for intersect in it.combinations(triplet, 2)
            ]
            #print('triplet_intersections: ', triplet_intersections)

            #Independent maximization
            ind_max = sum(
                [
                    np.amax(self.objective_new[frozenset(intersect)].values)
                    for intersect in triplet_intersections
                ]
            )
            #print('ind_max: ', ind_max)
            #print('self.objective: ', self.objective)
            # for intersect in triplet_intersections:
            #     print('frozenset(intersect): ', frozenset(intersect))
            #     print('self.objective[frozenset(intersect)].values: ', self.objective[frozenset(intersect)].values)

            # Joint maximization
            joint_max = self.objective_new[frozenset(triplet_intersections[0])]
            for intersect in triplet_intersections[1:]:
                joint_max += self.objective_new[frozenset(intersect)]
            joint_max = np.amax(joint_max.values)
            # score = Independent maximization solution - Joint maximization solution
            score = ind_max - joint_max
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
                #new_dual_lp = sum(
                #    [np.amax(self.objective_new[obj].values) for obj in self.objective_new]
                #)
                #print('check dual_lp: ', new_dual_lp)
                scope = factor.scope()
                if len(scope) == 2:
                    #print('Next Factor')
                    edge = frozenset(factor.scope())
                    #commented just for debug
                    sending_cluster = self.cluster_set[edge]
                    #print('edge: ', edge)
                    #print('self.active_messages: ', self.active_messages)
                    if self.active_messages[edge]:
                        self._update_message_edge_edge(sending_cluster)
                    else :
                        self._update_message(sending_cluster)
                if len(scope) == 3:
                    triplet = frozenset(factor.scope())
                    sending_cluster = self.cluster_set[triplet]
                    #print('triplet: ', triplet)
                    #print('sending_cluster.cluster_potential: ', sending_cluster.cluster_potential)
                    #for intersect in sending_cluster.intersection_sets_for_cluster_c:
                        #print('intersect: ', intersect)
                        #print('objective_new[intersect]: ', self.objective_new[intersect])
                        #print('self.sum_messages[intersect]: ', self.sum_messages[intersect])
                    self._update_message_triplet_edge(sending_cluster)
            # Find an integral solution by locally maximizing the single node beliefs
            self._local_decode()
            # If mplp converges to a global/local optima, we break.
            if (
                self._is_converged(self.dual_threshold, self.integrality_gap_threshold)
                and niter >= 16
            ):
                break

    #helper function for changing mode from max product to add triplet mode.  
    def triplet_setup(self):
        #sum messages initialize to zero for each edge
        for factor in self.model.get_factors():  
            scope = factor.scope()
            if len(scope) == 2:
                card = factor.get_cardinality(list(scope))[scope[0]]
                self.sum_messages[frozenset(scope)] = DiscreteFactor(scope,[card,card],np.zeros(card**2)) 

        #objective initialize to zero for each edge
        for factor in self.model.get_factors():
            scope = factor.scope()
            if len(scope) == 2:
                card = factor.get_cardinality(list(scope))[scope[0]]
                self.objective_new[frozenset(scope)] = DiscreteFactor(scope,[card,card],np.zeros(card**2))

    def _tighten_triplet(self, max_iterations, later_iter, max_triplets, prolong):
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
        """
        # Find all the triplets that are possible in the present model
        self.triplet_setup()
        triangles = self.find_triangles()
        #print('triangles: ', triangles)
        # Evaluate scores for each of the triplets found above
        triplet_scores = self._get_triplet_scores(triangles)
        print('triplet scores: ', triplet_scores)
        # Arrange the keys on the basis of increasing order of the values of the dict. triplet_scores
        sorted_scores = sorted(triplet_scores, key=triplet_scores.get)
        for niter in range(max_iterations):
            #print('niter: ', niter)
            if self._is_converged(
                integrality_gap_threshold=self.integrality_gap_threshold
            ):
                break
            #print('adding triplets')
            # add triplets that are yet not added.
            add_triplets = []
            for triplet_number in range(len(sorted_scores)):
                # At once, we can add at most 5 triplets
                if triplet_number >= max_triplets:
                    print('break triplet number')
                    break
                add_triplets.append(sorted_scores.pop())
            # Break from the tighten triplets loop if there are no triplets to add if the prolong is set to False
            print('add_triplets: ', add_triplets)
            if not add_triplets and prolong is False:
                print('break add triplets and prolong')
                break
            # Update the eligible triplets to tighten the relaxation
            #print('update triangles')
            self._update_triangles(add_triplets)
            # Run MPLP for a maximum of later_iter times.
            self._run_mplp(later_iter)

    def get_integrality_gap(self):
        """
        Returns the integrality gap of the current state of the Mplp algorithm. The lesser it is, the closer we are
                towards the exact solution.
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
        >>> mplp.map_query()
        >>> int_gap = mplp.get_integrality_gap()
        """

        return self.integrality_gap

    def query(self):
        raise NotImplementedError("map_query() is the only query method available.")

    def map_query(
        self,
        init_iter=100,
        later_iter=20,
        dual_threshold=0.0002,
        integrality_gap_threshold=0.0002,
        tighten_triplet=True,
        max_triplets=5,
        max_iterations=100,
        prolong=False,
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
        self.dual_threshold = dual_threshold
        self.integrality_gap_threshold = integrality_gap_threshold
        # Run MPLP initially for a maximum of init_iter times.
        self._run_mplp(init_iter)
        # If triplets are to be used for the tightening, we proceed as follows
        if tighten_triplet:
            print('tightening triplets')
            self._tighten_triplet(max_iterations, later_iter, max_triplets, prolong)
        return {list(key)[0]: val for key, val in self.best_assignment.items()}