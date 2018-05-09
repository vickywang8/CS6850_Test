"""
entropy_parameter: corresponds to the alpha parameter, only needs to be updated if we are doing testing that involves alpha
in vote_and_elect, change the True in votes_with_entropy to False if doing tests that involve alpha
in the infection function, change infected_bound and infection_rate accordingly
change num_initial_spreaders
"""

from collections import defaultdict
import random
import math
import numpy as np
from scipy.sparse import csgraph
import pickle

nodes_set = set()
edges_set = set()
edges_directed = set()
node_neighbors_dict = {}

num_initial_spreaders = 100
elected_spreaders = []
entropy_parameter = 100

with open('ca-CondMat.txt') as inputfile:
	for line in inputfile:
		nodes = line.strip().split()
		nodes.sort()
		for node in nodes:
			nodes_set.add(node)
		edges_set.add(tuple(nodes))
num_nodes = len(nodes_set) 	#23133
num_edges = len(edges_set)	#93497

# with open('web-BerkStan.txt') as inputfile:
# 	for i, line in enumerate(inputfile):
# 		if i >= 4:
# 			nodes = line.strip().split()
# 			nodes.sort()
# 			for node in nodes:
# 				nodes_set.add(node)
# 			edges_set.add(tuple(nodes))
			# if tuple(nodes) in edges_directed:
			# 	edges_set.add(tuple(nodes))
			# 	# edges_directed.remove(tuple(nodes))
			# else:
			# 	edges_directed.add(tuple(nodes))

num_nodes = len(nodes_set) 	#Condmat: 23133  BerkStan: 685230
nodes = list(nodes_set)
print(num_nodes)
num_directed_edges = len(edges_set)
print(num_directed_edges)
edges = list(edges_set)
# num_directed_edges = len(edges_directed)	#Condmat 93497	 BerkStan (directed): 6649470 BerkStan (undirected): 951125
# print(num_directed_edges)

neighbors_dict = defaultdict(set)
for (node_0, node_1) in edges_set:
	neighbors_dict[node_0].add(node_1)
	neighbors_dict[node_1].add(node_0)
# graph = np.zeros((num_nodes, num_nodes))
# average degree of network
node_index_dict = {}
for i in range(num_nodes):
	node_index_dict[nodes[i]] = i
average_degree = 0
for node, neighbors in neighbors_dict.items():
	# for neighbor in neighbors:
		# graph[node_index_dict[node]][node_index_dict[neighbor]] = 1
		# graph[node_index_dict[neighbor]][node_index_dict[node]] = 1
	average_degree += len(neighbors)
average_degree /= num_nodes
dist_matrix = np.load("dist_matrix.npy")
# print("average degree of the graph is :" + str(average_degree))
# dist_matrix = csgraph.shortest_path(graph)
# print("done with dist_matrix")
# np.save("dist_matrix", dist_matrix)

# dist_dict = {}
# for i in range(num_nodes):
# 	node_i = nodes[i]
# 	dist_dict[node_i] = {}
# 	for j in range(num_nodes):
# 		node_j = nodes[j]
# 		if dist_matrix[i][j] != math.inf:
# 			dist_dict[node_i][node_j] = dist_matrix[i][j]

# with open("dist_dict.pickle", "wb") as handle:
#     pickle.dump(dist_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_avg_dist_from_selected(elected_spreaders, dist_matrix, node):
	sum_dist = 0
	for elected in elected_spreaders:
		dist = dist_matrix[node_index_dict[elected]][node_index_dict[node]]
		sum_dist += 2*average_degree if dist == math.inf else dist
	print(sum_dist/len(elected_spreaders))
	return sum_dist/len(elected_spreaders)

def get_pj(node, target_node, neighbors_dict):
	return len(neighbors_dict[node])/sum([len(neighbors_dict[n]) for n in neighbors_dict[target_node]])

def get_entropy(target_node, neighbors_dict):
	numerator = sum([-get_pj(neighbor, target_node, neighbors_dict)*math.log(get_pj(neighbor, target_node, neighbors_dict)) for neighbor in neighbors_dict[target_node]])
	# print(len(neighbors_dict[target_node]))
	if len(neighbors_dict[target_node]) == 0:
		return 0
	denominator = math.log(len(neighbors_dict[target_node]))
	if denominator == 0:
		return 0
	#print(target_node)
	# print(numerator/denominator)
	return numerator/denominator

def votes_with_entropy(num_votes_received, node, neighbors_dict, elected_spreaders, scale = 0):
	if scale == 0:
		return num_votes_received*get_entropy(node, neighbors_dict)
	elif scale == 1:
		return num_votes_received+entropy_parameter*get_entropy(node, neighbors_dict)*num_votes_received
	elif scale == 2:
		return num_votes_received**(2*get_entropy(node, neighbors_dict))
	elif scale == 3:
		if len(elected_spreaders) == 0:
			return num_votes_received
		return num_votes_received*get_avg_dist_from_selected(elected_spreaders, dist_matrix, node)

def vote_and_elect(node_voting_info, elected_spreaders):
	node_with_max_votes = ""
	max_votes = 0
	for node, voting_info in node_voting_info.items():
		neighbors = neighbors_dict[node]
		num_votes_received = 0
		for neighbor in neighbors:
			num_votes_received += node_voting_info[neighbor][1]
		num_votes_received = votes_with_entropy(num_votes_received, node, neighbors_dict, elected_spreaders, 3)
		if num_votes_received > max_votes:
			max_votes = num_votes_received
			node_with_max_votes = node
		node_voting_info[node] = (num_votes_received, node_voting_info[node][1])
	return node_with_max_votes

def update_voting_ability(elected_node, neighbors_dict, node_voting_info, average_degree):
	for neighbor in neighbors_dict[elected_node]:
		new_voting_ability = node_voting_info[neighbor][1] - 1/average_degree
		if new_voting_ability >= 0:
			node_voting_info[neighbor] = (node_voting_info[neighbor][0], new_voting_ability)
		else:
			node_voting_info[neighbor] = (node_voting_info[neighbor][0], 0)

# initialize the algorithm
node_voting_info = {}
for node in nodes_set:
	node_voting_info[node] = (0, 1) # (number of votes received, voting power)
while (len(elected_spreaders) < num_initial_spreaders):
	elected_node = vote_and_elect(node_voting_info, elected_spreaders)
	# print(elected_node)
	if elected_node not in elected_spreaders:
		elected_spreaders.append(elected_node)
	node_voting_info[elected_node] = (0, 0)
	update_voting_ability(elected_node, neighbors_dict, node_voting_info, average_degree)
print(elected_spreaders)

### JUST INFORMATION ENTROPY (NOT VOTERANK CODE)
# nodes_votes = []
# for node in nodes_set:
# 	nodes_votes.append((get_entropy(node, neighbors_dict), node))
# nodes_votes.sort()
# nodes_votes = nodes_votes[-num_initial_spreaders:]
# print(nodes_votes)
# elected_spreaders = [node for entropy,node in nodes_votes]
# print(elected_spreaders)

condmat_elected_spreaders_without_alpha_200 = ['73647', '52658', '78667', '97632', '101425', '97788', '95372', '22987', '22757', '91392', '83259', '46269', '101355', '15439', '101191', '11063', '84209', '26075', '8536', '29380', '46016', '9991', '31762', '26750', '56672', '55210', '905', '95940', '107009', '61271', '73122', '102365', '57070', '71461', '33410', '485', '1895', '7399', '2716', '15345', '12915', '14096', '88363', '106876', '35010', '37206', '35171', '83197', '96395', '57340', '72331', '18654', '53994', '50541', '80915', '17933', '96866', '15113', '52364', '2962', '85266', '62113', '27892', '92360', '101743', '41266', '99977', '62327', '30365', '90690', '9533', '83824', '97009', '26130', '36740', '73252', '83069', '64278', '28953', '36382', '58706', '45942', '48139', '103420', '60057', '60251', '69685', '34845', '91541', '31208', '7204', '72730', '99870', '51336', '6185', '60662', '23411', '74055', '23127', '57478', '38468', '71222', '94304', '21672', '1764', '23548', '88748', '53906', '83037', '53880', '74869', '43077', '93813', '8350', '35688', '97347', '20562', '48940', '66460', '28121', '34770', '21181', '77524', '56414', '47468', '79087', '58293', '57036', '55406', '74250', '48875', '53624', '24840', '36435', '83876', '80859', '79387', '2451', '8100', '46144', '32554', '86764', '72044', '101472', '88071', '49235', '87299', '60432', '86808', '46805', '52472', '59595', '37250', '61105', '49031', '85840', '98676', '9489', '11174', '100439', '10099', '34346', '8810', '52287', '32332', '96245', '41240', '45769', '100587', '22461', '14023', '45251', '40747', '16963', '52098', '24254', '5215', '28575', '72079', '81509', '79496', '44960', '90477', '34703', '62943', '55966', '30488', '42722', '45051', '65099', '42478', '66908', '83984', '32903', '20179', '42245', '93764', '24047', '48626', '23983']
# Average # of time steps until convergence: 13.77
# Average # of time steps until convergence: 19.75(condmat_elected_spreaders_without_alpha_200[:20])
# Average # of time steps until convergence: 15.51(condmat_elected_spreaders_without_alpha_200[:100])
# Average % of infected nodes: 0.4060472052911425 (condmat_elected_spreaders_without_alpha_200[:100])
# Average % of infected nodes: 0.4984390264989409 (condmat_elected_spreaders_without_alpha_200)
condmat_elected_spreaders_with_alpha_1_200 = ['73647', '52658', '78667', '97632', '101425', '97788', '95372', '22987', '22757', '91392', '83259', '46269', '101355', '15439', '101191', '26075', '84209', '11063', '46016', '8536', '29380', '9991', '31762', '56672', '26750', '55210', '95940', '905', '107009', '61271', '73122', '57070', '102365', '71461', '12915', '485', '33410', '2716', '1895', '7399', '15345', '83197', '14096', '88363', '35171', '35010', '106876', '37206', '96395', '72331', '57340', '18654', '53994', '17933', '80915', '50541', '96866', '15113', '52364', '85266', '27892', '2962', '90690', '62113', '101743', '92360', '41266', '99977', '83824', '36740', '62327', '30365', '9533', '64278', '26130', '97009', '83069', '73252', '38468', '28953', '91541', '6185', '103420', '34845', '36382', '23127', '60251', '48139', '58706', '69685', '60662', '74055', '7204', '51336', '31208', '45942', '23548', '72730', '23411', '60057', '99870', '53906', '88748', '1764', '35688', '57478', '71222', '53880', '94304', '21672', '74869', '83037', '8350', '43077', '97347', '48940', '93813', '21181', '34770', '79087', '20562', '57036', '56414', '66460', '28121', '2451', '77524', '24840', '47468', '48875', '58293', '55406', '74250', '46144', '101472', '80859', '86764', '83876', '53624', '79387', '36435', '88071', '8100', '32554', '52287', '72044', '87299', '5215', '85840', '49235', '9489', '60432', '37250', '49031', '33099', '86808', '11174', '59595', '98676', '61105', '28575', '52472', '46805', '8810', '32332', '41240', '100439', '34346', '45769', '14023', '89340', '10099', '96245', '22461', '52098', '24254', '40747', '25735', '30488', '16963', '100587', '81509', '45251', '55966', '34703', '72079', '42478', '62943', '44960', '79496', '90477', '66908', '85796', '93764', '83984', '46066', '48626', '32903', '20179', '42245']
condmat_elected_spreaders_with_alpha_5_200 = ['73647', '52658', '78667', '97632', '101425', '97788', '95372', '22987', '22757', '91392', '83259', '46269', '101355', '15439', '101191', '26075', '84209', '11063', '8536', '29380', '46016', '9991', '31762', '56672', '26750', '55210', '905', '95940', '107009', '61271', '73122', '102365', '57070', '71461', '33410', '485', '2716', '7399', '1895', '15345', '12915', '14096', '83197', '88363', '35010', '35171', '37206', '106876', '96395', '72331', '57340', '18654', '53994', '50541', '80915', '17933', '96866', '15113', '52364', '85266', '2962', '27892', '62113', '92360', '101743', '90690', '41266', '99977', '62327', '30365', '9533', '83824', '26130', '97009', '36740', '64278', '73252', '83069', '28953', '45942', '58706', '36382', '103420', '6185', '48139', '34845', '60251', '69685', '38468', '91541', '7204', '31208', '60057', '51336', '60662', '99870', '74055', '72730', '23127', '23411', '57478', '71222', '94304', '1764', '21672', '88748', '23548', '53906', '53880', '83037', '74869', '35688', '93813', '43077', '8350', '48940', '97347', '66460', '20562', '47468', '21181', '34770', '28121', '79087', '56414', '77524', '58293', '57036', '55406', '74250', '48875', '24840', '53624', '36435', '80859', '83876', '79387', '46144', '2451', '8100', '86764', '32554', '101472', '72044', '88071', '87299', '49235', '60432', '52287', '86808', '37250', '9489', '59595', '49031', '61105', '85840', '98676', '52472', '11174', '100439', '10099', '34346', '8810', '32332', '46805', '5215', '41240', '14023', '96245', '45769', '28575', '22461', '100587', '52098', '40747', '16963', '24254', '45251', '72079', '81509', '44960', '79496', '30488', '34703', '89340', '62943', '90477', '55966', '42478', '45051', '66908', '42722', '65099', '32903', '83984', '42245', '93764', '20179', '48626', '85796']
condmat_elected_spreaders_with_alpha_100_200 = ['73647', '52658', '78667', '97632', '101425', '97788', '95372', '22987', '22757', '91392', '83259', '46269', '101355', '15439', '101191', '11063', '84209', '26075', '8536', '29380', '46016', '9991', '31762', '26750', '56672', '55210', '905', '95940', '107009', '61271', '73122', '102365', '57070', '71461', '33410', '485', '1895', '7399', '2716', '15345', '12915', '14096', '88363', '106876', '35010', '37206', '35171', '83197', '96395', '57340', '72331', '18654', '53994', '50541', '80915', '17933', '96866', '15113', '52364', '2962', '85266', '62113', '27892', '92360', '101743', '41266', '99977', '62327', '30365', '90690', '9533', '83824', '97009', '26130', '36740', '73252', '83069', '64278', '28953', '36382', '58706', '45942', '48139', '103420', '60057', '60251', '69685', '34845', '91541', '31208', '7204', '72730', '99870', '51336', '6185', '60662', '23411', '74055', '23127', '57478', '38468', '71222', '94304', '21672', '1764', '23548', '88748', '53906', '83037', '53880', '74869', '43077', '93813', '8350', '35688', '97347', '48940', '20562', '66460', '28121', '34770', '21181', '77524', '47468', '56414', '79087', '58293', '57036', '55406', '74250', '48875', '53624', '24840', '36435', '83876', '80859', '79387', '2451', '8100', '46144', '32554', '86764', '72044', '101472', '88071', '49235', '87299', '60432', '86808', '46805', '52472', '59595', '37250', '61105', '49031', '85840', '98676', '9489', '11174', '100439', '10099', '34346', '8810', '52287', '32332', '41240', '96245', '45769', '22461', '100587', '14023', '45251', '40747', '16963', '52098', '24254', '5215', '28575', '72079', '81509', '79496', '44960', '90477', '34703', '62943', '55966', '30488', '42722', '45051', '65099', '42478', '66908', '83984', '32903', '20179', '42245', '93764', '24047', '48626', '23983']
#Average # of time steps until convergence: 13.73 (100 iterations)
condmat_elected_spreaders_with_exp_200 = ['73647', '52658', '78667', '22987', '97632', '101425', '97788', '95372', '22757', '15439', '11063', '83259', '8536', '29380', '46269', '101355', '91392', '101191', '9991', '7399', '84209', '61271', '95940', '905', '26075', '55210', '1895', '56672', '14096', '33410', '46016', '35010', '107009', '31762', '71461', '73122', '2716', '88363', '57070', '102365', '37206', '485', '106876', '18654', '57340', '26750', '12915', '53994', '62113', '50541', '28953', '92360', '83197', '96395', '35171', '15345', '97009', '101743', '48139', '30365', '80915', '62327', '31208', '83824', '99977', '72331', '36382', '69685', '99870', '60057', '58706', '96866', '2962', '73252', '43077', '57478', '88071', '26130', '85266', '7204', '41266', '20562', '83069', '51336', '93813', '45942', '103420', '34845', '60251', '72730', '94304', '33099', '52364', '91541', '17933', '58293', '36740', '71222', '15113', '77524', '66460', '28121', '64278', '83037', '53624', '21672', '27892', '9533', '97347', '36435', '1764', '56414', '60662', '23411', '23127', '6185', '86808', '55406', '74055', '79387', '23548', '38204', '8350', '21181', '74250', '34770', '49932', '90690', '16963', '100587', '48940', '45251', '59595', '80859', '32554', '53906', '42722', '60432', '83876', '24840', '2451', '48875', '79496', '88748', '53880', '46805', '49235', '98676', '72079', '57036', '61105', '52472', '79087', '25735', '38468', '40747', '11174', '32332', '68262', '90477', '37250', '24047', '35688', '10099', '34346', '8100', '49031', '96245', '72044', '85840', '19606', '24254', '62943', '44960', '29163', '74869', '81509', '101472', '87299', '32903', '34703', '8810', '46144', '45051', '75507', '23983', '6712', '58236', '22461', '100439', '66908', '9489', '52098', '42245', '65099', '55966', '78851', '69843', '54448', '53856']
# Average # of time steps until convergence: 13.9 (100 iterations)
condmat_elected_spreaders_with_2_mul_exp_200 = ['73647', '52658', '78667', '22987', '97632', '101425', '97788', '95372', '22757', '15439', '11063', '83259', '8536', '29380', '46269', '101355', '91392', '101191', '9991', '7399', '84209', '61271', '95940', '905', '26075', '55210', '1895', '56672', '14096', '33410', '46016', '35010', '107009', '31762', '71461', '73122', '2716', '88363', '57070', '102365', '37206', '485', '106876', '18654', '57340', '26750', '12915', '53994', '62113', '50541', '28953', '92360', '83197', '96395', '35171', '15345', '97009', '101743', '48139', '30365', '80915', '62327', '31208', '83824', '99977', '72331', '36382', '69685', '99870', '60057', '58706', '96866', '2962', '73252', '43077', '57478', '88071', '26130', '85266', '7204', '41266', '20562', '83069', '51336', '93813', '45942', '103420', '34845', '60251', '72730', '94304', '33099', '52364', '91541', '17933', '58293', '36740', '71222', '15113', '77524', '66460', '28121', '64278', '83037', '53624', '21672', '27892', '9533', '97347', '36435', '1764', '56414', '60662', '23411', '23127', '6185', '86808', '55406', '74055', '79387', '23548', '38204', '8350', '21181', '74250', '34770', '49932', '90690', '16963', '100587', '48940', '45251', '59595', '80859', '32554', '53906', '42722', '60432', '83876', '24840', '2451', '48875', '79496', '88748', '53880', '46805', '49235', '98676', '72079', '57036', '61105', '52472', '79087', '25735', '38468', '40747', '11174', '32332', '68262', '90477', '37250', '24047', '35688', '10099', '34346', '8100', '49031', '96245', '72044', '85840', '19606', '24254', '62943', '44960', '29163', '74869', '81509', '101472', '87299', '32903', '34703', '8810', '46144', '45051', '75507', '23983', '6712', '58236', '22461', '100439', '66908', '9489', '52098', '42245', '65099', '55966', '78851', '69843', '54448', '53856']
condmat_elected_spreaders_with_dist_20 = ['73647', '46269', '52658', '78667', '40271', '97632', '58515', '101425', '95372', '54054', '97788', '73122', '91392', '27892', '22757', '28953', '101355', '26075', '29380', '22987']
# Average # of time steps until convergence: 19.82
condmat_elected_spreaders_with_dist_100_multiply = ['73647', '30874', '52658', '78667', '97632', '14096', '97788', '37206', '22987', '15113', '101425', '83259', '96395', '22757', '91392', '72331', '17933', '101355', '95372', '93764', '46269', '84209', '33099', '26075', '101191', '9991', '102365', '79145', '46016', '66908', '29380', '15439', '11063', '8536', '56672', '7204', '26750', '95940', '48875', '31762', '4750', '12915', '24840', '55210', '73122', '11174', '107009', '33410', '485', '905', '98676', '9533', '50586', '71461', '80859', '7399', '57070', '45051', '1895', '10353', '61271', '53806', '15345', '6121', '71548', '2716', '90690', '66746', '2962', '79257', '57340', '106876', '52364', '54054', '101743', '88574', '88363', '27892', '3554', '35171', '62327', '92360', '15164', '50541', '66800', '36740', '85266', '56095', '41266', '97009', '36382', '62113', '103225', '37675', '64278', '59202', '80915', '34770', '79387', '58470']
# Average # of time steps until convergence: 15.63
# >>> len(set(condmat_elected_spreaders_without_alpha_200[:100])condmat_elected_spreaders_without_alpha_200[:100])).intersection(set(condmat_elected_spreaders_with_dist_100)))
# 69
# Average % of infected nodes: 0.40498378939177776
###################################

# def infection(neighbors_dict, elected_spreaders, infected_bound = num_nodes*0.5, infection_rate = 0.8):
# 	infected_set = set(elected_spreaders)
# 	newly_infected_set = set()
# 	t = 0
# 	while len(infected_set) < infected_bound:
# 		for infected_node in infected_set:
# 			neighbor = random.choice(list(neighbors_dict[infected_node]))
# 			if random.uniform(0,1) <= infection_rate:
# 				newly_infected_set.add(neighbor)
# 		t+=1
# 		infected_set = infected_set.union(newly_infected_set)
# 	return t
def infection(neighbors_dict, elected_spreaders, fixed_time_steps = 20, infection_rate = 0.5):
	infected_set = set(elected_spreaders)
	newly_infected_set = set()
	t = 0
	while t < fixed_time_steps:
		for infected_node in infected_set:
			neighbor = random.choice(list(neighbors_dict[infected_node]))
			if random.uniform(0,1) <= infection_rate:
				newly_infected_set.add(neighbor)
		t += 1
		infected_set = infected_set.union(newly_infected_set)
	return len(infected_set) / num_nodes

# run infection
# avg = 0
# for i in range(0,100):
# 	avg+=infection(neighbors_dict, condmat_elected_spreaders_with_dist_100)
# print("Average # of time steps until convergence: " + str(avg/100))

# run infection
# avg = 0
# for i in range(0,100):
# 	avg+=infection(neighbors_dict, condmat_elected_spreaders_with_dist_100_multiply_add)
# print("Average % of infected nodes: " + str(avg/100))













