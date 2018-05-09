from collections import defaultdict
import random

nodes_set = set()
edges_set = set()
edges_directed = set()
node_neighbors_dict = {}

num_initial_spreaders = 200
elected_spreaders = []
longest_shortest_path = 14

# with open('ca-CondMat.txt') as inputfile:
# 	for i, line in enumerate(inputfile):
# 		nodes = line.strip().split()
# 		nodes.sort()
# 		for node in nodes:
# 			nodes_set.add(node)
# 		edges_set.add(tuple(nodes))

with open('web-BerkStan.txt') as inputfile:
	for i, line in enumerate(inputfile):
		if i >= 4:
			nodes = line.strip().split()
			nodes.sort()
			for node in nodes:
				nodes_set.add(node)
			edges_set.add(tuple(nodes))
			# if tuple(nodes) in edges_directed:
			# 	edges_set.add(tuple(nodes))
			# 	# edges_directed.remove(tuple(nodes))
			# else:
			# 	edges_directed.add(tuple(nodes))

num_nodes = len(nodes_set) 	#Condmat: 23133  BerkStan: 685230
print(num_nodes)
num_directed_edges = len(edges_set)
print(num_directed_edges)
# num_directed_edges = len(edges_directed)	#Condmat 93497	 BerkStan (directed): 6649470 BerkStan (undirected): 951125
# print(num_directed_edges)

neighbors_dict = defaultdict(set)
for (node_0, node_1) in edges_set:
	neighbors_dict[node_0].add(node_1)
	neighbors_dict[node_1].add(node_0)

# average degree of network
average_degree = 0
for node, neighbors in neighbors_dict.items():
	average_degree += len(neighbors)
average_degree /= num_nodes
print("average degree of the graph is :" + str(average_degree))

def vote_and_elect(node_voting_info):
	node_with_max_votes = ""
	max_votes = 0
	for node, voting_info in node_voting_info.items():
		neighbors = neighbors_dict[node]
		num_votes_received = 0
		for neighbor in neighbors:
			num_votes_received += node_voting_info[neighbor][1]
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
# node_voting_info = {}
# for node in nodes_set:
# 	node_voting_info[node] = (0, 1) # (number of votes received, voting power)
# while (len(elected_spreaders) < num_initial_spreaders):
# 	elected_node = vote_and_elect(node_voting_info)
# 	# print(elected_node)
# 	if elected_node not in elected_spreaders:
# 		elected_spreaders.append(elected_node)
# 	node_voting_info[elected_node] = (0, 0)
# 	update_voting_ability(elected_node, neighbors_dict, node_voting_info, average_degree)
# print(elected_spreaders)

cond_mat_elected_spreaders_200 = ['73647', '52658', '78667', '97632', '101425', '97788', '95372', '22987', '22757', '91392', '83259', '46269', '101355', '15439', '101191', '26075', '84209', '46016', '11063', '29380', '8536', '9991', '31762', '56672', '26750', '55210', '95940', '905', '107009', '73122', '61271', '102365', '57070', '71461', '12915', '485', '33410', '2716', '15345', '1895', '83197', '88363', '72331', '14096', '96395', '35010', '37206', '106876', '35171', '7399', '17933', '15113', '96866', '80915', '57340', '52364', '53994', '90690', '18654', '50541', '85266', '27892', '2962', '99977', '41266', '26130', '101743', '36740', '62113', '92360', '64278', '62327', '9533', '30365', '83824', '38468', '97009', '83069', '73252', '91541', '60662', '34845', '74055', '6185', '103420', '23127', '28953', '36382', '60251', '69685', '58706', '48139', '23548', '53906', '7204', '47468', '51336', '88748', '35688', '72730', '31208', '57036', '60057', '53880', '99870', '1764', '21672', '45942', '57478', '34770', '94304', '71222', '8350', '23411', '83037', '79087', '48940', '101472', '97347', '21181', '2451', '24840', '56414', '86764', '46144', '20562', '74869', '30488', '48875', '93813', '28121', '74250', '43077', '55406', '77524', '58293', '66460', '80859', '5215', '83876', '32554', '87299', '52287', '53624', '8100', '36435', '85840', '9489', '28575', '79387', '72044', '49235', '11174', '37250', '49031', '60432', '45769', '46805', '66908', '59595', '86808', '14023', '41240', '98676', '61105', '89340', '88071', '10099', '34346', '8810', '100439', '32332', '52098', '52472', '22461', '96245', '93764', '42478', '85796', '24254', '55966', '1034', '25735', '40747', '81509', '34703', '26002', '16963', '46066', '62943', '98553', '44960', '100587', '72079', '45251', '79496', '83984', '48626', '90477', '45051']
berkstan_mat_elected_spreaders_200 = ['438238', '210376', '272919', '86237', '601656', '210305', '571447', '571448', '623254', '319209', '479054', '417965', '544858', '401873', '477985', '158750', '502214', '388649', '657843', '54008', '462728', '501481', '657219', '388546', '270625', '210142', '631045', '380036', '458002', '48633', '428428', '397429', '109863', '507147', '486303', '663981', '680968', '280874', '257183', '652424', '53386', '106615', '434168', '51837', '678148', '435614', '52855', '652047', '257139', '536676', '513542', '186755', '272509', '538396', '164237', '422953', '442904', '496050', '668472', '511918', '491069', '422961', '665513', '631048', '129159', '331840', '378760', '257305', '375375', '551025', '634081', '65334', '127554', '422971', '400189', '203336', '641479', '129500', '601182', '372775', '210139', '549479', '331645', '158099', '145528', '101735', '207274', '384684', '150862', '641752', '402659', '119422', '383732', '475508', '427185', '428414', '199892', '254913', '355227', '125476', '383095', '542082', '406550', '148061', '153347', '118188', '146231', '151515', '433180', '673942', '409270', '210983', '317952', '637476', '496990', '651634', '142527', '469473', '661771', '169096', '443543', '50420', '204080', '406212', '143567', '128041', '142900', '642948', '383124', '384621', '427786', '427814', '168272', '319210', '494374', '49460', '398418', '123864', '451111', '653080', '406559', '383741', '499304', '640090', '117856', '40', '372332', '48634', '45327', '331807', '331467', '629221', '656104', '331718', '644462', '45362', '259982', '523680', '548797', '451020', '319672', '167593', '462163', '319599', '186968', '672762', '319763', '579915', '319416', '645239', '610092', '61461', '422996', '356987', '550130', '644712', '149417', '49553', '152533', '154704', '354425', '386597', '147253', '608824', '476659', '652687', '542269', '666763', '319735', '644639', '528617', '102397', '514788', '611653', '648914', '651262', '514542', '442633', '670081', '515037']
# Average % of infected nodes: 0.40668525483076123 (cond_mat_elected_spreaders_200[:100])
# Average % of infected nodes: 0.4986551679419015 (cond_mat_elected_spreaders_200)
###################################

# def infection(neighbors_dict, elected_spreaders, infected_bound = num_nodes*0.5, infection_rate = 0.5):
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

# run infection
# avg = 0
# for i in range(0,10):
# 	avg+=infection(neighbors_dict, elected_spreaders)
# print("Average # of time steps until convergence: " + str(avg/10))

def infection(neighbors_dict, elected_spreaders, fixed_time_steps = 35, infection_rate = 0.5):
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
test_spreaders_set = [berkstan_mat_elected_spreaders_200[:1], berkstan_mat_elected_spreaders_200[:10], berkstan_mat_elected_spreaders_200[:20], berkstan_mat_elected_spreaders_200[:40], berkstan_mat_elected_spreaders_200[:60], berkstan_mat_elected_spreaders_200[:80], berkstan_mat_elected_spreaders_200[:100], berkstan_mat_elected_spreaders_200[:150], berkstan_mat_elected_spreaders_200[:200]]
for test in test_spreaders_set:
	avg = 0
	print("number of initial is " + str(len(test)))
	for i in range(0,20):
		avg+=infection(neighbors_dict, test)
	print("Average % of infected nodes: " + str(avg/20))











