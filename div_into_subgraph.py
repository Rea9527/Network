import random

n_graph = 3
graph_dir = 'UndirectedGraphData(964,3000).txt'
with open(graph_dir, 'r') as graph_file:
    content = graph_file.readlines()
    n_node = len(content)
    subgraph_cap = int(n_node / n_graph) + 1
    nodes_list = list(range(1, n_node + 1))
    random.shuffle(nodes_list)
    subgraph_nodes_list = []
    subgraph_edges_list = {}
    for i in range(n_graph):
        subgraph_nodes_list.append(nodes_list[i * subgraph_cap:(i + 1) * subgraph_cap])

    for line in content:
        s, e = line.replace('\n', '').split()
        for i in range(n_graph):
            if int(s) in subgraph_nodes_list[i] and int(e) in subgraph_nodes_list[i]:
                try:
                    subgraph_edges_list[i].append([s, e])
                except:
                    subgraph_edges_list[i] = [[s, e]]
                break
    for i in subgraph_edges_list.keys():
        with open('graph_{}.txt'.format(i), 'w') as subgraph_file:
            for edge in subgraph_edges_list[i]:
                [s, e] = edge
                subgraph_file.write(s+' '+e+'\n')
