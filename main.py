import networkx as nx
import numpy as np
import operator


# 保证边betweenness字典里的边（tuple）都是（小号结点，大号结点）
def asc(edge):
    return (edge[edge[0] > edge[1]], edge[edge[0] < edge[1]])


class Community:

    # 生存随机图（n_node,prob)或者读文件graph_dir
    def __init__(self, n_node=0, prob=0.0, graph_dir=None):
        self.graph = nx.Graph()

        if n_node != 0:
            self.graph = nx.fast_gnp_random_graph(n_node, prob).to_undirected()
        else:
            self.read_graph(graph_dir)

        self.n_node = len(self.graph.nodes())
        self.n_edge = len(self.graph.edges())
        self.betweenness_list = {}

    # 读图文件
    def read_graph(self, graph_dir):
        edges = []
        with open(graph_dir, 'r') as graph_file:
            content = graph_file.readlines()
            for line in content:
                s, e = line.replace('\n', '').split()
                edges.append((int(s), int(e)))
        self.graph.add_edges_from(edges)
        print("G's {} nodes:\n".format(len(self.graph.nodes())), self.graph.nodes())
        print("G's {} edges:\n".format(len(self.graph.edges())), self.graph.edges())

    # 计算sp_betweenness
    def __calcul_betw_sp(self):

        for start in self.graph.nodes():
            betweenness_tmp = {}
            for edge in self.graph.edges():
                betweenness_tmp[asc(edge)] = 0.0
            queue = []
            leaves = []
            dist = np.zeros(self.n_node + 1) - 1
            weight = np.zeros(self.n_node + 1)
            dist[start] = 0
            weight[start] = 1
            queue.append(start)

            while queue != []:
                now = queue.pop(0)
                d = dist[now]
                w = weight[now]
                is_leaf = 1
                neighbors = self.graph.neighbors(now)
                for next_node in neighbors:
                    if dist[next_node] > d or dist[next_node] == -1:
                        is_leaf = 0
                    if dist[next_node] == -1:
                        dist[next_node] = d + 1
                        weight[next_node] = w
                        queue.append(next_node)
                    elif dist[next_node] == d + 1:
                        weight[next_node] += w
                    else:
                        continue
                if is_leaf:
                    leaves.append(now)

            for leaf in leaves:
                neighbors = self.graph.neighbors(leaf)
                for neighbor in neighbors:
                    betweenness_tmp[asc((leaf, neighbor))] += weight[neighbor] * 1.0 / weight[leaf]
                    if neighbor not in queue:
                        queue.append(neighbor)

            while queue != []:
                now = queue.pop(0)
                neighbors = self.graph.neighbors(now)
                d = dist[now]
                w = 1.0
                parents = []
                for neighbor in neighbors:
                    if dist[neighbor] < d:
                        parents.append(neighbor)
                    else:
                        w += betweenness_tmp[asc((now, neighbor))]
                divi = len(parents)
                for next_node in parents:
                    betweenness_tmp[asc((now, next_node))] += w / divi
                    if next_node not in queue:
                        queue.append(next_node)

            for edge in self.graph.edges():
                self.betweenness_list[asc(edge)] += betweenness_tmp[asc(edge)]

    # 计算resistor betweenness
    def __calcul_betw_rs(self):
        Lap = nx.laplacian_matrix(self.graph).toarray()
        for source in self.graph.nodes():
            for sink in self.graph.nodes():
                if sink == source:
                    continue
                reference = 1
                while reference == source or reference == sink:
                    reference += 1
                Lapv = np.delete(Lap, reference - 1, 0)
                Lapv = np.delete(Lapv, reference - 1, 1)
                S = np.zeros((1, self.n_node))
                S[0][source - 1] = 1
                S[0][sink - 1] = -1
                Sv = np.delete(S, reference - 1, 1).T
                Vv = np.matmul(np.linalg.inv(Lapv), Sv)
                Vr = np.array([0])
                V = np.insert(Vv, reference - 1, values=Vr, axis=0)
                for edge in self.graph.edges():
                    self.betweenness_list[asc(edge)] += abs(V[edge[0] - 1][0] - V[edge[1] - 1][0])

    # 计算random_walk betweenness
    def __calcul_betw_rw(self):
        Lap = nx.laplacian_matrix(self.graph).toarray()
        for source in self.graph.nodes():
            for terminate in self.graph.nodes():
                if terminate == source:
                    continue
                Lapv = np.delete(Lap, terminate - 1, 0)
                Lapv = np.delete(Lapv, terminate - 1, 1)
                S = np.zeros((1, self.n_node))
                S[0][source - 1] = 1
                Sv = np.delete(S, terminate - 1, 1).T
                Vv = np.matmul(np.linalg.inv(Lapv), Sv)
                Vr = np.array([0])
                V = np.insert(Vv, terminate - 1, values=Vr, axis=0)
                for edge in self.graph.edges():
                    self.betweenness_list[asc(edge)] += abs(V[edge[0] - 1][0] - V[edge[1] - 1][0])

    # 计算betweenness，选择计算模式mode
    def calcul_betw(self, mode=0):
        self.betweenness_list = {}
        for i in self.graph.edges():
            self.betweenness_list[asc(i)] = 0.0

        if mode == 0:
            self.__calcul_betw_sp()
        elif mode == 1:
            self.__calcul_betw_rs()
        elif mode == 2:
            self.__calcul_betw_rw()
        else:
            print("invalid mode!\nonly 3 modes: shortest-path: 0(default),resistor-network: 1,random-walk: 2")

    # find edge with the highest score and remove it from the network
    def cut_edge(self, mode=0):
        originEdgeNum = len(self.graph.edges)
        print(originEdgeNum)
        print(self.graph.edges)
        
        while(len(self.graph.edges) > originEdgeNum*2.0/3.0):
            index = max(self.betweenness_list.items(), key=operator.itemgetter(1))[0]
            del self.graph.edges[index]
        

    def evaluation(self):
        pass


a = Community(graph_dir='graph.txt')
a.calcul_betw(mode=2)
print(a.betweenness_list)
a.cut_edge(mode=2)
