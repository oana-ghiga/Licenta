import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def read_matrices(file_path)->dict:
    '''
    Read adjacency matrices from a file
    :param file_path:  path to the file
    :return: dictionary of matrices
    '''
    with open(file_path, 'r') as file:
        matrices = {}
        matrix_name = None
        matrix_data = []

        for line in file:
            line = line.strip()
            if line.startswith('# Matrix'):
                if matrix_name is not None and matrix_data:
                    matrices[matrix_name] = np.array(matrix_data)
                matrix_name = line.split()[2]
                matrix_data = []
            elif line:
                matrix_data.append(list(map(int, line.split())))

        if matrix_name is not None and matrix_data:
            matrices[matrix_name] = np.array(matrix_data)

    return matrices


def calculate_metrics(matrix)->tuple:
    '''
    Calculate the degree and clustering coefficient of a network
    :param matrix: adjacency matrix
    :return: degree and clustering coefficient
    '''
    G = nx.from_numpy_array(matrix)
    degree = dict(G.degree())
    clustering_coeff = nx.clustering(G)

    return degree, clustering_coeff


def plot_network(matrix, matrix_name)->None:
    '''
    Plot the network graph
    :param matrix: adjacency matrix
    :param matrix_name: name of the matrix
    :return: None
    '''
    G = nx.from_numpy_array(matrix)

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_color='darkred', node_size=500)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='salmon')
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif', font_color='white')

    plt.title(f"Network Graph: {matrix_name}", fontsize=14, color='maroon')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    '''
    Main function to read the adjacency matrices, calculate the degree and clustering coefficient of the network,
    show the network graph and plot the degree and clustering coefficient comparison.
    :return: None
    '''
    file_path = 'adj_matrices.txt'
    matrices = read_matrices(file_path)

    degree_all = {}
    clustering_coeff_all = {}

    for name, matrix in matrices.items():
        degree, clustering_coeff = calculate_metrics(matrix)
        degree_all[name] = degree
        clustering_coeff_all[name] = clustering_coeff

        plot_network(matrix, name)  #building the network graph


    #separate healthy and sick cases
    healthy_names = [name for name in degree_all.keys() if 'healthy' in name]
    sick = [name for name in degree_all.keys() if 'healthy' not in name]

    #healthy cases degree
    plt.figure(figsize=(12, 6))
    for name in healthy_names:
        degree = degree_all[name]
        plt.plot(list(degree.keys()), list(degree.values()), marker='o', label=name)
    plt.title('Degree Comparison (Physiological)', fontsize=14, color='green')
    plt.xlabel('Node')
    plt.ylabel('Degree')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #sick cases degree
    plt.figure(figsize=(12, 6))
    for name in sick:
        degree = degree_all[name]
        plt.plot(list(degree.keys()), list(degree.values()), marker='o', label=name)
    plt.title('Degree Comparison (Pathological)', fontsize=14, color='red')
    plt.xlabel('Node')
    plt.ylabel('Degree')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #healthy cases clustering
    plt.figure(figsize=(12, 6))
    for name in healthy_names:
        clustering_coeff = clustering_coeff_all[name]
        plt.plot(list(clustering_coeff.keys()), list(clustering_coeff.values()), marker='o', label=name)
    plt.title('Clustering Coefficient Comparison (Physiological)', fontsize=14, color='green')
    plt.xlabel('Node')
    plt.ylabel('Clustering Coefficient')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #sick cases clustering
    plt.figure(figsize=(12, 6))
    for name in sick:
        clustering_coeff = clustering_coeff_all[name]
        plt.plot(list(clustering_coeff.keys()), list(clustering_coeff.values()), marker='o', label=name)
    plt.title('Clustering Coefficient Comparison (Pathological)', fontsize=14, color='red')
    plt.xlabel('Node')
    plt.ylabel('Clustering Coefficient')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
