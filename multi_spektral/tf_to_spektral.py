import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from spektral.data import Graph

def convert_tf_dataset_to_spektral(tf_dataset):
    graphs = []
    labels = []
    NUM_CLASSES = 3  # <--- KRITIČNA IZMENA: 3 klase

    print(f"Pretvaranje TF Dataset-a u Spektral Graph objekte i One-Hot enkodiranje labela za {NUM_CLASSES} klase...")
    
    for features, label in tf_dataset:
        # Pretvori labele u skalarnu NumPy float vrednost
        label_numpy = label.numpy()
        
        # One-hot enkodiranje za 3 klase (0, 1, ili 2)
        # 0 -> [1., 0., 0.]
        # 1 -> [0., 1., 0.]
        # 2 -> [0., 0., 1.]
        one_hot_label = np.zeros(NUM_CLASSES, dtype=np.float32)
        
        if label_numpy in [0.0, 1.0, 2.0]:
            # Koristimo int() da pretvorimo float labelu u ceo broj za indeksiranje
            index = int(label_numpy)
            one_hot_label[index] = 1.0
        else:
            # Ako labela nije očekivana, preskačemo je ili postavljamo na npr. klasu 0
            print(f"Upozorenje: Neočekivana labela: {label_numpy}. Postavljam na prvu klasu [1., 0., 0.].")
            one_hot_label[0] = 1.0
        
        labels.append(one_hot_label)

        # Izvuci podatke iz TF features za Spektral Graph
        x = features['node_features'].numpy()
        a_indices = features['adjacency_indices'].numpy()
        a_values = features['adjacency_values'].numpy()
        
        # Spektral zahteva sparse matricu susedstva za 'a'
        num_nodes = features['num_nodes'].numpy()
        
        if a_indices.size == 0:
            a = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32).tocsr()
        else:
            rows = a_indices[:, 0]
            cols = a_indices[:, 1]
            data = a_values
            # Kreiramo sparse matricu i konvertujemo u CSR format
            a = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()

        # Globalne karakteristike (u Spektralu se obično zovu 'u')
        u = features['global_features'].numpy()
        
        # Ostale informacije za Spektral Graph
        molecule_name = features['preferredName'].numpy().decode('utf-8')

        # Kreiraj Spektral Graph objekat
        # Koristimo one-hot enkodiranu labelu za y
        graph = Graph(x=x, a=a, u=u, y=one_hot_label, molecule_name=molecule_name)
        graphs.append(graph)
    print_graph(graphs[0])
    return graphs, np.array(labels, dtype=np.float32)

def print_graph(graph):
    print(f"\n--- Detalji Grafa ({graph.molecule_name}) ---")
    print(f"Dimenzija Matrice Karakteristika Čvorova (X.shape): {graph.x.shape}")
    print(f"Broj Čvorova: {graph.n_nodes}")
    print(f"Broj Karakteristika Čvorova: {graph.n_node_features}")

# Ispis dimenzija Sparse Matrice Susedstva (A)
    print(f"Dimenzija Matrice Susedstva (A.shape): {graph.a.shape}")
    print(f"Broj Ivica (A.nnz): {graph.a.nnz}")

    # Ispis dimenzija Globalnih Karakteristika (U)
    #   U je obično 1D vektor
    print(f"Dimenzija Globalnih Karakteristika (U.shape): {graph.u.shape}")
    print(f"Globalne Karakteristike (prve 5, ako ih ima): {graph.u[:5]}")

    # Ispis Labele (Y)
    print(f"Labela (Y): {graph.y}")
    print("---------------------------------")

def convert_tf_dataset_to_spektral_to_overfitt(tf_dataset, ds_name=""):
    graphs = []
    labels = []
    print(f"Trenutno se obradjuje skup tf dataset: {ds_name}")
    print(f"Pretvaranje TF Dataset-a u Spektral Graph objekte i One-Hot enkodiranje labela...")
    
    for features, label in tf_dataset:
        # Pretvori labele u NumPy array
        label_numpy = label.numpy()
        
        # One-hot enkodiranje za multi klase (0, 1, 2, 3)
        # Kreiramo vektor dimenzije 4
        one_hot_label = np.zeros(4, dtype=np.float32)

        # Uveravamo se da je labela u opsegu 0-3 pre enkodiranja
        if 0 <= label_numpy <= 3:
            one_hot_label[int(label_numpy)] = 1.0
        else:
            print(f"Upozorenje: Neočekivana labela: {label_numpy}. One-hot enkodiranje možda neće biti ispravno.")
        
        labels.append(one_hot_label)

        # Izvuci podatke iz TF features za Spektral Graph
        x = features['node_features'].numpy()
        a_indices = features['adjacency_indices'].numpy()
        a_values = features['adjacency_values'].numpy()
        
        # Spektral zahteva sparse matricu susedstva za 'a'
        num_nodes = features['num_nodes'].numpy()
        if a_indices.size == 0:
            a = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32).tocsr()
        else:
            rows = a_indices[:, 0]
            cols = a_indices[:, 1]
            data = a_values
            a = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()

        u = features['global_features'].numpy()
        
        # Ostale informacije za Spektral Graph
        molecule_name = features['preferredName'].numpy().decode('utf-8')

        # Kreiraj Spektral Graph objekat
        graph = Graph(x=x, a=a, u=u, y=one_hot_label, molecule_name=molecule_name)
        graphs.append(graph)
        
    return graphs, np.array(labels, dtype=np.float32)