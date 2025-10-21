import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import scipy.sparse as sp

class CustomGraphSequence(Sequence):
    """
    Keras Sequence za ručno kreiranje batch-eva za GNN. 
    Garancija isporuke (X, A, I, U).
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset 
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset))
        self.on_epoch_end() # Mešanje na početku

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        # 1. Dobijanje Spektral Graph objekata
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_graphs_list = [self.dataset[i] for i in batch_indices]

        list_x = []
        list_u = []
        list_y = []
        
        # Komponente za Sparse Matriku Susedstva (A)
        all_indices = []
        all_values = []
        
        # Komponente za Pooling Indekse (I)
        pooling_indices = []
        current_offset = 0 # Čvorni ofset za Sparse matricu
        
        # 2. Iteracija i prikupljanje podataka
        for i, g in enumerate(batch_graphs_list):
            
            # Prikupljanje X, U, Y
            list_x.append(g.x)
            list_u.append(g.u)
            list_y.append(g.y)

            a_coo = g.a.tocoo() 
        
            rows = a_coo.row + current_offset
            cols = a_coo.col + current_offset
            
            all_indices.append(np.stack([rows, cols], axis=1))
            all_values.append(a_coo.data)
            
            pooling_indices.append(np.full(g.n_nodes, i, dtype=np.int64))
            
            current_offset += g.n_nodes # Ažuriranje ofseta za sledeći graf

        X = tf.constant(np.concatenate(list_x, axis=0), dtype=tf.float32)

        # U: Globalne Karakteristike
        U = tf.constant(np.array(list_u), dtype=tf.float32)
        
        # Y: Labele
        Y_batch = tf.constant(np.array(list_y), dtype=tf.float32)

        # I: Pooling Indeksi
        I = tf.constant(np.concatenate(pooling_indices), dtype=tf.int64)

        # A: Sparse Matrica Susedstva
        final_indices = np.concatenate(all_indices, axis=0)
        final_values = np.concatenate(all_values, axis=0)
        total_nodes = current_offset # Ukupan broj čvorova u batchu
        
        A = tf.SparseTensor(
            indices=final_indices, 
            values=final_values, 
            dense_shape=[total_nodes, total_nodes]
        )
        
        A = tf.sparse.reorder(A) 

        # 4. Finalni format za modele
        model_inputs = (X, A, I, U)
        return model_inputs, Y_batch

    def on_epoch_end(self):
        # Mešanje indeksa grafa nakon svake epohe
        np.random.shuffle(self.indices)