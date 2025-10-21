import tensorflow as tf
from spektral.layers import GCNConv, GlobalMaxPool
from tensorflow.keras.layers import Dense, Dropout

class GCN_MultiClass(tf.keras.Model):
    def __init__(self, num_features, num_labels, hidden_units, dense_units, dropout_rate, l2_rate=0.01):
        super(GCN_MultiClass, self).__init__()
        # ... (Svi GNN, Dropout i Dense slojevi ostaju isti) ...
        
        self.conv1 = GCNConv(hidden_units, activation='relu')
        self.dropout1 = Dropout(dropout_rate)

        self.conv2 = GCNConv(hidden_units, activation='relu')
        self.dropout2 = Dropout(dropout_rate)

        self.conv3 = GCNConv(hidden_units, activation='relu')
        
        # Ostajete na GlobalMaxPool
        self.global_pool = GlobalMaxPool() 

        self.dense1 = Dense(dense_units[0], activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l2(l2_rate))
        self.dropout3 = Dropout(dropout_rate)
        
        self.dense2 = Dense(dense_units[1], activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l2(l2_rate))
        self.output_layer = Dense(num_labels, activation='softmax')


    def call(self, inputs, training=False):
        x, a, i, u = inputs 
        
        # 1. GNN Faze (GCN koristi samo X i A)
        x = self.conv1([x, a])
        x = self.dropout1(x, training=training)

        x = self.conv2([x, a])
        x = self.dropout2(x, training=training)
        
        x = self.conv3([x, a])
        
        pooled_x = self.global_pool([x, i]) 
        
        combined_features = tf.concat([pooled_x, u], axis=-1)
        x = combined_features 
        
        x = self.dense1(x)
        x = self.dropout3(x, training=training)
        
        x = self.dense2(x)
        output = self.output_layer(x)

        return output