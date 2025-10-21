import tensorflow as tf
from spektral.layers import GATConv, GlobalSumPool
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

class GATMultiClass(Model):
    def __init__(self, units, n_heads, dropout_rate, num_classes):
        super(GATMultiClass, self).__init__()
        self.gat1 = GATConv(units, n_heads, dropout_rate=dropout_rate, activation='relu')
        self.gat2 = GATConv(units, n_heads, dropout_rate=dropout_rate, activation='relu')
        self.batch_norm1 = BatchNormalization()
        self.batch_norm2 = BatchNormalization()
        self.dropout = Dropout(dropout_rate)
        
        self.pool = GlobalSumPool() 
        
        # Dense slojevi
        self.dense1 = Dense(units * 2, activation='relu')
        self.dense2 = Dense(units, activation='relu')
        self.output_layer = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x, a, i, u = inputs 
        
        x = self.gat1([x, a])
        x = self.batch_norm1(x, training=training)
        x = self.dropout(x, training=training)
        
        x = self.gat2([x, a])
        x = self.batch_norm2(x, training=training)
        x = self.dropout(x, training=training)
        
        pooled_x = self.pool([x, i]) 

        combined_features = tf.concat([pooled_x, u], axis=-1)
        x = combined_features 
        
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training) 
        output = self.output_layer(x)
        return output