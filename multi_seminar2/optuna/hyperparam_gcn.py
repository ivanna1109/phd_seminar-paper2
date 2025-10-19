import sys
import os

sys.path.append('/home/ivanam/BIO-Info/bio_info/multi_seminar2/')

import load_tfds_multi as tfds
import metrics.calculate_metrics as metric
from models.gcn import GCN_MultiClass as GCN
import multi_spektral.tf_to_spektral as tf_to_s
from multi_spektral.spektral_dataset import MyDataset
import pandas as pd
import optuna
import tensorflow as tf
from spektral.data import BatchLoader
from sklearn.utils.class_weight import compute_class_weight
import math
import numpy as np
#import evaluate_model as eval
import numpy as np
import math

COMMON_GLOBAL_FEATURES_LIST = [
        'monoisotopicMass', 'ac50',]

dataset_dir = '/home/ivanam/BIO-Info/bio_info/multi_seminar2/datasets/tfrecords_full'

print("GCN OPTUNA Training (FULL SET)..............................................")

train_ds, val_ds, test_ds = tfds.load_tf_datasets(output_directory=dataset_dir, common_global_features_list=COMMON_GLOBAL_FEATURES_LIST)
train_size = train_ds.cardinality().numpy()
print(f"Veličina trening skupa: {train_size}")
#tfds.count_labels(train_ds, 'Training')
print("-"*56)
val_size = val_ds.cardinality().numpy()
#print(f"Veličina validacionog skupa: {val_size}")
#tfds.count_labels(val_ds, 'Val')
print("-"*56)
test_size = test_ds.cardinality().numpy()
#print(f"Veličina test skupa: {test_size}")
#tfds.count_labels(test_ds, 'Test')
print("-"*56)
        
print("\n--- Konvertovanje trening skupa ---")
X_train_graphs, y_train_one_hot = tf_to_s.convert_tf_dataset_to_spektral(train_ds)

num_features = 0
num_labels =  3  # Broj klasa (podela po jačini aktivnosti u odnosu ac50)
if X_train_graphs: 
    num_features = X_train_graphs[0].x.shape[-1] 
    print(f"Automatski određujemo num_features: {num_features}")
else:
    num_features = 0 
    print("Upozorenje: X_train_graphs je prazan, num_features postavljeno na 0")

        
print("\n--- Konvertovanje validacionog skupa ---")
X_val_graphs, y_val_one_hot = tf_to_s.convert_tf_dataset_to_spektral(val_ds)
        
print("\n--- Konvertovanje test skupa ---")
X_test_graphs, y_test_one_hot = tf_to_s.convert_tf_dataset_to_spektral(test_ds)

print("\nKreiranje dataset instanci na osnovu konvertovanih podataka za Spektral..")
train_dataset = MyDataset(X_train_graphs, y_train_one_hot)
val_dataset = MyDataset(X_val_graphs, y_val_one_hot)
test_dataset = MyDataset(X_test_graphs, y_test_one_hot)

print(f"\nDimenzije finalnih datasetova:")
print(f"Train dataset: {len(train_dataset)} grafova, labeli oblika: {train_dataset.labels_data.shape}")
print(f"Validation dataset: {len(val_dataset)} grafova, labeli oblika: {val_dataset.labels_data.shape}")
print(f"Test dataset: {len(test_dataset)} grafova, labeli oblika: {test_dataset.labels_data.shape}")

print(f"Broj labela u train dataset: {len(train_dataset.labels_data)}")
print(f"Broj grafova u train datasetu: {len(train_dataset.graphs_data)}")
print("Prelazimo na batch loadere...")

train_loader = BatchLoader(train_dataset, batch_size=16)
val_loader = BatchLoader(val_dataset, batch_size=16)
test_loader = BatchLoader(test_dataset, batch_size=16)
i = 0


print("Idemo u objective funkciju....")

def objective(trial):
    global i
    hidden_units = trial.suggest_categorical('hidden_units', [32, 64, 96, 128])
    
    # 2. Precizniji raspon za Dense slojeve
    dense_units = [
        trial.suggest_categorical(f'dense_units_{j}', [32, 64, 128]) 
        for j in range(2) # Koristite j umesto i, jer je i globalna
    ]
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.4)
    learning_rate = trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # Kreiraj GCN_MultiClass model sa Optuna parametrima
    model = GCN(num_features, num_labels, hidden_units, dense_units, dropout_rate)

    y_train_labels = np.argmax(y_train_one_hot, axis=1)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
    class_weights_dict = dict(enumerate(class_weights))   
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=[metric.f1_score])  # Koristi F1-score kao metricu
    
    steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    validation_steps = math.ceil(len(val_dataset) / batch_size)
    test_steps = math.ceil(len(test_dataset) / batch_size)

    
    history = model.fit(train_loader.load(),
                    validation_data=val_loader.load(),
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    epochs=100,
                    batch_size=batch_size,
                    #class_weight=class_weights_dict,
                    verbose=0)
    
    print("Pisemo history treninga..")
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'/home/ivanam/BIO-Info/bio_info/multi_seminar2/optuna/results/gcn/history_results_{i}.csv', index=False)
    i+=1

    f1 = model.evaluate(test_loader.load(),
    steps=test_steps,
    verbose=0)[1]  # F1-score je na indeksu 1
    
    return f1  # Maksimizujemo F1-score za optimizaciju


nt = 100
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=nt)

output_file_path = "/home/ivanam/BIO-Info/bio_info/multi_seminar2/optuna/results/gcn/optuna_final_results.txt" 

with open(output_file_path, "w") as f:
    original_stdout = sys.stdout
    sys.stdout = f

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {:.5f}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
