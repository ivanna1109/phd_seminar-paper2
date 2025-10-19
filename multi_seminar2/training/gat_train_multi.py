import sys
import os

sys.path.append('/home/ivanam/BIO-Info/bio_info/multi_seminar2/')

import load_tfds_multi as tfds
import metrics.calculate_metrics as metric
from models.gat import GATMultiClass as GAT
import multi_spektral.tf_to_spektral as tf_to_s
from multi_spektral.spektral_dataset import MyDataset
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import AUC
from spektral.data import BatchLoader
from sklearn.utils.class_weight import compute_class_weight
import math
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
#import evaluate_model as eval


COMMON_GLOBAL_FEATURES_LIST = [
        'monoisotopicMass', 'ac50',]

#Graphs + global features
#dataset_dir = '/home/ivanam/BIO-Info/bio_info/multi_seminar2/datasets/tfrecords_full'
#Graphs only
dataset_dir = '/home/ivanam/BIO-Info/bio_info/multi_seminar2/datasets/tfrecords_graph_only'
print("GCN Multi Training (Graphs Only)..............................................")

print("GAT Multi Training (Graphs + Global Features)..............................................")

train_ds, val_ds, test_ds = tfds.load_tf_datasets(output_directory=dataset_dir, common_global_features_list=COMMON_GLOBAL_FEATURES_LIST, include_global_features=False)
train_size = train_ds.cardinality().numpy()
print(f"Veličina trening skupa: {train_size}")
#tfds.count_labels(train_ds, 'Training')
print("-"*56)
val_size = val_ds.cardinality().numpy()
print(f"Veličina validacionog skupa: {val_size}")
#tfds.count_labels(val_ds, 'Val')
print("-"*56)
test_size = test_ds.cardinality().numpy()
print(f"Veličina test skupa: {test_size}")
#tfds.count_labels(test_ds, 'Test')
print("-"*56)
        
print("\n--- Konvertovanje trening skupa ---")
X_train_graphs, y_train_one_hot = tf_to_s.convert_tf_dataset_to_spektral(train_ds)

num_features = 0
num_labels = 3
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


batch_size = 64
print("Prelazimo na batch loadere...")
train_loader = BatchLoader(train_dataset, batch_size=batch_size)
val_loader = BatchLoader(val_dataset, batch_size=batch_size)
test_loader = BatchLoader(test_dataset, batch_size=batch_size)

batch = train_loader.__next__()
inputs, target = batch
x, a = inputs
print(x.shape)
print(a.shape)
print(target.shape)

for batch in train_loader:
    x_batch, l_batch = batch
    print("Dimenzije x_batch0:", x_batch[0].shape)
    print("Dimenzije x_batch1:", x_batch[1].shape)
    print("Dimenzije a_batch:", l_batch.shape)
    break

print("Tri skupa prilagodjena za trening, val i test kreirani da bi mogli pustiti u GAT.")
n_heads = 8
units = 32
dropout_rate = 0.25
learning_rate = 0.00254
batch_size = 64
    
num_epoch = 50
model = GAT(units, n_heads, dropout_rate, num_labels)

print("Kreiran GAT model.")

# Dodajemo LearningRateScheduler i EarlyStopping sheduler za monitoring treninga
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#custom call back za pracene f1 val score-a
class F1ScoreCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_f1 = logs.get('val_f1_score')
        if val_f1 is not None:
            print(f'\nEpoch {epoch + 1}: val_f1_score = {val_f1:.4f}')

f1_score_callback = F1ScoreCallback()
print("Dodati LR i ES i custom f1 score")

model.compile(tf.keras.optimizers.Adam(learning_rate=0.00072), #lr po optuna proracunu
              loss='categorical_crossentropy', 
              metrics=['accuracy', 
                       metric.f1_score,
                       AUC(name='roc_auc', multi_label=True),
                    tf.keras.metrics.AUC(curve='PR', name='average_precision')])
                    #metric.weighted_f1_score,
                    #metric.balanced_accuracy

print("Model kompajliran.")

y_train_labels = np.argmax(y_train_one_hot, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weights_dict = dict(enumerate(class_weights))

train_steps_per_epoch = math.ceil(len(X_train_graphs) / batch_size)
val_steps_per_epoch = math.ceil(len(X_val_graphs) / batch_size)

print("Start treninga...........................")
history = model.fit(
    train_loader,
    validation_data=val_loader,
    epochs = num_epoch,
    steps_per_epoch=train_steps_per_epoch,
    validation_steps=val_steps_per_epoch,
    class_weight = class_weights_dict,
    callbacks=[lr_scheduler, early_stopping, f1_score_callback]
)

print("Zavrsen trening....")
print("Pisemo sumarry modela u fajl......")

final_dir = '/home/ivanam/BIO-Info/bio_info/multi_seminar2/training/final_results_graphs_only/gat'

with open(f'{final_dir}/model_summary-gat-e{num_epoch}.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
print("Zavrseno pisanje u fajl.")

print("Pisemo history treninga..")
history_df = pd.DataFrame(history.history)
history_df.to_csv(f'{final_dir}/gat_history_e{num_epoch}.csv', index=False)

print("Cuvanje tezina modela..")
model.save_weights(f'{final_dir}/gat_model_e{num_epoch}.h5')

output_file_path = f"{final_dir}/gat_test_results-e{num_epoch}.txt"

# Otvori fajl za pisanje i preusmeri standardni izlaz
with open(output_file_path, "w") as f:
    # Sačuvaj originalni stdout
    original_stdout = sys.stdout
    # Preusmeri stdout na fajl
    sys.stdout = f

    # Sve komande koje ispisuju na konzolu sada će pisati u fajl
    steps_for_test = math.ceil(len(X_test_graphs) / batch_size)

    test_loss, test_acc, test_f1_score, test_roc_auc, test_avg_precision = model.evaluate(test_loader.load(), batch_size= batch_size, steps =steps_for_test, verbose=1)
    print("Evalucija modela....")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1Score: {test_f1_score:.4f}")
    print(f"Test ROC AUC: {test_roc_auc:.4f}")
    print(f"Test Average Precision: {test_avg_precision:.4f}")

    print("\nIdemo na classification report....")

    y_true_list = []
    y_pred_list = []

    for step, batch in enumerate(test_loader):
        if step >= steps_for_test:
            break
        inputs, target = batch
        x, a = inputs
        y_true_batch = target 
        y_pred_batch = model.predict_on_batch((x, a))
        y_pred_batch = np.argmax(y_pred_batch, axis=-1)
        
        if len(y_true_batch.shape) > 1 and y_true_batch.shape[1] > 1:
            y_true_batch = np.argmax(y_true_batch, axis=-1)
        
        y_true_list.extend(y_true_batch)
        y_pred_list.extend(y_pred_batch)

    print(f"\nDužina y_true_list: {len(y_true_list)}")
    # Printanje prvog elementa y_true_list može biti nepotrebno u finalnom izveštaju
    # print(f"Prvi element y_true_list: {y_true_list[0]}") 
    print(f"Dužina y_pred_list: {len(y_pred_list)}")
    # Printanje prvog elementa y_pred_list može biti nepotrebno u finalnom izveštaju
    # print(f"Prvi element y_pred_list: {y_pred_list[0]}")
    
    y_true = y_true_list
    y_pred = y_pred_list

    # Classification report će biti ispisan u fajl
    print("\n" + classification_report(y_true, y_pred))

     # 2. MATRICA KONFUZIJE (NOVI DEO)
    cm = confusion_matrix(y_true, y_pred)
    
    # Lepše formatiranje matrice pre ispisa
    print("\n" + "--- Matrica Konfuzije (Confusion Matrix) ---")
    
    # Dodajemo naslove kolona (klase 0, 1, 2)
    # Možete koristiti labels=[0, 1, 2] ako su klase numeričke
    class_labels = ['Neaktivni (0)', 'Aktivni na ESR1 (1)', 'Dualno aktivni (2)'] 
    
    # Formatiranje u DataFrame za čitljivost u fajlu
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    
    # Dodavanje naslova reda
    cm_df.index.name = 'Stvarna Klasa'
    cm_df.columns.name = 'Predviđena Klasa'
    
    print(cm_df)


    print("Sve uspesno zavrseno.")

    # Vrati standardni izlaz na konzolu
    sys.stdout = original_stdout

print(f"\nRezultati su uspešno sačuvani u '{output_file_path}'")