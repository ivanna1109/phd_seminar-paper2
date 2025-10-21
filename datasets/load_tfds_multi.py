import os
import tensorflow as tf
import numpy as np
from rdkit import Chem
from visualize_mol_example import visualize_rdkit_molecule

def get_atom_features(atom):
    # Dummies, samo da bi RDKit mogao da vrati broj feature-a
    features = [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        atom.GetImplicitValence(),
        atom.GetFormalCharge(),
        atom.GetIsAromatic(),
        atom.IsInRingSize(3), atom.IsInRingSize(4), atom.IsInRingSize(5),
        atom.IsInRingSize(6), atom.IsInRingSize(7), atom.IsInRingSize(8),
    ]
    return np.array(features, dtype=np.float32)

def get_bond_features(bond):
    # Dummies, samo da bi RDKit mogao da vrati broj feature-a
    features = [
        bond.GetBondTypeAsDouble(),
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    return np.array(features, dtype=np.float32)

def count_labels(dataset, dataset_name):
    # Sada imamo 3 klase (0, 1, 2)
    counts = {'0': 0, '1': 0, '2': 0}
    
    print(f"\n--- Brojanje labela za {dataset_name} dataset ---")
    
    # Koristimo .as_numpy_iterator() za efikasnije brojanje velikih skupova
    total_count = 0
    for _, label in dataset.as_numpy_iterator():
        label_int = int(round(label)) # Labela je float, pretvaramo u int
        
        if label_int == 0:
            counts['0'] += 1
        elif label_int == 1:
            counts['1'] += 1
        elif label_int == 2:
            counts['2'] += 1
        else:
            print(f"Upozorenje: Pronađena neočekivana labela: {label_int}")
            
        total_count += 1
            
    print(f"Ukupno instanci u {dataset_name}: {total_count}")
    print(f"  Labela 0 (Neaktivan): {counts['0']} instanci")
    print(f"  Labela 1 (Selektivan ERa): {counts['1']} instanci")
    print(f"  Labela 2 (Dualna Aktivnost): {counts['2']} instanci")
    
    return counts

def load_tf_datasets(
    output_directory: str,
    common_global_features_list: list[str] = None,
    # === NOVO: Za ablaciju! ===
    include_global_features: bool = True 
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train_path = os.path.join(output_directory, 'train_dataset')
    val_path = os.path.join(output_directory, 'val_dataset')
    test_path = os.path.join(output_directory, 'test_dataset')

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        # ... (logika greške ostaje ista) ...
        print(f"Greška: Neki od putanja datasetova ne postoje u '{output_directory}'.")
        print("Uverite se da ste prethodno sačuvali datasete.")
        return None, None, None

    print(f"\n--- Počinjem učitavanje TF Datasetova iz '{output_directory}' ---")
    
    # ... (Kod za određivanje dummy_atom_dim i dummy_bond_dim ostaje isti) ...
    # Zbog prethodne greške, evo sigurne definicije dummy dimenzija ovde:
    try:
        dummy_atom_dim = get_atom_features(Chem.Atom(0)).shape[0]
        dummy_mol_for_bond_dim = Chem.MolFromSmiles('CC')
        if dummy_mol_for_bond_dim and dummy_mol_for_bond_dim.GetNumBonds() > 0:
            dummy_bond_dim = get_bond_features(dummy_mol_for_bond_dim.GetBonds()[0]).shape[0]
        else:
             dummy_bond_dim = 3
    except Exception:
        dummy_atom_dim = 12
        dummy_bond_dim = 3
        
    # === KLJUČNA IZMENA ZA ABLACIJU ===
    global_features_dim = 0
    if include_global_features:
        if common_global_features_list is None or not common_global_features_list:
            print("Greška: common_global_features_list mora biti prosleđen da bi se odredila dimenzija globalnih karakteristika.")
            return None, None, None
        
        global_features_dim = len(common_global_features_list)
        print(f"Učitavam datasete: Uključene Globalne Karakteristike (Dim: {global_features_dim})")
    else:
        # Ako je include_global_features=False, dimenzija MORA biti 0.
        global_features_dim = 0
        print("Učitavam datasete: ISKLJUČENE Globalne Karakteristike (Dim: 0)")


    reloaded_output_signature = (
        {
            'node_features': tf.TensorSpec(shape=(None, dummy_atom_dim), dtype=tf.float32),
            'edge_features': tf.TensorSpec(shape=(None, dummy_bond_dim), dtype=tf.float32),
            'adjacency_indices': tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
            'adjacency_values': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            # DINAMIČKI SHAPE
            'global_features': tf.TensorSpec(shape=(global_features_dim,), dtype=tf.float32),
            'num_nodes': tf.TensorSpec(shape=(), dtype=tf.int32),
            'num_edges': tf.TensorSpec(shape=(), dtype=tf.int32),
            'preferredName': tf.TensorSpec(shape=(), dtype=tf.string),
            'esr': tf.TensorSpec(shape=(), dtype=tf.string),
            'source_assay_name': tf.TensorSpec(shape=(), dtype=tf.string)
        },
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )

    # ... (ostatak koda za učitavanje sa tf.data.Dataset.load ostaje isti) ...
    try:
        loaded_train_ds = tf.data.Dataset.load(train_path, element_spec=reloaded_output_signature)
        loaded_val_ds = tf.data.Dataset.load(val_path, element_spec=reloaded_output_signature)
        loaded_test_ds = tf.data.Dataset.load(test_path, element_spec=reloaded_output_signature)
        print("Svi datasetovi uspešno učitani.")
    except Exception as e:
        print(f"Kritična greška pri učitavanju datasetova: {e}")
        return None, None, None
    
    print("\n--- Učitavanje TF Datasetova završeno ---")
    return loaded_train_ds, loaded_val_ds, loaded_test_ds

def element_details(element, output_dir):
    try:
        for features, label in element:
            preferred_name = features['preferredName'].numpy().decode('utf-8')
            
            
            clean_preferred_name = "".join(c for c in preferred_name if c.isalnum() or c in (' ', '.', '-')).replace(' ', '_')
            
        
            image_filename = os.path.join(output_dir, f"{clean_preferred_name}_mol.png")
            text_filename = os.path.join(output_dir, f"{clean_preferred_name}_details.txt")

            text_output_path_full = text_filename # Ovo je putanja za text fajl
            text_output_dir_full = os.path.dirname(text_output_path_full)
            if text_output_dir_full and not os.path.exists(text_output_dir_full):
                os.makedirs(text_output_dir_full, exist_ok=True)

            label_value = label.numpy()
            label_description = ""
            if label_value == 0.0:
                label_description = " (Potpuno Neaktivan)"
            elif label_value == 1.0:
                label_description = " (Selektivan ERa)"
            elif label_value == 2.0:
                label_description = " (Dualna Aktivnost)"

            with open(text_output_path_full, 'w') as f:
                f.write(f"Labela (hitc): {label_value}{label_description}\n")
        
                f.write(f"--- Detalji molekula: '{preferred_name}' ---\n")
                f.write(f"Naziv molekula (preferredName): {preferred_name}\n")
                f.write(f"Labela (hitc): {label.numpy()}\n")
                f.write(f"ESR: {features['esr'].numpy().decode('utf-8')}\n")
                f.write(f"Broj atoma : {features['num_nodes'].numpy()}\n")
                f.write(f"Broj veza: {features['num_edges'].numpy()}\n")
                f.write(f"Globalne karakteristike: {features['global_features'].numpy()}\n")
                f.write(f"Shape node_features: {features['node_features'].shape}\n")
                f.write(f"Shape edge_features: {features['edge_features'].shape}\n")
                f.write("\nMatrica susedsta (Sparse reprezentacija):\n")
                f.write(f"Adjacency_indices:\n{features['adjacency_indices'].numpy()}\n")
                f.write(f"Adjacency_values:\n{features['adjacency_values'].numpy()}\n")
                
                num_nodes = features['num_nodes'].numpy()
                adj_matrix_dense = np.zeros((num_nodes, num_nodes), dtype=np.float32)
                indices = features['adjacency_indices'].numpy()
                values = features['adjacency_values'].numpy()
                for i, (u, v) in enumerate(indices):
                    adj_matrix_dense[u, v] = values[i]
                f.write("\nMatrica susedstva (Gusta reprezentacija - prvih 5x5 ako je veća):\n")
                if num_nodes <= 5:
                    f.write(f"{np.array_str(adj_matrix_dense)}\n")
                else:
                    f.write(f"{np.array_str(adj_matrix_dense[:5, :5])}\n")
                    f.write(f"    ... (Prikazano samo prvih 5x5 od {num_nodes}x{num_nodes} matrice)\n")
            
            print(f"Detalji molekula sačuvani kao: {text_output_path_full}")
            
            visualize_rdkit_molecule(features, label, 
                                    show_atom_idx=True, 
                                    output_filepath=image_filename)
            
    except Exception as e:
        print(f"Greška pri obradi elementa: {e}")


if __name__ == "__main__":
    # Vaša lista globalnih feature-a
    COMMON_GLOBAL_FEATURES_LIST = [
        'monoisotopicMass', 'ac50',] # Proširite ovo sa stvarnom listom!

    # === SCENARIJ 1: PUNI MODEL (Grafovi + Globalni Feature-i) ===
    dataset_dir_full = '/home/ivana-milutinovic/Documents/Doktorske/BIORad/GitHub/BIO-info-multiclass/data-processing/dataset_for_multi/datasets/tfrecords_full'
    
    train_ds_full, val_ds_full, test_ds_full = load_tf_datasets(
        output_directory=dataset_dir_full,
        common_global_features_list=COMMON_GLOBAL_FEATURES_LIST,
        include_global_features=True # Uključeno
    )

    if train_ds_full:
        print("\n=== Analiza: PUNI MODEL ===")
        count_labels(train_ds_full, "Trening (FULL)")
        count_labels(val_ds_full, "Validacija (FULL)")
        count_labels(test_ds_full, "Test (FULL)")
        
        # Prikaz detalja prvog elementa (Globalni feature-i bi trebalo da su vidljivi)
        output_dir_full = 'data/output_molecules/full_example'
        print(f"\n--- Provera prvog elementa iz FULL Trening Skupa ({output_dir_full}) ---")
        element_details(train_ds_full.take(1), output_dir_full)


    # === SCENARIJ 2: ABLACIJA GLOBALNIH (Samo Grafovi) ===
    dataset_dir_graph_only = 'datasets/tfrecords_graph_only'
    
    train_ds_graph, val_ds_graph, test_ds_graph = load_tf_datasets(
        output_directory=dataset_dir_graph_only,
        common_global_features_list=COMMON_GLOBAL_FEATURES_LIST,
        include_global_features=False # ISKLJUČENO
    )

    if train_ds_graph:
        print("\n=== Analiza: GRAFOVI SAMO ===")
        count_labels(train_ds_graph, "Trening (GRAPH ONLY)")
        count_labels(val_ds_graph, "Validacija (GRAPH ONLY)")
        count_labels(test_ds_graph, "Test (GRAPH ONLY)")

        # Prikaz detalja prvog elementa (Globalni feature-i bi trebalo da su prazni/0-dimenzionalni)
        output_dir_graph = 'data/output_molecules/graph_only_example'
        print(f"\n--- Provera prvog elementa iz GRAPH ONLY Trening Skupa ({output_dir_graph}) ---")
        element_details(train_ds_graph.take(1), output_dir_graph)

    print("\nUspešno učitani i analizirani TF Datasetovi za Ablacionu Studiju.")


##### MAIIIN METOD####
if __name__ == "__main__":
    COMMON_GLOBAL_FEATURES_LIST = [
        'monoisotopicMass', 'ac50',] 
    base_dir = '/home/ivana-milutinovic/Documents/Doktorske/BIORad/GitHub/BIO-info-multiclass/data-processing/dataset_for_multi/'

    # =========================================================================
    # SCENARIJ A: PUNI MODEL (Grafovi + Globalni Feature-i)
    # =========================================================================
    print("======================================================================")
    print("=== SCENARIJ A: PUNI MODEL (Grafovi + Globalni Feature-i) ===")
    print("======================================================================")
    
    dataset_dir_full = f'{base_dir}datasets/tfrecords_full'

    # Učitavanje sa uključivanjem globalnih feature-a (include_global_features=True)
    train_ds_full, val_ds_full, test_ds_full = load_tf_datasets(
        output_directory=dataset_dir_full,
        common_global_features_list=COMMON_GLOBAL_FEATURES_LIST,
        include_global_features=True
    )

    if train_ds_full:
        print("\n[PUNI MODEL] Učitani svi skupovi podataka.")

        ### Veličine Skupova Podataka
        print("\n--- Veličine TF Datasetova (FULL) ---")
        count_labels(train_ds_full, "Trening (FULL)")
        count_labels(val_ds_full, "Validacija (FULL)")
        count_labels(test_ds_full, "Test (FULL)")
        
        ### Detalji Prvog Elementa Trening Skupa
        print("\n--- Provera prvog elementa iz FULL Trening Skupa (Globalni feature-i bi trebalo da su prisutni) ---")
        output_dir_full = f'{base_dir}data/output_molecules/full_example'
        # Uverite se da putanja za vizualizaciju postoji!
        os.makedirs(output_dir_full, exist_ok=True) 
        element_details(train_ds_full.take(1), output_dir_full)
    else:
        print("\n[PUNI MODEL] Nije moguće učitati sve datasete. Proverite putanje i dimenzije.")

    
    # =========================================================================
    # SCENARIJ B: ABLACIJA GLOBALNIH (Samo Grafovi)
    # =========================================================================
    print("\n\n======================================================================")
    print("=== SCENARIJ B: ABLACIJA GLOBALNIH (Samo Grafovi) ===")
    print("======================================================================")
    
    dataset_dir_graph_only = f'{base_dir}datasets/tfrecords_graph_only'

    # Učitavanje sa isključivanjem globalnih feature-a (include_global_features=False)
    train_ds_graph, val_ds_graph, test_ds_graph = load_tf_datasets(
        output_directory=dataset_dir_graph_only,
        common_global_features_list=COMMON_GLOBAL_FEATURES_LIST,
        include_global_features=False
    )

    if train_ds_graph:
        print("\n[SAMO GRAFOVI] Učitani svi skupovi podataka.")

        ### Veličine Skupova Podataka
        print("\n--- Veličine TF Datasetova (GRAPH ONLY) ---")
        count_labels(train_ds_graph, "Trening (GRAPH ONLY)")
        count_labels(val_ds_graph, "Validacija (GRAPH ONLY)")
        count_labels(test_ds_graph, "Test (GRAPH ONLY)")

        ### Detalji Prvog Elementa Trening Skupa
        print("\n--- Provera prvog elementa iz GRAPH ONLY Trening Skupa (Globalni feature-i bi trebalo da budu prazni/shape=(0,)) ---")
        output_dir_graph = f'{base_dir}data/output_molecules/graph_only_example'
        os.makedirs(output_dir_graph, exist_ok=True) 
        element_details(train_ds_graph.take(1), output_dir_graph)
    else:
        print("\n[SAMO GRAFOVI] Nije moguće učitati sve datasete. Proverite putanje i dimenzije (posebno da li je global_features_dim=0 u load funkciji).")

    print("\n--- Analiza Ablacione Studije ZAVRŠENA ---")
    