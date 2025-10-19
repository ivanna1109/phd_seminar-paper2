import warnings

# Potiskuje specifično RDKit upozorenje o zastarelosti funkcije GetValence
warnings.filterwarnings(
    "ignore", 
    message="DEPRECATION WARNING: please use GetValence(getExplicit=False)", 
    category=DeprecationWarning
)
from rdkit import RDLogger

# Utišavanje svih RDKit upozorenja i informacija (postavlja nivo na CRITICAL)
RDLogger.logger().setLevel(RDLogger.CRITICAL)

import os
import re
import pandas as pd
import numpy as np
from rdkit import Chem # Trebaće nam za potencijalno učitavanje Mola, ali za ovu funkciju samo putanje
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_all_data_into_maps(
    csv_files_directory: str,
    mol_files_parent_directory: str
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]]]:
    csv_dataframes = {}
    mol_paths_by_csv_name = {}

    print(f"Skeniram CSV direktorijum: {csv_files_directory}")
    # --- Učitavanje svih CSV fajlova ---
    for item_name in os.listdir(csv_files_directory):
        item_path = os.path.join(csv_files_directory, item_name)
        
        if os.path.isfile(item_path) and item_name.lower().endswith('.csv'):
            csv_name_without_ext = os.path.splitext(item_name)[0]
            try:
                df = pd.read_csv(item_path)
                csv_dataframes[csv_name_without_ext] = df
                print(f"  Učitan CSV: {item_name} sa {len(df)} redova.")
            except Exception as e:
                print(f"  Greška pri učitavanju CSV-a '{item_name}': {e}. Preskačem.")
    
    print(f"\nSkeniram MOL direktorijum: {mol_files_parent_directory}")
    # --- Mapiranje putanja MOL fajlova ---
    # Prolazimo kroz sve stavke u roditeljskom direktorijumu MOL fajlova
    for subfolder_name in os.listdir(mol_files_parent_directory):
        subfolder_path = os.path.join(mol_files_parent_directory, subfolder_name)

        if os.path.isdir(subfolder_path):
            current_mol_map = {} # Mapa za mol fajlove unutar ovog podfoldera
            
            # Prolazimo kroz fajlove unutar podfoldera
            for mol_file_name in os.listdir(subfolder_path):
                if mol_file_name.lower().endswith('.mol'):
                    mol_name_without_ext = os.path.splitext(mol_file_name)[0]
                    full_mol_path = os.path.join(subfolder_path, mol_file_name)
                    
                    # ### APLIKACIJA STANDARDIZACIJE NA IME MOL FAJLA ###
                    standardized_mol_name = standardize_mol_name(mol_name_without_ext)
                    current_mol_map[standardized_mol_name] = full_mol_path
            
            if current_mol_map: # Dodajemo samo ako ima pronađenih mol fajlova
                mol_paths_by_csv_name[subfolder_name] = current_mol_map
                print(f"  Pronađeno {len(current_mol_map)} MOL fajlova u podfolderu '{subfolder_name}'.")
            else:
                print(f"  Nema MOL fajlova u podfolderu '{subfolder_name}'.")

    print("\nUčitavanje podataka završeno.")
    print(f"Ukupno učitanih CSV fajlova: {len(csv_dataframes)}")
    print(f"Ukupno mapiranih MOL podfoldera: {len(mol_paths_by_csv_name)}")

    return csv_dataframes, mol_paths_by_csv_name

def determine_assay_type(csv_name: str) -> str:
    csv_name_lower = csv_name.lower()
    if 'tox21' in csv_name_lower: # Npr. 'assay_binder_data'
        return 'tox21'
    else: 
        return 'binders'
    
def standardize_mol_name(name: str) -> str:
    # Prebacivanje u mala slova
    name = name.lower()
    # Uklanjanje svih karaktera koji NISU slova (a-z) ili brojevi (0-9)
    # ^ (caret) unutar [] negira set karaktera, pa [^a-z0-9] znači "bilo koji karakter koji nije a-z ili 0-9"
    name = re.sub(r'[^a-z0-9]', '', name)
    return name
    
# --- Glavna funkcija za obradu i kombinovanje podataka (REVIDIRANA ZA 4 KLASE AGREGACIJOM) ---

# Globalna mapa za praćenje agregacije:
# {mol_name: {'ERa_HITC': 0/1, 'ERb_HITC': 0/1, 'mol_file_path': '...', 'global_features': {...}}}
# Inicijalne vrednosti 0.0 znače "nema potvrđene aktivnosti".
aggregation_map = {} 
final_molecular_data = {} # Finalni rečnik sa labelama 0, 1, 2

def map_to_new_3_class_label(era_hitc: float, erb_hitc: float) -> int:
    """
    Mapira agregirane HITC labele u 3 klase (0, 1, 2).
    Molekuli koji su selektivni za ERb (0, 1) se preskaču (vraćaju -1).
    """
    
    # 0: Potpuno Neaktivan (0, 0) -> Klasa 0
    if era_hitc == 0.0 and erb_hitc == 0.0:
        return 0 
    
    # 1: Selektivan ERa (1, 0) -> Klasa 1
    elif era_hitc == 1.0 and erb_hitc == 0.0:
        return 1
        
    # 2: Dualna aktivnost (1, 1) -> Klasa 2 (Menjamo oznaku iz 3 u 2)
    elif era_hitc == 1.0 and erb_hitc == 1.0:
        return 2
        
    # 3: Selektivan ERb (0, 1) - Ovu klasu izbacujemo iz skupa!
    elif era_hitc == 0.0 and erb_hitc == 1.0:
        return -1 # Oznaka za preskakanje
        
    # Za sve ostalo (npr. NaN, Unknown, etc.)
    return -1

def process_and_combine_molecular_data(
    csv_dataframes: dict[str, pd.DataFrame],
    mol_paths_by_csv_name: dict[str, dict[str, str]],
    column_mapping: dict, 
    common_label_name: str, 
    common_global_features_list: list[str]
) -> dict[str, dict]:
    global aggregation_map 
    global final_molecular_data
    
    aggregation_map = {} # Resetujemo mapu na početku
    final_molecular_data = {}
    
    total_processed_rows = 0
    total_skipped_rows = 0

    print("\nZapočinjem procesiranje i AGREGACIJU molekularnih podataka (4 klase)...")

    # --- FAZA 1: Agregacija po molekulu i receptoru (Prioritet 1.0 > 0.0) ---
    for csv_name, df in csv_dataframes.items():
        print(f"\nProcesiram podatke iz CSV: '{csv_name}'")
        
        assay_type = determine_assay_type(csv_name)
        if assay_type not in column_mapping:
            print(f"Upozorenje: Tip assay-a '{assay_type}' nije prepoznat. Preskačem.")
            total_skipped_rows += len(df)
            continue

        current_assay_col_names = column_mapping[assay_type]
        mol_id_col_source = current_assay_col_names['mol_id_col']
        hitc_col_source = current_assay_col_names[common_label_name] 

        df[mol_id_col_source] = df[mol_id_col_source].astype(str).apply(standardize_mol_name)
        
        df_indexed = df.set_index(mol_id_col_source).copy()
        current_mol_paths_map = mol_paths_by_csv_name.get(csv_name, {})
        
        if not current_mol_paths_map:
            total_skipped_rows += len(df)
            continue

        # --- EKSTRAKCIJA ESR TIPA IZ FAJL PUTANJE/IMENA CSV-a ---
        csv_name_lower = csv_name.lower()
        esr_type = None
        if 'era' in csv_name_lower or 'esr1' in csv_name_lower:
            esr_type = 'ERa'
        elif 'erb' in csv_name_lower or 'esr2' in csv_name_lower:
            esr_type = 'ERb'

        if esr_type is None:
            # Ako ne možemo odrediti tip receptora iz CSV imena, to je problem za 4 klase
            print(f"Greška: Nije moguće odrediti ER tip iz CSV imena '{csv_name}'. Preskačem.")
            total_skipped_rows += len(df)
            continue

        # Iteriramo kroz redove DataFrame-a
        for mol_name, row_data in df_indexed.iterrows():
            mol_file_path = current_mol_paths_map.get(mol_name)
            
            if mol_file_path is None or not os.path.exists(mol_file_path):
                total_skipped_rows += 1
                continue

            # Dohvatanje i standardizacija HITC vrednosti (0.0 ili 1.0)
            hitc_value = row_data.get(hitc_col_source)
            if pd.isna(hitc_value) or hitc_value == ' ':
                total_skipped_rows += 1
                continue
                
            if isinstance(hitc_value, str):
                hitc_value = 1.0 if hitc_value.lower() == 'active' or '1' in hitc_value else 0.0
            else:
                hitc_value = float(hitc_value)
            
            total_processed_rows += 1

            # Inicijalizacija molekula ako ga nema u aggregation_map
            if mol_name not in aggregation_map:
                # 4. Dohvatanje Globalnih karakteristika (ovo radimo samo jednom)
                global_features_values = {}
                for common_feat_name in common_global_features_list:
                    source_col = current_assay_col_names.get(common_feat_name, common_feat_name)
                    feature_val = row_data.get(source_col)
                    
                    if pd.isna(feature_val) or feature_val is None:
                        global_features_values[common_feat_name] = 'NaN'
                    else:
                        try:
                            global_features_values[common_feat_name] = float(feature_val)
                        except (ValueError, TypeError):
                            global_features_values[common_feat_name] = 'NaN'

                # Inicijalizacija sa neutralnim statusom i putanjom MOL fajla
                aggregation_map[mol_name] = {
                    'ERa_HITC': 0.0, 
                    'ERb_HITC': 0.0,
                    'mol_file_path': mol_file_path,
                    'global_features': global_features_values,
                    'source_assay_name': csv_name
                }
            
            # --- AGREGACIONA LOGIKA (Prioritet: 1.0 nad 0.0) ---
            if hitc_value == 1.0:
                if esr_type == 'ERa':
                    aggregation_map[mol_name]['ERa_HITC'] = 1.0
                elif esr_type == 'ERb':
                    aggregation_map[mol_name]['ERb_HITC'] = 1.0
            # Ako je hitc_value 0.0 (Neaktivan), ne radimo ništa, jer 0.0 ne nadjačava 1.0.
            # Ako molekul uđe samo kao 0.0, ostaće 0.0 u inicijalnoj mapi.
    
    
    # --- FAZA 2: Generisanje finalne 4-klasne labele ---
    print("\nZapočinjem generisanje finalne 3-klasne labele (0, 1, 2)...")
    total_classified_mols = 0
    total_skipped_erb_selective = 0
    
    for mol_name, data in aggregation_map.items():
        era_hitc = data['ERa_HITC']
        erb_hitc = data['ERb_HITC']

        final_label = map_to_new_3_class_label(era_hitc, erb_hitc)
        
        # Preskačemo molekule za koje je vraćena oznaka -1 (Selektivan ERb)
        if final_label == -1:
            total_skipped_erb_selective += 1
            continue
        
        # Svi molekuli su klasifikovani (jer su inicijalizovani sa 0.0)
        
        # Popunjavamo finalni rečnik
        final_molecular_data[mol_name] = {
            'mol_file_path': data['mol_file_path'],
            'preferredName': mol_name,
            'ERa_HITC': era_hitc, # Dodajemo za proveru
            'ERb_HITC': erb_hitc, # Dodajemo za proveru
            'label': float(final_label), # Finalna labela 0.0, 1.0, 2.0 ili 3.0
            'global_features': data['global_features'],
            'source_assay_name': data['source_assay_name'] 
        }
        total_classified_mols += 1
    
    print(f"\nProcesiranje završeno. Ukupno jedinstvenih molekula obrađeno: {total_classified_mols}.")
    print(f"Definisane klase (0=Neaktivan, 1=Selektivan ERa, 2=Selektivan ERb, 3=Dualna Aktivnost).")
    
    # Preimenovanje globalnog rečnika radi jasnoće
    return final_molecular_data

def analyze_class_distribution_3_class(all_molecules_processed_data: dict):
    """
    Analizira i ispisuje raspodelu 3-klasnih labela (0, 1, 2).
    """
    labels = [data['label'] for data in all_molecules_processed_data.values()]
    label_series = pd.Series(labels)
    
    class_counts = label_series.value_counts().sort_index()
    total_samples = len(label_series)
    class_percentages = (class_counts / total_samples) * 100

    print("\n" + "="*70)
    print("ANALIZA RASPODELE NOVIH 3-KLASNIH LABELA")
    print("======================================================================")
    
    # Mapiranje labela za bolji ispis
    label_map = {
        0.0: '0 - Potpuno Neaktivan (Baseline)',
        1.0: '1 - Selektivan ERa (ESR1)',
        2.0: '2 - Dualna Aktivnost (Oba)'
    }
    
    distribution_df = pd.DataFrame({
        'Broj jedinstvenih molekula': class_counts,
        'Procenat (%)': class_percentages.round(2)
    })
    
    distribution_df.index = distribution_df.index.map(lambda x: label_map.get(x, f'Klasa {x} (Nepoznato)'))
    
    print(f"Ukupno obrađenih jedinstvenih molekula: {total_samples}")
    print("\nRaspodela po klasi:")
    print(distribution_df)
    print("="*70)
    
    # ... (ostatak koda za analizu neuravnoteženosti)
    return distribution_df

COMMON_LABEL_NAME = 'esr_activity_class'
COMMON_GLOBAL_FEATURES_LIST = [ # Imena kolona za globalne karakteristike u finalnom setu podataka
     'monoisotopicMass', 'ac50', 
]

ACTUAL_COLUMN_NAMES_PER_ASSAY_TYPE = {
    'binders': {
        'mol_id_col': 'preferredName',  # Stvarni naziv kolone za ID molekula u 'binders' CSV-ovima
        COMMON_LABEL_NAME: 'hitc',       # Stvarni naziv kolone za 'hitc' u 'binders' CSV-ovima
        'averageMass':'averageMass',
        'monoisotopicMass': 'monoisotopicMass',
        'ac50':'ac50'
    },
    'tox21': {
        'mol_id_col': 'preferredName',  
        COMMON_LABEL_NAME: 'HIT CALL',       
        'monoisotopicMass': 'MONOISOTOPIC MASS',
        'ac50': 'AC50',
    }
}

# --- POMOĆNE FUNKCIJE ZA EKSTRAKCIJU GRAFOVSKIH FEATURE-A (IZ PRETHODNOG ODGOVORA) ---
# Moraju biti definisane pre nego što pozovete process_mols_for_gnn_dataset
def get_atom_features(atom):
    features = [
        atom.GetAtomicNum(),  # Atomic number
        atom.GetDegree(),     # Number of neighbors
        atom.GetTotalNumHs(), # Total number of hydrogens
        atom.GetImplicitValence(), # Implicit valence
        atom.GetFormalCharge(), # Formal charge
        atom.GetIsAromatic(), # Is aromatic
        atom.IsInRingSize(3), atom.IsInRingSize(4), atom.IsInRingSize(5), # In a ring of size 3,4,5
        atom.IsInRingSize(6), atom.IsInRingSize(7), atom.IsInRingSize(8), # In a ring of size 6,7,8
    ]
    return np.array(features, dtype=np.float32)

def get_bond_features(bond):
    features = [
        bond.GetBondTypeAsDouble(), # Bond type (1.0 for single, 2.0 for double, etc.)
        bond.GetIsConjugated(),     # Is conjugated
        bond.IsInRing(),            # Is in a ring
    ]
    return np.array(features, dtype=np.float32)


def process_mols_for_gnn_dataset(all_molecules_processed_data: dict) -> list[dict]:
    dataset_ready_molecules = []
    skipped_count = 0

    print("\n--- Započinjem generisanje grafovskih reprezentacija za TF Dataset ---")
    total_mols = len(all_molecules_processed_data)

    for i, (mol_name, data) in enumerate(all_molecules_processed_data.items()):
        if i % 100 == 0: # Ispis napretka svakih 100 molekula
            print(f"  Procesiram molekul {i+1}/{total_mols}: {mol_name}")

        mol_file_path = data['mol_file_path']

        # --- Učitavanje MOL fajla i generisanje grafovskih feature-a ---
        try:
            mol = Chem.MolFromMolFile(mol_file_path)
            if mol is None:
                print(f"    Upozorenje: RDKit nije uspeo da učita MOL fajl '{mol_file_path}' za molekul '{mol_name}'. Preskačem.")
                skipped_count += 1
                continue

            # Node Features (Atomske karakteristike)
            node_features = []
            for atom in mol.GetAtoms():
                node_features.append(get_atom_features(atom))
            
            if not node_features: # Molekul bez atoma? Vrlo retko, ali sigurnosti radi.
                print(f"    Upozorenje: Molekul '{mol_name}' nema atoma. Preskačem.")
                skipped_count += 1
                continue
            
            node_features = np.array(node_features, dtype=np.float32)

            # Edge Features (Karakteristike veza) i Adjacency Matrix
            edge_features = []
            adjacency_indices = [] 
            adjacency_values = [] 

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                bond_feat = get_bond_features(bond)

                edge_features.append(bond_feat) # (i,j)
                adjacency_indices.append([i, j])
                adjacency_values.append(1.0) # Vrednost veze

                edge_features.append(bond_feat) # (j,i) - Za neusmerene grafove
                adjacency_indices.append([j, i])
                adjacency_values.append(1.0)
            
            # Ako nema veza (npr. monoatomski molekul), osiguraj prazne array-e sa ispravnim dimenzijama
            if not adjacency_indices:
                # Dimenzija za edge_features bi trebalo da odgovara get_bond_features izlazu
                # Moramo pozvati get_bond_features na dummy bond da dobijemo shape,
                # ili ga hardkodirati ako znamo dimenziju (npr. 3 u ovom primeru)
                dummy_bond_feat_dim = get_bond_features(Chem.Bond()).shape[0] if Chem.Mol().GetNumBonds() == 0 else 3 
                                    # Chem.Bond() ne postoji, ali je trik za dobijanje dimenzija
                                    # Bolje je da se dimenzija node/edge feature-a definiše globalno
                                    # ili da se pretpostavi npr. 3 (ako get_bond_features uvek vraća 3)
                
                adjacency_indices = np.empty((0, 2), dtype=np.int64)
                adjacency_values = np.empty((0,), dtype=np.float32)
                edge_features = np.empty((0, dummy_bond_feat_dim), dtype=np.float32)
                num_edges = 0
            else:
                adjacency_indices = np.array(adjacency_indices, dtype=np.int64)
                adjacency_values = np.array(adjacency_values, dtype=np.float32)
                edge_features = np.array(edge_features, dtype=np.float32)
                num_edges = adjacency_indices.shape[0] # Broj veza (dupliranih)

            # --- Pakovanje podataka ---
            processed_data = {
                'preferredName': data['preferredName'],
                'esr': data.get('esr', 'unknown'), # Sa .get() je sigurnije
                'label': data['label'],
                'global_features': data['global_features'],
                'node_features': node_features,
                'edge_features': edge_features,
                'adjacency_indices': adjacency_indices,
                'adjacency_values': adjacency_values,
                'num_nodes': node_features.shape[0],
                'num_edges': num_edges,
                'source_assay_name': data['source_assay_name']
            }
            dataset_ready_molecules.append(processed_data)

        except Exception as e:
            print(f"    Greška pri generisanju grafovskih feature-a za '{mol_name}' iz '{mol_file_path}': {e}. Preskačem molekul.")
            skipped_count += 1
            continue
            
    print(f"\n--- Generisanje grafovskih reprezentacija završeno. Ukupno obrađenih: {len(dataset_ready_molecules)}, Preskočeno: {skipped_count}. ---")
    return dataset_ready_molecules


# --- NEW: Helper to extract base graph features from a MOL file (returns NumPy arrays) ---
# This is essentially the core logic extracted from your original process_mols_for_gnn_dataset
# and _generate_graph_features_for_mol that deals with RDKit.
def _extract_base_graph_features_from_mol_file(mol_file_path: str, mol_name: str):
    mol = Chem.MolFromMolFile(mol_file_path)
    
    # 1. KRITIČNA KOREKCIJA: RDKit Sanacija (prevencija grešaka)
    from rdkit.Chem.rdmolops import SanitizeMol, SanitizeFlags
    if mol is None:
        raise ValueError(f"RDKit failed to load MOL file: {mol_file_path}")
    try:
        SanitizeMol(mol, SanitizeFlags.SANITIZE_ALL) 
        mol.UpdatePropertyCache(strict=False)
    except Exception as san_e:
        raise ValueError(f"RDKit Sanacija neuspešna za '{mol_name}': {san_e}") from san_e

    # Node Features
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(get_atom_features(atom))
    node_features = np.array(node_features, dtype=np.float32)

    if node_features.shape[0] == 0:
         raise ValueError(f"Molecule '{mol_name}' has no atoms.")

    # Edge Features i Adjacency
    edge_features = [] # <--- DODATO: Inicijalizacija liste edge feature-a
    adjacency_indices = []
    adjacency_values = []
    
    for bond in mol.GetBonds():
        bond_feat = get_bond_features(bond) # <--- POZIV ZA EKSTRAKCIJU EDGE FEATURE-A
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Smer (i, j)
        adjacency_indices.append([i, j])
        adjacency_values.append(1.0)
        edge_features.append(bond_feat) # <--- DODATO: Pakovanje edge feature-a
        
        # Smer (j, i) - Za neusmeren graf
        adjacency_indices.append([j, i])
        adjacency_values.append(1.0)
        edge_features.append(bond_feat) # <--- DODATO: Pakovanje edge feature-a

    if not adjacency_indices:
        # Fallback za molekule bez veza
        dummy_bond_feat_dim = 3
        
        # Moramo inicijalizovati edge_features čak i ako je prazan!
        edge_features = np.empty((0, dummy_bond_feat_dim), dtype=np.float32) # <--- DODATO: Prazan array
        
        adjacency_indices = np.empty((0, 2), dtype=np.int64)
        adjacency_values = np.empty((0,), dtype=np.float32)
        num_edges = 0
    else:
        adjacency_indices = np.array(adjacency_indices, dtype=np.int64)
        adjacency_values = np.array(adjacency_values, dtype=np.float32)
        edge_features = np.array(edge_features, dtype=np.float32) 
        num_edges = adjacency_indices.shape[0]

    return {
        'node_features': node_features,
        'edge_features': edge_features, 
        'adjacency_indices': adjacency_indices,
        'adjacency_values': adjacency_values,
        'num_nodes': node_features.shape[0],
        'num_edges': num_edges,
    }


# --- NEW: Function to apply augmentation to the entire dataset (REVISED for Class 2) ---
def augment_molecular_data(
    all_molecules_processed_data: dict,
    target_label: float = 2.0, # NOVO: Ciljamo Klasu 2 (Dualna Aktivnost)
    num_augmentations_per_sample: int = 4 # Cilj: 5 dodatnih uzoraka po originalnom
) -> dict[str, dict]:
    """
    Applies node permutation augmentation to molecular graph data.
    Returns a new dictionary containing original and augmented molecules.
    
    Logika: Samo uzorci sa target_label dobijaju N augmentacija.
    Svi uzorci dobijaju ekstrakciju grafovskih feature-a.
    """
    augmented_data = {}
    total_original_mols = len(all_molecules_processed_data)
    processed_count = 0

    print(f"\n--- Starting data augmentation (node permutation) for {total_original_mols} molecules ---")
    print(f"Cilj: Klasa {target_label} ({num_augmentations_per_sample}x augmentacija)")

    for mol_name, data in all_molecules_processed_data.items():
        processed_count += 1
        if processed_count % 300 == 0: # Smanjio sam ispis da bude manje verovatno
            print(f"  Augmenting molecule {processed_count}/{total_original_mols}: {mol_name}")

        try:
            # 1. Ekstrakcija baznih grafovskih feature-a (MORA se desiti za SVE molekule)
            base_graph_features = _extract_base_graph_features_from_mol_file(data['mol_file_path'], mol_name)
            
            # Kombinovanje svih feature-a za ORIGINALNI molekul
            original_entry = {
                **data, # Kopirajte sve postojeće metapodatke
                **base_graph_features, # Dodajte ekstrahovane grafovske feature-e (NumPy nizove)
            }
            # Dodavanje ORIGINALNOG molekula
            augmented_data[mol_name] = original_entry

            # 2. Provera da li treba primeniti augmentaciju
            should_augment = (data['label'] == target_label)

            if should_augment and base_graph_features['num_nodes'] > 0:
                #print(f"Labela trenutnog: {data['label']}")
            
                for aug_idx in range(num_augmentations_per_sample):
                    
                    # Logika permutacije je ispravna, koristićemo je:
                    permutation = np.random.permutation(base_graph_features['num_nodes'])
                    
                    # Permutacija node features
                    x_permuted = base_graph_features['node_features'][permutation, :]

                    # Permutacija adjacency indeksa
                    a_indices_permuted = np.copy(base_graph_features['adjacency_indices'])
                    
                    if a_indices_permuted.size > 0:
                        old_to_new_idx_map = {old_idx: new_idx for old_idx, new_idx in enumerate(permutation)}
                        a_indices_permuted[:, 0] = np.vectorize(old_to_new_idx_map.get)(a_indices_permuted[:, 0])
                        a_indices_permuted[:, 1] = np.vectorize(old_to_new_idx_map.get)(a_indices_permuted[:, 1])
                    
                    # 3. Kreiranje novog unosa za augmentovani molekul
                    augmented_mol_name = f"{mol_name}_aug_perm{aug_idx+1}"
                    augmented_entry = {
                        **data, 
                        'preferredName': augmented_mol_name, 
                        'node_features': x_permuted,
                        'edge_features': base_graph_features['edge_features'],
                        'adjacency_indices': a_indices_permuted,
                        'adjacency_values': base_graph_features['adjacency_values'],
                        'num_nodes': base_graph_features['num_nodes'],
                        'num_edges': base_graph_features['num_edges'],
                    }
                    augmented_data[augmented_mol_name] = augmented_entry
                    #return
        except Exception as e:
            print(f"    Warning: Skipping molecule '{mol_name}' due to error during feature extraction or augmentation: {e}")
            continue

    print(f"\n--- Data augmentation finished. Total molecules (original + augmented): {len(augmented_data)} ---")
    return augmented_data


# Ova funkcija uzima sirove features (NumPy/Python) i pretvara ih u TF tenzore.
def _generate_graph_features_for_mol(
    mol_name: str, 
    mol_data: dict, 
    common_global_features_list: list[str],
    include_global_features: bool = True # NOVI ARGUMENT
):
    try:
        # LOGIKA ZA GLOBALNE KARAKTERISTIKE I ABLACIJU
        global_features_data = mol_data['global_features']
        
        if include_global_features:
            feature_values = []
            for key in common_global_features_list:
                value = global_features_data.get(key)
                if value == 'NaN' or value is None:
                    feature_values.append(0.0) 
                elif isinstance(value, str):
                     try:
                         feature_values.append(float(value))
                     except ValueError:
                         feature_values.append(0.0)
                else:
                    feature_values.append(value)

            global_features_np = np.array(feature_values, dtype=np.float32)
            final_global_features = tf.constant(global_features_np, dtype=tf.float32)
        else:
            # Scenarij Ablacije: Prazan tenzor
            final_global_features = tf.constant(np.array([]), dtype=tf.float32)
        
        # PAKOVANJE SVIH FEATURE-A I KONVERZIJA U TF Tenzore
        # Pristup mol_data['edge_features'] je ovde zaštićen sa try/except blokom.
        return {
            'preferredName': tf.constant(mol_data['preferredName'], dtype=tf.string),
            'esr': tf.constant(mol_data.get('esr', 'unknown'), dtype=tf.string),
            'label': mol_data['label'], # Labela je float, konvertujemo je kasnije u generator_fn
            'global_features': final_global_features,
            'node_features': tf.constant(mol_data['node_features'], dtype=tf.float32),
            'edge_features': np.empty((0, 3), dtype=np.float32), 
            'adjacency_indices': tf.constant(mol_data['adjacency_indices'], dtype=tf.int64),
            'adjacency_values': tf.constant(mol_data['adjacency_values'], dtype=tf.float32),
            'num_nodes': tf.constant(mol_data['num_nodes'], dtype=tf.int32),
            'num_edges': tf.constant(mol_data['num_edges'], dtype=tf.int32),
            'source_assay_name': tf.constant(mol_data['source_assay_name'], dtype=tf.string)
        }
    except Exception as e:
        # Kljucni deo: Ako se desi KeyError (kao 'edge_features') ili neka druga greska, vracamo None.
        print(f"Greška pri generisanju TF feature-a za '{mol_name}': {e}. Preskačem.")
        return None
    
def create_tf_dataset_from_ids(
    mol_ids: list[str],
    all_molecules_processed_data: dict,
    common_global_features_list: list[str],
    output_signature: tuple,
    include_global_features: bool = True # DODAT ARGUMENT ZA KONTROLU
) -> tf.data.Dataset:
    
    def generator_fn():
        for mol_id in mol_ids:
            mol_data = all_molecules_processed_data.get(mol_id)
            
            # KRITIČNA KOREKCIJA: Prva provera da li je mol_data pronađen
            if mol_data is None:
                print(f"Upozorenje: Podaci za mol_id '{mol_id}' nisu pronađeni. Preskačem.")
                continue

            # POZIVANJE POMOĆNE FUNKCIJE SA NOVIM ARGUMENTOM
            processed_mol = _generate_graph_features_for_mol(
                mol_id, 
                mol_data, 
                common_global_features_list,
                include_global_features # Prosleđivanje argumenta za ablaciju
            )
            
            # Ova provera hvata 'None' koji je vraćen zbog KeyError: 'edge_features'
            if processed_mol is None:
                continue

            # YIELD KOD (kao u Vašoj staroj funkciji koja je radila, ali sa TF tenzorima iz processed_mol)
            yield {
                'node_features': processed_mol['node_features'],
                'edge_features': processed_mol['edge_features'],
                'adjacency_indices': processed_mol['adjacency_indices'],
                'adjacency_values': processed_mol['adjacency_values'],
                'global_features': processed_mol['global_features'],
                'num_nodes': processed_mol['num_nodes'],
                'num_edges': processed_mol['num_edges'],
                'preferredName': processed_mol['preferredName'],
                'esr': processed_mol['esr'],
                'source_assay_name': processed_mol['source_assay_name']
            }, tf.constant(processed_mol['label'], dtype=tf.float32) # Labela se konvertuje u TF ovde

    return tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=output_signature
    )

def split_data_and_create_tf_datasets_by_id(
    all_molecules_processed_data: dict,
    test_split_ratio: float = 0.2,
    val_split_ratio: float = 0.2,
    random_seed: int = 42,
    common_global_features_list: list[str] = None,
    # === NOVO ZA ABLACIJU ===
    include_global_features: bool = True 
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    
    if not all_molecules_processed_data:
        print("Upozorenje: Prazan ulazni rečnik podataka. Vraćam prazne datasete.")
        # ... (Vraćanje praznih dataseta) ...
        return tf.data.Dataset.from_tensor_slices({}), tf.data.Dataset.from_tensor_slices({}), tf.data.Dataset.from_tensor_slices({})

    mol_ids = list(all_molecules_processed_data.keys())
    labels = np.array([all_molecules_processed_data[mol_id]['label'] for mol_id in mol_ids])

    # Ostatak logike podele (train_val_ids, test_ids, train_ids, val_ids) OSTAJE ISTI
    # ... (kod za train_test_split se ne menja, jer uvek delimo ceo skup) ...
    
    train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
        mol_ids, labels,
        test_size=test_split_ratio,
        stratify=labels,
        random_state=random_seed
    )
    print(f"\nUkupno podataka: {len(mol_ids)}")
    print(f"Podela na Test ({test_split_ratio*100}%): {len(test_ids)} uzoraka.")
    print(f"Preostalo za Train+Val ({(1-test_split_ratio)*100}%): {len(train_val_ids)} uzoraka.")

    train_ids, val_ids, train_labels, val_labels = train_test_split(
        train_val_ids, train_val_labels,
        test_size=val_split_ratio,
        stratify=train_val_labels,
        random_state=random_seed
    )
    print(f"Podela Train+Val: Train ({len(train_ids)} uzoraka), Val ({len(val_ids)} uzoraka).")
    print(f"Finalni odnosi: Train: {len(train_ids)/len(mol_ids):.2f}, "
          f"Val: {len(val_ids)/len(mol_ids):.2f}, "
          f"Test: {len(test_ids)/len(mol_ids):.2f}")

    # ... (Određivanje dimenzija dummy_atom_dim i dummy_bond_dim ostaje isto)
    try:
        dummy_atom_dim = get_atom_features(Chem.Atom(0)).shape[0]
    except Exception:
        print("Upozorenje: Nije moguće dobiti dimenziju atomskih feature-a iz Chem.Atom(0). Pretpostavljam 12.")
        dummy_atom_dim = 12

    try:
        dummy_mol_for_bond_dim = Chem.MolFromSmiles('CC')
        if dummy_mol_for_bond_dim and dummy_mol_for_bond_dim.GetNumBonds() > 0:
            dummy_bond_dim = get_bond_features(dummy_mol_for_bond_dim.GetBonds()[0]).shape[0]
        else:
            raise ValueError("Could not get bond feature dimension from dummy molecule.")
    except Exception:
        print("Upozorenje: Nije moguće dobiti dimenziju veza feature-a iz Chem.Bond() ili dummy molekula. Pretpostavljam 3.")
        dummy_bond_dim = 3


    # === LOGIKA ZA GLOBALNE KARAKTERISTIKE ===
    if include_global_features:
        if common_global_features_list is None or not common_global_features_list:
            if mol_ids:
                first_mol_id = mol_ids[0]
                first_mol_data = all_molecules_processed_data[first_mol_id]
                global_features_dim = len(first_mol_data['global_features'])
                print(f"Determinisana dimenzija globalnih karakteristika: {global_features_dim}")
            else:
                global_features_dim = 0
                print("Upozorenje: Nema podataka za određivanje dimenzije globalnih karakteristika. Postavljeno na 0.")
        else:
            global_features_dim = len(common_global_features_list)
    else:
        # Ako ne uključujemo globalne feature-e, dimenzija je 0.
        global_features_dim = 0
        print("Ablaciona Studija: Globalne karakteristike su ISKLJUČENE (dimenzija: 0).")


    # === KREIRANJE OUTPUT SIGNATURE ===
    output_signature = (
        {
            'node_features': tf.TensorSpec(shape=(None, dummy_atom_dim), dtype=tf.float32),
            'edge_features': tf.TensorSpec(shape=(None, dummy_bond_dim), dtype=tf.float32),
            'adjacency_indices': tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
            'adjacency_values': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            # NOVO: Dinamički shape (0 ili stvarna dimenzija)
            'global_features': tf.TensorSpec(shape=(global_features_dim,), dtype=tf.float32), 
            'num_nodes': tf.TensorSpec(shape=(), dtype=tf.int32),
            'num_edges': tf.TensorSpec(shape=(), dtype=tf.int32),
            'preferredName': tf.TensorSpec(shape=(), dtype=tf.string),
            'esr': tf.TensorSpec(shape=(), dtype=tf.string),
            'source_assay_name': tf.TensorSpec(shape=(), dtype=tf.string)
        },
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )

    # === Kreiranje Datasets-a (proširujemo i ovde parametar) ===
    train_tf_dataset = create_tf_dataset_from_ids(
        train_ids, all_molecules_processed_data, common_global_features_list, output_signature, include_global_features
    )
    val_tf_dataset = create_tf_dataset_from_ids(
        val_ids, all_molecules_processed_data, common_global_features_list, output_signature, include_global_features
    )
    test_tf_dataset = create_tf_dataset_from_ids(
        test_ids, all_molecules_processed_data, common_global_features_list, output_signature, include_global_features
    )

    return train_tf_dataset, val_tf_dataset, test_tf_dataset

# --- Funkcija za čuvanje datasetova ---
def save_tf_datasets(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    output_directory: str = 'processed_tf_datasets'
):
    """
    Čuva trening, validacioni i test TensorFlow Datasetove u navedeni direktorijum.

    Args:
        train_ds (tf.data.Dataset): Trening dataset.
        val_ds (tf.data.Dataset): Validacioni dataset.
        test_ds (tf.data.Dataset): Test dataset.
        output_directory (str): Putanja do direktorijuma gde će se datasetovi sačuvati.
    """
    os.makedirs(output_directory, exist_ok=True)

    train_path = os.path.join(output_directory, 'train_dataset')
    val_path = os.path.join(output_directory, 'val_dataset')
    test_path = os.path.join(output_directory, 'test_dataset')

    print(f"\n--- Počinjem čuvanje TF Datasetova u {output_directory} ---")
    
    try:
        train_ds.save(train_path)
        print(f"Trening dataset sačuvan na: {train_path}")
    except Exception as e:
        print(f"Greška pri čuvanju trening dataseta: {e}")

    try:
        val_ds.save(val_path)
        print(f"Validacioni dataset sačuvan na: {val_path}")
    except Exception as e:
        print(f"Greška pri čuvanju validacionog dataseta: {e}")

    try:
        test_ds.save(test_path)
        print(f"Test dataset sačuvan na: {test_path}")
    except Exception as e:
        print(f"Greška pri čuvanju test dataseta: {e}")

    print("\n--- Čuvanje TF Datasetova završeno ---")


if __name__ == "__main__":
    test_base_dir = '/home/ivana-milutinovic/Documents/Doktorske/BIORad/GitHub/BIO-info-multiclass/data-processing/data'
    csv_dir = os.path.join(test_base_dir, 'binders_and_aa')
    mol_dir = os.path.join(test_base_dir, 'mols_from_original_csv')
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(mol_dir, exist_ok=True)

    print("--- 1. Učitavanje CSV i MOL putanja ---")
    csv_dataframes_map, mol_paths_map = load_all_data_into_maps(csv_dir, mol_dir)

    # 2. Process and combine molecular data (AGREGACIJA i kreiranje 4 klase)
    # Rezultat je rečnik sa JEDINSTVENIM molekulima i finalnom labelom (0, 1, 2, ili 3)
    print("\n--- 2. Procesiranje, Agregacija i Kreiranje 4-Klasne Labele ---")
    
    all_molecules_metadata = process_and_combine_molecular_data(
        csv_dataframes_map,
        mol_paths_map,
        ACTUAL_COLUMN_NAMES_PER_ASSAY_TYPE,
        COMMON_LABEL_NAME,
        COMMON_GLOBAL_FEATURES_LIST
    )
    
    # 3. ANALIZA RASPODELE KLASA (NOVI KORAK)
    print("\n--- 3. Analiza raspodele 3-klasne labele (Bez Augmentacije) ---")
    if all_molecules_metadata:
        class_report = analyze_class_distribution_3_class(all_molecules_metadata)
    else:
        print("Greška: Nema obrađenih molekula za analizu.")
    
    # =========================================================================
    # 4. PRIMENA AUGMENTACIJE
    # =========================================================================
    print("\n--- 4. PRIMENA CILJANE AUGMENTACIJE (Klasa 2) ---")
    # Ciljamo Klasu 2 (Dualna Aktivnost), 5 puta augmentacija (5 novih uzoraka po originalnom)
    augmented_all_molecules_data = augment_molecular_data(
        all_molecules_metadata,
        target_label=2.0,      # NOVO: Ciljamo Dualnu Aktivnost (Klasa 2)
        num_augmentations_per_sample=4
    )

    # 5. ANALIZA RASPODELE KLASA NAKON AUGMENTACIJE
    print("\n--- 5. Analiza raspodele NAKON Augmentacije ---")
    if augmented_all_molecules_data:
        class_report_augmented = analyze_class_distribution_3_class(augmented_all_molecules_data)
    else:
        print("Greška: Nema augmentiranih molekula za analizu.")

    
    # SCENARIJ A: PUNI MODEL (Grafovi + Globalni Feature-i)
    print("\n--- KREIRANJE DATASETA: PUNI MODEL (Grafovi + Globalni Feature-i) ---")
    train_full, val_full, test_full = split_data_and_create_tf_datasets_by_id(
    augmented_all_molecules_data, 
    test_split_ratio=0.2, val_split_ratio=0.25,
    common_global_features_list=COMMON_GLOBAL_FEATURES_LIST,
    include_global_features=True 
    )
    out_dir = '/home/ivana-milutinovic/Documents/Doktorske/BIORad/GitHub/BIO-info-multiclass/data-processing/dataset_for_multi/datasets/'
    save_tf_datasets(train_full, val_full, test_full, output_directory=f'{out_dir}tfrecords_full')

    
    # SCENARIJ B: ABLACIJA GLOBALNIH (Samo Grafovi)
    print("\n--- KREIRANJE DATASETA: ABLACIJA GLOBALNIH (Samo Grafovi) ---")
    train_graph_only, val_graph_only, test_graph_only = split_data_and_create_tf_datasets_by_id(
        augmented_all_molecules_data, 
        test_split_ratio=0.2, val_split_ratio=0.25,
        common_global_features_list=COMMON_GLOBAL_FEATURES_LIST,
        include_global_features=False # KLJUČNA PROMENA
    )
    save_tf_datasets(train_graph_only, val_graph_only, test_graph_only, output_directory=f'{out_dir}tfrecords_graph_only')

    print("\nUspešno kreirana DVA SKUPA TFRecords-a za Ablacionu Studiju. Spremni ste za modelovanje! 🚀")
    #"""