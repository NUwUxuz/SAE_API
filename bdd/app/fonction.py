# import

import csv
import os
import re
import random
import hashlib
import pandas as pd
import ast
import csv
from datetime import datetime
from bs4 import BeautifulSoup
from pathlib import Path

# --- Fonction utilitaire ---
def ensure_output_dir(path="./output/"):
    Path(path).mkdir(parents=True,exist_ok=True)

def ensure_output_csv(path="./output.csv"):
    """Create the output CSV file if it does not already exist."""
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).touch(exist_ok=True)

def suppr_fichier(path:str):
    if os.path.isfile(path):
        try:
            os.remove(path)
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la suppression de '{path}' : {e}")

# --- Nettoyage : suppression des lignes vides ---
def remove_empty_rows(input_csv:str, output_csv:str, column_name:str):
    # ensure_output_dir()
    ensure_output_csv(output_csv)
    with open(input_csv, 'r', newline='', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if column_name in row and row[column_name].strip():
                writer.writerow(row)

# --- Filtrage des lignes selon une valeur ---

def filter(input_csv:str, output_csv:str, column_name:str, valeurs: list,replace_by : str=''):
    """
    Parcourt un CSV et vide les cellules de la colonne 'column_name'
    si leur valeur figure dans 'valeurs' (insensible à la casse et aux espaces).
    Les lignes ne sont pas supprimées.
    """
    # ensure_output_dir()
    
    ensure_output_csv(output_csv)

    with open(input_csv, 'r', newline='', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()

        valeurs_normalisées = [v.strip().lower() for v in valeurs]

        for row in reader:
            if column_name in row:
                cell_value = row[column_name].strip().lower()
                if cell_value in valeurs_normalisées:
                    # On vide la cellule si la valeur est à exclure
                    row[column_name] = replace_by
            writer.writerow(row)


# --- Suppression de colonnes ---
def remove_columns(input_csv:str, output_csv:str, columns_to_remove:dict):
    ensure_output_dir()
    ensure_output_csv(output_csv)
    with open(input_csv, 'r', newline='', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        columns_to_keep = [c for c in reader.fieldnames if c not in columns_to_remove]
        writer = csv.DictWriter(outfile, fieldnames=columns_to_keep)
        writer.writeheader()
        for row in reader:
            writer.writerow({c: row[c] for c in columns_to_keep})


# --- Conservation de colonnes spécifiques ---
def keep_columns(input_csv:str, output_csv:str, columns_to_keep:dict):
    # ensure_output_dir()
    ensure_output_csv(output_csv)
    with open(input_csv, 'r', newline='', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        valid_columns = [c for c in columns_to_keep if c in reader.fieldnames]
        writer = csv.DictWriter(outfile, fieldnames=valid_columns)
        writer.writeheader()

        single_column = len(valid_columns) == 1
        col_name = valid_columns[0] if single_column else None

        for row in reader:
            if single_column:
                
                value = row[col_name].strip()
                
                
                writer.writerow({col_name: value})
               
            else:
                writer.writerow({c: row[c] for c in valid_columns})


def explode_single_column_csv_to_ids(input_csv, output_csv, mapping_dict=None, delimiter=','):
    """
    Transforme un CSV à une colonne en un CSV plat (user, id_valeur).
    - mapping_dict (optionnel) : dict {valeur: id}
    - Si une valeur n'existe pas dans le dict, un nouvel id est créé automatiquement.
    - Retourne le dictionnaire final mis à jour.
    """
    
    ensure_output_csv(output_csv)
    # Initialiser le dictionnaire si absent
    if mapping_dict is None:
        dict = {}
    else :
        dict = mapping_dict
    # Calculer le prochain ID disponible
    next_id = max(dict.values(), default=0) + 1

    with open(input_csv, 'r', newline='', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)

        # Normaliser le nom de la seule colonne
        fieldnames = [f.strip().strip('"').strip("'") for f in reader.fieldnames]
        if len(fieldnames) != 1:
            raise ValueError(f"Le fichier doit contenir une seule colonne, trouvé : {fieldnames}")

        column_name = fieldnames[0]
        print(f"[DEBUG] Colonne détectée : {column_name!r}")

        with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
            fieldnames_out = ['id_user', f'id_{column_name}']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames_out)
            writer.writeheader()

            for user_id, row in enumerate(reader, start=1):
                # Trouver la clé réelle dans le dictionnaire
                for key in row.keys():
                    if key.strip().strip('"').strip("'") == column_name:
                        cell_value = row[key]
                        break
                else:
                    raise KeyError(f"Impossible de trouver la colonne {column_name} dans la ligne {row}")

                if not cell_value:
                    continue

                # Nettoyer la cellule
                cell_value = cell_value.strip().strip('"').strip("'")
                items = [item.strip() for item in cell_value.split(delimiter) if item.strip()]

                for item in items:
                    # Vérifier si la valeur a déjà un ID
                    if item not in dict:
                        dict[item] = next_id
                        next_id += 1
                        print(f"[DEBUG] Nouvelle valeur mappée : {item!r} -> ID {dict[item]}")
                    writer.writerow({'id_user': user_id, f'id_{column_name}': dict[item]})
                # print(f'//id_{column_name}//')

    return dict



def safe_filename(name: str) -> str:
    """Nettoie un nom de fichier en remplaçant les caractères interdits."""
    # Supprime ou remplace les caractères interdits
    name = name.strip().replace(' ', '_')
    # Enlève tous les caractères non alphanumériques sauf `_` et `-`
    name = re.sub(r'[^A-Za-z0-9_\-]', '', name)
    return name





def donne_table_association(input: str, colonne: str,
                            mapping_dict_: dict = None,
                            only_map: bool = False):
    """
    Génère les tables d'association et renvoie un mapping final
    pour la colonne choisie dans un CSV.
    """
    ensure_output_dir("./relation/")
    

    col_nom = safe_filename(str(colonne))

    # Sécurisation des dictionnaires
    mapping_full_input = mapping_dict_.copy() if mapping_dict_ else {}
    mapping_final_input = mapping_dict_.copy() if mapping_dict_ else {}

    # print("Clés actuelles du mapping :", mapping_final_input.keys())

    # Fichiers temporaires
    tmp_file = Path(f"./relation/{col_nom}_temp.csv")
    filtered_file = Path(f"./relation/{col_nom}.csv")
    id_file = Path(f"./relation/{col_nom}_id.csv")
    id_filtered_file = Path(f"./relation/{col_nom}_id_final.csv")
    ensure_output_csv(tmp_file)
    ensure_output_csv(filtered_file)
    ensure_output_csv(id_file)
    ensure_output_csv(id_filtered_file)


    nom_colonne_id = f"id_{colonne}"
    nom_colonne_id = nom_colonne_id.strip().strip('"').strip("'")
    # Étape 1 : extraction de la colonne
    keep_columns(input, tmp_file, [colonne])

    # Étape 2 : filtrage des valeurs non désirées
    filter(tmp_file, filtered_file, colonne, ["Ne souhaite pas répondre"])

    # Étape 3 : on supprime le fichier temporaire si présent
    if tmp_file.exists():
        tmp_file.unlink()

    # Étape 4 : génération du mapping (ID ↔ valeurs)
    mapping_full = explode_single_column_csv_to_ids(
        filtered_file,
        id_file,
        mapping_full_input
    )

    # Mode only_map → filtrage sur les clés du dictionnaire fourni
    if only_map:
        # print("[INFO] Mode only_map activé")

        allowed_keys = set(mapping_final_input.keys())
        # print("Clés autorisées :", allowed_keys)

        # valeurs à retirer
        to_remove = [key for key in mapping_full.keys() if key not in allowed_keys]
        # print("Valeurs supprimées :", to_remove)
        to_remove_id =[str(mapping_full[key]) for key in to_remove]
        print(to_remove_id)
        # Filtrage du fichier d’IDs
        print(f"//{nom_colonne_id}//")
        filter(id_file, id_filtered_file, nom_colonne_id, to_remove_id)

        # Suppression du fichier intermédiaire
        # if id_file.exists():
        #     id_file.unlink()

    return mapping_full


def normalize(s):
    return str(s).strip().casefold()


def remapper_colonne_csv(input_csv, output_csv, colonne, mapping_dict, sep=",", nouvelle_colonne=None):
    """
    Remappe une colonne selon un dictionnaire + gère les multi-valeurs.
    """
    ensure_output_csv(output_csv)
    # Lecture CSV
    with open(input_csv, newline='', encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)

    if colonne not in reader.fieldnames:
        raise ValueError(f"La colonne '{colonne}' n'existe pas dans le CSV.")

    # Nom final de la colonne
    col_finale = nouvelle_colonne if nouvelle_colonne else colonne

    # -------------------------
    # 1) TRAITEMENT DES LIGNES
    # -------------------------
    for row in rows:
        valeur = row[colonne]

        if not valeur or valeur.strip() == "":
            row[col_finale] = ""
            continue

        # Essayer liste Python
        try:
            parsed = ast.literal_eval(valeur)
            if isinstance(parsed, list):
                items = [str(v).strip() for v in parsed]
            else:
                items = [str(parsed).strip()]
        except Exception:
            # Sinon split
            items = [v.strip() for v in valeur.split(sep)]

        # Mapping
        mapped_items = []
        for item in items:
            key_norm = normalize(item)
            mapped = mapping_dict.get(key_norm, mapping_dict.get(item, item))
            mapped_items.append(str(mapped))

        row[col_finale] = sep.join(mapped_items)

    # -------------------------
    # 2) SUPPRESSION ANCIENNE COLONNE
    # -------------------------
    if nouvelle_colonne:
        for row in rows:
            if colonne in row:
                del row[colonne]

    # -------------------------
    # 3) MISE À JOUR DES FIELDNAMES
    # -------------------------
    if nouvelle_colonne:
        fieldnames = [
            nouvelle_colonne if fn == colonne else fn
            for fn in reader.fieldnames
        ]
    else:
        fieldnames = reader.fieldnames

    # -------------------------
    # 4) ÉCRITURE CSV
    # -------------------------
    with open(output_csv, "w", newline='', encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def jointure_csv(csv1, csv2, col1, col2, output_csv):
    """
    Réalise une jointure interne (INNER JOIN) entre deux CSV.

    - csv1 : chemin du premier CSV
    - csv2 : chemin du second CSV
    - col1 : nom de la colonne du csv1 utilisée pour la jointure
    - col2 : nom de la colonne du csv2 utilisée pour la jointure
    - output_csv : nom du CSV final
    
    La fonction produit un CSV fusionné contenant :
        - toutes les colonnes du CSV1
        - toutes les colonnes du CSV2 (sauf la clé 2 qui serait redondante)
    """
    ensure_output_csv(output_csv)
    # --- Lecture du CSV1 ---

    with open(csv1, newline='', encoding="utf-8") as f1:
        r1 = csv.DictReader(f1)
        rows1 = list(r1)

    # --- Lecture du CSV2 ---
    with open(csv2, newline='', encoding="utf-8") as f2:
        r2 = csv.DictReader(f2)
        rows2 = list(r2)

    # Vérification colonnes
    if col1 not in r1.fieldnames:
        raise ValueError(f"Colonne '{col1}' absente du fichier {csv1}")
    if col2 not in r2.fieldnames:
        raise ValueError(f"Colonne '{col2}' absente du fichier {csv2}")

    # Indexation du CSV2 par sa clé
    index = {}
    for row in rows2:
        key = row[col2]
        index.setdefault(key, []).append(row)

    # Construction des entêtes du fichier final
    # évite de dupliquer la colonne de jointure du csv2
    output_fields = r1.fieldnames + [c for c in r2.fieldnames if c != col2]

    # Jointure proprement dite
    joined_rows = []

    for row1 in rows1:
        key = row1[col1]
        if key in index:
            for row2 in index[key]:
                merged = {**row1, **{k: v for k, v in row2.items() if k != col2}}
                joined_rows.append(merged)

    # Écriture du fichier final
    with open(output_csv, "w", newline='', encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(joined_rows)


def mapping_colonne_ids(csv_list, col_name, sep=",", key=None):
    """
    Lit plusieurs CSV et crée un dictionnaire {valeur_unique: ID}
    pour une colonne qui peut contenir :
    - une seule valeur texte
    - une liste encodée comme chaîne (ex: "['a', 'b']")
    - une liste de dictionnaires (ex: "[{'genre_id': '21', 'genre_title': 'Hip-Hop'}]")
    
    Arguments :
    - csv_list : liste des chemins CSV
    - col_name : nom de la colonne à analyser
    - sep : séparateur si plusieurs valeurs simples (par défaut ",")
    - key : si les éléments sont des dictionnaires, extraire cette clé (ex: "genre_id")
    """
    
    valeurs_set = set()

    for csv_path in csv_list:
        ensure_output_csv(csv_path)
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                valeur = row[col_name].strip()
                if not valeur:
                    continue

                try:
                    parsed = ast.literal_eval(valeur)
                    if isinstance(parsed, list):
                        for v in parsed:
                            if isinstance(v, dict) and key and key in v:
                                valeurs_set.add(v[key].strip())
                            elif v:
                                valeurs_set.add(str(v).lower().strip())
                        continue
                except (ValueError, SyntaxError):
                    pass

                # Gestion séparateur simple
                if sep in valeur:
                    for v in valeur.split(sep):
                        valeurs_set.add(v.lower().strip())
                else:
                    valeurs_set.add(valeur.lower())

    # Création du mapping avec IDs auto-incrémentés
    valeurs_sorted = sorted(valeurs_set)
    mapping = {val: i + 1 for i, val in enumerate(valeurs_sorted)}

    return mapping



def mapping_2_colonnes(input:str, col_key:str, col_value:str)-> dict:
    """
    Lit un CSV et crée un dictionnaire {clé: valeur}
    à partir de deux colonnes spécifiées.
    
    :param input: chemin du fichier CSV
    :param col_key: nom de la colonne pour les clés
    :param col_value: nom de la colonne pour les valeurs ici c'est l'id
    :return: dictionnaire Python
    """
    mapping = {}
    with open(input, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row[col_key].strip()
            value = int(row[col_value].strip())
            if key:
                mapping[key] = value
    return mapping


def dict_to_csv(data: dict, output: str,col1 = "clé",col2 = "valeur"):
    """
    Transforme un dictionnaire en fichier CSV avec deux colonnes : clé et valeur.
    
    :param data: dictionnaire Python
    :param output: nom du fichier CSV de sortie
    """
    with open(output, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # écrire l'en-tête
        writer.writerow([col1, col2])
        # écrire chaque paire clé/valeur
        for key, value in data.items():
            writer.writerow([key, value])





def strip_html(text: str) -> str:
    """
    Supprime les balises HTML et retourne uniquement le texte propre.
    """
    if text is None:
        return ""

    text = str(text)

    if "<" not in text:
        return text.strip()

    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def normalize_cell(value):
    """
    Nettoie un champ CSV :
    - supprime HTML
    - convertit les listes/dicts en texte lisible
    - gère None / NaN
    """
    if value is None:
        return ""

    text = str(value).strip()

    # Gérer NaN
    if text.lower() == "nan":
        return ""

    # Convertir string→liste/dict si c'est du Python littéral
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
        try:
            parsed = ast.literal_eval(text)
            return str(parsed)
        except Exception:
            pass

    # Nettoyer HTML
    if "<" in text:
        return strip_html(text)

    return text


def clean_csv(df):
    """
    Lit un CSV, nettoie toutes les colonnes, écrit un CSV propre.
    """
    for col in df.columns:
        df[col] = df[col].apply(normalize_cell)
    return df

def extraire_genre_ids(input_csv, output_csv, colonne, sep=","):
    """
    Lit un CSV et extrait les genre_id depuis une colonne contenant
    des listes de dictionnaires encodées en chaîne.
    
    - input_csv : fichier source
    - output_csv : fichier final
    - colonne : nom de la colonne à analyser
    - sep : séparateur utilisé pour écrire plusieurs IDs (par défaut ",")
    """
    ensure_output_csv(output_csv)

    with open(input_csv, newline='', encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)

    if colonne not in reader.fieldnames:
        raise ValueError(f"La colonne '{colonne}' n'existe pas dans le CSV.")

    for row in rows:
        valeur = row[colonne]

        if not valeur or valeur.strip() == "":
            row[colonne] = ""
            continue

        try:
            # Convertir la chaîne en liste de dictionnaires
            parsed = ast.literal_eval(valeur)
            if isinstance(parsed, list):
                ids = [d.get("genre_id") for d in parsed if isinstance(d, dict) and "genre_id" in d]
                row[colonne] = sep.join(ids)
            else:
                row[colonne] = ""
        except (ValueError, SyntaxError):
            # Si parsing échoue, on laisse vide
            row[colonne] = ""

    # Écriture du nouveau CSV
    with open(output_csv, "w", newline='', encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
def _random_hash(text):
    return hashlib.sha256(str(text).encode()).hexdigest()[:32]

def _encode_gender(g):
    if pd.isna(g): return "O"
    g = str(g).strip().lower()
    return "H" if g == "homme" else "F" if g == "femme" else "O"

def _interval_to_date(interval):
    """Transforme '18-25 ans' en '2003-01-01'"""
    if pd.isna(interval): return None
    try:
        clean_str = str(interval).replace("ans", "").strip()
        parts = clean_str.split("-")
        if len(parts) == 2:
            age = random.randint(int(parts[0]), int(parts[1]))
            year = datetime.now().year - age
            return f"{year}-01-01"
    except:
        pass
    return None
        

def convertir_duree(valeur):
    """
    Transforme une durée h:m:s ou m:s en secondes (float).
    Ex: "03:45" -> 225.0
    Ex: "1:00:00" -> 3600.0
    """
    # Si c'est vide ou nul
    if pd.isna(valeur) or valeur == "":
        return None
    
    valeur = str(valeur).strip()
    
    # Cas 1 : C'est déjà un nombre (ex: "225")
    if valeur.replace('.', '', 1).isdigit():
        return float(valeur)
    
    # Cas 2 : C'est un format "mm:ss" ou "hh:mm:ss"
    if ':' in valeur:
        try:
            parties = valeur.split(':')
            # Format mm:ss (ex: 03:45)
            if len(parties) == 2:
                minutes = int(parties[0])
                secondes = int(parties[1])
                return float(minutes * 60 + secondes)
            # Format hh:mm:ss (ex: 01:00:00)
            elif len(parties) == 3:
                heures = int(parties[0])
                minutes = int(parties[1])
                secondes = int(parties[2])
                return float(heures * 3600 + minutes * 60 + secondes)
        except ValueError:
            return None # En cas de format bizarre
            
    return None
