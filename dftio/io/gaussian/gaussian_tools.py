import glob
import json
import os
import re
import shutil
from collections import Counter
from pprint import pprint
import warnings
import ase
from ase.db import connect

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms, Atom
from typing import Union, List, Dict, Any, Set

from .gaussian_conventions import orbital_idx_map
from emoles.fchk_parser import read_fchk


def chk_valid_gau_log_unit(file_path, hamiltonian=False, overlap=False, density_matrix=False,
                           is_fixed_convention=False):
    print('Start parsing {}'.format(file_path))
    required_patterns = [
        r"Standard orientation:",
        r"NBasis=",
        r'Normal termination of Gaussian'
    ]

    if hamiltonian:
        required_patterns.append(r" Fock matrix")
    if overlap:
        required_patterns.append(r"\*+\s*Overlap\s*\*+")
    if density_matrix:
        required_patterns.append(r" Total density matrix:")

    if not is_fixed_convention:
        required_patterns.extend([
            r"Standard basis:",
            r"\s*Gross orbital populations:"
        ])

    patterns = [re.compile(pattern) for pattern in required_patterns]
    found_patterns = [False] * len(patterns)

    with open(file_path, 'r') as file:
        for line in file:
            for i, pattern in enumerate(patterns):
                if not found_patterns[i] and pattern.search(line):
                    found_patterns[i] = True
    if all(found_patterns):
        print('All patterns parsed successfully for {}'.format(file_path))
        return True
    else:
        print('Patterns not parsed successfully for {}'.format(file_path))
        return False


def chk_valid_gau_logs(
    root,
    filename,
    hamiltonian=False,
    overlap=False,
    density_matrix=False,
    is_fixed_convention=False,
    valid_gau_info_path='./valid_gaussian_logs.txt',
    invalid_gau_info_path='./invalid_gaussian_logs.txt',
):
    """
    Traverse the directory tree starting at `root`, find all files containing `filename` in their name,
    and check each with `chk_valid_gau_log_unit`. Valid and invalid paths are written to separate log files.
    """
    valid_count = 0
    invalid_count = 0

    # Ensure output files exist and open in append mode
    with open(valid_gau_info_path, 'a') as valid_file, \
         open(invalid_gau_info_path, 'a') as invalid_file:

        # Walk through all directories and subdirectories
        for dirpath, dirnames, filenames in os.walk(root):
            for fname in filenames:
                if filename in fname:
                    a_file = os.path.join(dirpath, fname)

                    # Perform the validity check
                    if chk_valid_gau_log_unit(
                        a_file,
                        hamiltonian=hamiltonian,
                        overlap=overlap,
                        density_matrix=density_matrix,
                        is_fixed_convention=is_fixed_convention
                    ):
                        valid_file.write(f"{a_file}\n")
                        valid_count += 1
                    else:
                        invalid_file.write(f"{a_file}\n")
                        invalid_count += 1

    # Summary
    print(f"Valid Gaussian log files: {valid_count}")
    print(f"Invalid Gaussian log files: {invalid_count}")
    print(f"Valid file paths written to: {valid_gau_info_path}")
    print(f"Invalid file paths written to: {invalid_gau_info_path}")



def split_files_by_atoms(file_list_path, output_train_path, output_valid_path, output_test_path):
    # Read file paths from the input file
    with open(file_list_path, 'r') as file:
        file_paths = file.readlines()

    train_set = []
    valid_set = []
    test_set = []

    # Iterate over the file paths
    for file_path in file_paths:
        file_path = file_path.strip()  # Clean up any extra spaces or newlines
        if os.path.exists(file_path):
            try:
                nbasis, atoms = get_basic_info(file_path)
                num_atoms = len(atoms)

                # Classify the file paths based on the number of atoms
                if num_atoms <= 20:
                    train_set.append(file_path)
                elif num_atoms in [21, 22]:
                    valid_set.append(file_path)
                else:
                    test_set.append(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Save the file paths into the respective output files
    with open(output_train_path, 'w') as train_file:
        train_file.write("\n".join(train_set))

    with open(output_valid_path, 'w') as valid_file:
        valid_file.write("\n".join(valid_set))

    with open(output_test_path, 'w') as test_file:
        test_file.write("\n".join(test_set))

    print(f'train/valid/test size: {len(train_set)}/{len(valid_set)}/{len(test_set)}')


def transform_matrix(matrix, transform_indices):
    matrix = matrix[..., transform_indices, :]
    matrix = matrix[..., :, transform_indices]
    return matrix


def cut_matrix(full_matrix, atom_in_mo_indices, threshold=1e-8):
    partitioned_blocks = {}
    atom_indeces = sorted(set(atom_in_mo_indices))
    atom_positions = {atom: [i for i, x in enumerate(atom_in_mo_indices) if x == atom] for atom in atom_indeces}

    # Extract blocks for each pair of atoms
    for ii, i in enumerate(atom_indeces):
        for j in atom_indeces[ii:]:
            key = f"{i}_{j}_0_0_0"
            rows = atom_positions[i]
            cols = atom_positions[j]
            block = full_matrix[np.ix_(rows, cols)]
            if np.max(np.abs(block)) > threshold:
                partitioned_blocks[key] = block
    return partitioned_blocks


def get_nbasis(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    nbasis = None
    for line in lines:
        # Extract NBasis
        if line.strip().startswith("NBasis"):
            # 直接匹配第一个出现的连续数字
            match = re.search(r"\d+", line)
            if match:
                nbasis = int(match.group(0)) # 恢复使用 group(0)
                break
    if nbasis is None:
        print("NBasis keyword not found in the log file.")
    return nbasis


def get_basic_info(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    in_standard_orientation = False
    atoms = Atoms()
    for idx, line in enumerate(lines):
        # Extract atomic coordinates in standard orientation
        if "Standard orientation:" in line:
            in_standard_orientation = True
            atom_data = []  # Reset atom data for the latest standard orientation
        elif in_standard_orientation and "---" in line:
            if len(atoms) > 0:
                break
        elif in_standard_orientation:
            parts = line.split()
            if len(parts) == 6:  # We expect 6 parts in a valid atom data line
                try:
                    atomic_number = int(parts[1])
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                    atoms.append(Atom(symbol=atomic_number, position=(x, y, z)))
                except ValueError:
                    continue
    nbasis = None
    for line in lines[idx:]:
        # Extract NBasis
        if line.strip().startswith("NBasis"):
            # 直接匹配第一个出现的连续数字
            match = re.search(r"\d+", line)
            if match:
                nbasis = int(match.group(0)) # 恢复使用 group(0)
                break
    if nbasis is None:
        raise RuntimeError("NBasis keyword not found in the log file.")
    return nbasis, atoms


def find_basis_set(file_path):
    target_line = 'Standard basis:'
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip().startswith(target_line):
                return line.strip()[len(target_line):].strip()
    return None


def simplify_orbitals(orbitals):
    simplified = ''
    seen_pdf = set()  # To keep track of 's' orbitals we've already added
    for orbital in orbitals:
        letter = orbital[-1]  # Get the last character (s, p, d, or f)
        if letter == 's':
            simplified += letter
        elif letter in 'pdf':
            if orbital not in seen_pdf:
                simplified += letter
                seen_pdf.add(orbital)
        else:
            raise NotImplementedError(f'Orbital {letter} has not been supported.')
    return simplified


def process_atomic_orbitals(orbital_list, orbital_idx_map):
    # Create a dictionary to store orbitals by type
    orbitals_by_type = {'s': [], 'p': [], 'd': [], 'f': []}

    # Sort orbitals into types
    for i, orbital in enumerate(orbital_list):
        orbital_type = orbital[-1]
        orbitals_by_type[orbital_type].append((i, orbital))

    transform_indices = []
    orbital_counts = Counter()
    sorted_orbital_str = ''
    division_list = [1, 3, 5, 7]
    # Process each orbital type in order: s, p, d, f
    for idx, orbital_type in enumerate(['s', 'p', 'd', 'f']):
        # Sort orbitals of this type by principal quantum number
        sorted_orbitals = sorted(orbitals_by_type[orbital_type], key=lambda x: int(x[1][0]))

        if orbital_type == 's':
            # For s orbitals, just add the indices
            transform_indices.extend([x[0] for x in sorted_orbitals])
        else:
            # For p, d, f orbitals, apply the shifts
            shifts = orbital_idx_map[orbital_type]
            for i in range(0, len(sorted_orbitals), len(shifts)):
                for shift in shifts:
                    if i + shift < len(sorted_orbitals):
                        transform_indices.append(sorted_orbitals[i + shift][0])
        # Count the orbitals
        orbital_counts[orbital_type] = len(sorted_orbitals)
        if orbital_counts[orbital_type] > 0:
            a_division = division_list[idx]
            assert abs(orbital_counts[orbital_type] % a_division) < 1e-6, f"{orbital_type} orbital is not multiple of {a_division}"
            sorted_orbital_str = sorted_orbital_str + str(orbital_counts[orbital_type] // a_division) + orbital_type

    return transform_indices, sorted_orbital_str


# Key word Pop=Full is required
def parse_orbital_populations(filename, nbf, orbital_idx_map):
    target_pattern = re.compile(r'\s*Gross orbital populations:')
    with open(filename, 'r') as f:
        # Search for the target pattern
        for line in f:
            if target_pattern.search(line):
                break
        else:
            raise ValueError(f"No match for orbital population block found in file {filename}")
        orbitals = []
        atoms_list = []
        next(f)  # Skip the line with column numbers
        for i in range(nbf):
            parts = next(f).split()
            if len(parts) >= 5:
                an_orbital = parts[3][:2].lower()
                atoms_list.append(parts[2])
            else:
                an_orbital = parts[1][:2].lower()
            orbitals.append(an_orbital)
        atom_to_orbitals = {}
        atom_to_simplified_orbitals = {}
        atom_to_dftio_orbitals = {}
        atom_to_transform_indices = {}
        orbital_index = 1
        for atom in atoms_list:
            # Get the corresponding orbitals for the current atom
            unit_orbitals = ['1s']
            while orbital_index < len(orbitals):
                orbital = orbitals[orbital_index]
                orbital_index += 1
                if orbital == '1s':
                    break
                unit_orbitals.append(orbital)
            if not atom in atom_to_orbitals.keys():
                atom_to_orbitals[atom] = unit_orbitals
                transform_indices, sorted_orbital_str = process_atomic_orbitals(unit_orbitals, orbital_idx_map)
                atom_to_dftio_orbitals[atom] = sorted_orbital_str
                atom_to_transform_indices[atom] = transform_indices
                atom_to_simplified_orbitals[atom] = simplify_orbitals(unit_orbitals)
    return orbitals, atom_to_orbitals, atom_to_simplified_orbitals, atom_to_dftio_orbitals, atom_to_transform_indices


def generate_molecule_transform_indices(atom_types, atom_to_transform_indices):
    molecule_transform_indices = []
    atom_in_mo_indices = []
    current_offset = 0

    for atomic_idx, atom_type in enumerate(atom_types):
        atom_indices = atom_to_transform_indices[atom_type]
        adjusted_indices = [index + current_offset for index in atom_indices]
        molecule_transform_indices.extend(adjusted_indices)
        atom_in_mo_indices.extend([atomic_idx] * len(atom_indices))
        current_offset += max(atom_indices) + 1

    return molecule_transform_indices, atom_in_mo_indices


def get_orbital_labels(atom_symbols, orbital_configs):
    """
    Generate orbital labels for a molecule based on atom symbols and orbital configurations.

    :param atom_symbols: List of atom symbols in the molecule (e.g., ['O', 'H', 'H'])
    :param orbital_configs: Dictionary of orbital configurations for each atom type
    :return: List of orbital labels for the entire molecule
    """
    orbital_labels = []

    for atom in atom_symbols:
        if atom not in orbital_configs:
            raise ValueError(f"Orbital configuration for atom {atom} not found.")

        atom_orbitals = orbital_configs[atom]
        for orbital in atom_orbitals:
            if orbital == 's':
                orbital_labels.append('s')
            elif orbital == 'p':
                orbital_labels.extend(['p'] * 3)
            elif orbital == 'd':
                orbital_labels.extend(['d'] * 5)
            elif orbital == 'f':
                orbital_labels.extend(['f'] * 7)
            else:
                raise ValueError(f"Unknown orbital type: {orbital}")

    return orbital_labels


def matrix_to_image(matrix, filename='matrix_image.png', orbital_labels=None):
    # Convert the matrix to a numpy array
    matrix = np.array(matrix)

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(30, 30))

    # Create a color-mapped image of the matrix
    im = ax.imshow(matrix, cmap='viridis')

    # Add a colorbar
    plt.colorbar(im)

    # Add annotations for each cell
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.annotate(f'{matrix[i, j]:.2f}',
                        xy=(j, i),
                        xytext=(0, 0),
                        textcoords='offset points',
                        ha='center',
                        va='center',
                        color='w',
                        fontweight='bold')

    # Set title and labels
    ax.set_title('Matrix Visualization')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Add custom tick labels for columns
    if orbital_labels:
        ax.set_xticks(range(len(orbital_labels)))
        ax.set_xticklabels(orbital_labels)
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='x', rotation=0, labelsize=10, pad=5)
    else:
        ax.set_xticks([])

    # Remove y-axis ticks
    ax.set_yticks([])

    # Adjust layout to prevent clipping of tick-labels
    plt.tight_layout()

    # Save the image
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Image saved as {filename}")


# modified from Mokit,
# see https://github.com/1234zou/MOKIT/blob/7499356b1ff0f9d8b9efbb846395059867dbba4c/src/rwwfn.f90#L895
# Key word IOp(3/33=1) is required
def read_int1e_from_gau_log(logname, matrix_type, nbf):
    matrix_types = {
        0: r"Overlap",
        1: r"Kinetic Energy",
        2: r"Potential Energy",
    }

    if matrix_type not in matrix_types:
        raise ValueError(
            f"Invalid matrix_type = {matrix_type}. Allowed values are 0/1/2 for Overlap/Kinetic/Potential.")
    target_pattern = re.compile(rf"\*+\s*{matrix_types[matrix_type]}\s*\*+")
    mat = np.zeros((nbf, nbf))
    with open(logname, 'r') as f:
        # Search for the target pattern
        for line in f:
            if target_pattern.search(line):
                break
        else:
            raise ValueError(f"No match for '{matrix_types[matrix_type]}' found in file {logname}")

        # Read the matrix data
        n = (nbf + 4) // 5  # Equivalent to ceiling division
        for i in range(n):
            next(f)  # Skip the line with column numbers
            k = 5 * i
            for j in range(k, nbf):
                line = next(f).split()
                m = min(5, nbf - k)
                actual_line_len = len(line[1:m + 1])
                mat[k:k + actual_line_len, j] = [float(x.replace('D', 'E')) for x in line[1:m + 1]]
    # Mirror the upper triangle to the lower triangle
    mat = mat + mat.T - np.diag(mat.diagonal())
    return mat


# modified from Mokit,
# see https://github.com/1234zou/MOKIT/blob/7499356b1ff0f9d8b9efbb846395059867dbba4c/src/rwwfn.f90#L895
# Key word IOp(5/33=3, 3/33=1) is required
def read_fock_from_gau_log(logname, nbf):
    target_pattern = re.compile(rf" Fock matrix")
    pattern_counts = 0
    with open(logname, 'r') as f:
        for line in f:
            if target_pattern.search(line):
                pattern_counts = pattern_counts + 1
    current_counts = 0
    mat = np.zeros((nbf, nbf))
    with open(logname, 'r') as f:
        for line in f:
            if target_pattern.search(line):
                current_counts = current_counts + 1
            if current_counts == pattern_counts:
                break
        # Read the matrix data
        n = (nbf + 4) // 5  # Equivalent to ceiling division
        for i in range(n):
            next(f)  # Skip the line with column numbers
            k = 5 * i
            for j in range(k, nbf):
                line = next(f).split()
                m = min(5, nbf - k)
                actual_line_len = len(line[1:m + 1])
                mat[k:k + actual_line_len, j] = [float(x.replace('D', 'E')) for x in line[1:m + 1]]
    # Mirror the upper triangle to the lower triangle
    mat = mat + mat.T - np.diag(mat.diagonal())
    return mat


# modified from Mokit,
# see https://github.com/1234zou/MOKIT/blob/7499356b1ff0f9d8b9efbb846395059867dbba4c/src/rwwfn.f90#L3405
# Key word IOp(5/33=3, 3/33=1) is required
def read_density_from_gau_log(logname, nbf):
    target_pattern = re.compile(rf" Total density matrix:")
    mat = np.zeros((nbf, nbf))
    with open(logname, 'r') as f:
        for line in f:
            if target_pattern.search(line):
                break
        # Read the matrix data
        n = (nbf + 4) // 5  # Equivalent to ceiling division
        for i in range(n):
            next(f)  # Skip the line with column numbers
            k = 5 * i
            for j in range(k, nbf):
                line = next(f).split()
                m = min(5, nbf - k)
                actual_line_len = len(line[1:m + 1])
                mat[k:k + actual_line_len, j] = [float(x.replace('D', 'E')) for x in line[1:m + 1]]
    # Mirror the upper triangle to the lower triangle
    mat = mat + mat.T - np.diag(mat.diagonal())
    return mat


def get_atoms(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    in_standard_orientation = False
    atoms = Atoms()

    for idx, line in enumerate(lines):
        # Extract atomic coordinates in standard orientation
        if "Standard orientation:" in line:
            in_standard_orientation = True
            atom_data = []  # Reset atom data for the latest standard orientation
        elif in_standard_orientation and "---" in line:
            if len(atoms) > 0:
                break
        elif in_standard_orientation:
            parts = line.split()
            if len(parts) == 6:  # We expect 6 parts in a valid atom data line
                try:
                    atomic_number = int(parts[1])
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                    atoms.append(Atom(symbol=atomic_number, position=(x, y, z)))
                except ValueError:
                    continue
    return atoms


def get_minimal_log_paths_for_all_elements(db_path: str, results_root: str) -> List[str]:
    """
    Scans an ASE DB to find a minimal set of log files that covers all unique elements.

    This function performs three main steps:
    1. Scans the database to gather information about all structures (elements, spin, charge).
    2. Uses a greedy algorithm to find the smallest set of IDs covering all elements.
    3. Traverses the 'results_root' directory to find the log files corresponding to these IDs,
       even if they are deeply nested.

    Args:
        db_path (str): Path to the ASE database file.
        results_root (str): The root directory where calculation results are stored.

    Returns:
        List[str]: A list of absolute paths to the necessary log files.
    """
    # --- 1. Scan DB once to collect all necessary data (No changes here) ---
    all_elements: Set[str] = set()
    id_info: Dict[int, Dict[str, Any]] = {}
    print(f"Connecting to ASE database at: {db_path}")
    try:
        with connect(db_path) as db:
            print("Scanning database for structure information...")
            for row in db.select():
                if not all(hasattr(row, key) for key in ['real_id', 'real_spin', 'real_charge', 'symbols']):
                    warnings.warn(f"Row id={row.id} is missing required keys. Skipping.")
                    continue

                real_id = row.real_id
                elements = set(row.symbols)
                all_elements.update(elements)
                id_info[real_id] = {
                    'elements': elements,
                    'spin': row.real_spin,
                    'charge': row.real_charge,
                }
    except Exception as e:
        print(f"Error processing database: {e}")
        return []

    if not id_info:
        print("No valid entries with required keys found in the database.")
        return []

    # --- 2. Greedy algorithm to find minimal set of IDs (No changes here) ---
    uncovered_elements = all_elements.copy()
    minimal_ids: List[int] = []
    print(f"\nTotal unique elements to cover: {sorted(list(uncovered_elements))}")

    while uncovered_elements:
        best_id, max_coverage = -1, 0
        for real_id, info in id_info.items():
            if real_id in minimal_ids: continue
            newly_covered_count = len(uncovered_elements.intersection(info['elements']))
            if newly_covered_count > max_coverage:
                max_coverage = newly_covered_count
                best_id = real_id

        if best_id != -1:
            covered = uncovered_elements.intersection(id_info[best_id]['elements'])
            minimal_ids.append(best_id)
            uncovered_elements -= covered
            print(f"  - Selected ID: {best_id}, covers new: {sorted(list(covered))}. "
                  f"Remaining: {len(uncovered_elements)}")
        else:
            print("No more elements can be covered. Stopping.")
            break

    print(f"\nMinimal set of IDs found: {minimal_ids}")

    # --- 3. Construct log file paths using os.walk() (This part is modified) ---

    # First, walk the directory tree ONCE to build a map of folder names to their full paths.
    # This is much more efficient than walking the tree for each ID.
    print(f"\nScanning file system under '{results_root}' to map job folders. This may take a while...")
    folder_path_map: Dict[str, str] = {}
    for dirpath, dirnames, _ in os.walk(results_root):
        # We only look at directory names that match our pattern.
        for dirname in dirnames:
            if dirname.startswith('id_') and '_spin_' in dirname and '_charge_' in dirname:
                folder_path_map[dirname] = os.path.join(dirpath, dirname)

    print(f"Found {len(folder_path_map)} potential job folders.")

    # Now, use the pre-built map to find the paths for our minimal set of IDs.
    log_paths: List[str] = []
    print("\nLocating corresponding log files from the file system map...")
    for real_id in minimal_ids:
        info = id_info[real_id]
        folder_name = f"id_{real_id}_spin_{info['spin']}_charge_{info['charge']}"

        # Look up the full path from our map.
        folder_path = folder_path_map.get(folder_name)

        if folder_path:
            # Search for .log or .out files inside the found folder.
            found_files = glob.glob(os.path.join(folder_path, '*.log')) + \
                          glob.glob(os.path.join(folder_path, '*.out'))

            if found_files:
                log_path = os.path.abspath(found_files[0])
                log_paths.append(log_path)
                print(f"  - Found: {log_path}")
                if len(found_files) > 1:
                    warnings.warn(f"Multiple log files in {folder_path}. Using '{found_files[0]}'.")
            else:
                warnings.warn(f"Log file (.log or .out) not found inside folder: {folder_path}")
        else:
            # This means the folder for this specific ID was not found during the walk.
            warnings.warn(f"Job folder '{folder_name}' not found anywhere under {results_root}")

    return log_paths

# Key word Pop=Full is required
def get_convention(
        filenames: Union[str, List[str]],
        dump_file: str = None,
        format: str = 'log',
        orbital_idx_map: dict = orbital_idx_map
) -> Dict[str, Any]:
    """
    Reads one or more Gaussian log files to extract and merge basis set convention information.

    Args:
        filenames (Union[str, List[str]]): A single log filename or a list of log filenames.
        dump_file (str, optional): If provided, saves the final convention dict to this JSON file.
        orbital_idx_map (dict, optional): Orbital index map.

    Returns:
        Dict[str, Any]: The merged convention dictionary containing information from all files.
    """
    # If a single filename (string) is passed, convert it to a list.
    if isinstance(filenames, str):
        filenames = [filenames]

    # Initialize an empty convention dictionary to store merged results.
    merged_convention = {
        'atom_to_simplified_orbitals': {},
        'atom_to_dftio_orbitals': {},
        'atom_to_transform_indices': {},
        'basis_name': None,
    }

    # Process each file in the list.
    for filename in filenames:
        print(f"Processing file: {filename}...")

        if format == 'log':
            nbasis = get_nbasis(filename)
            # Parse orbital info from the current file.
            _, _, atom_to_simplified_orbitals, atom_to_dftio_orbitals, atom_to_transform_indices = parse_orbital_populations(
                filename, nbasis, orbital_idx_map
            )
            basis_name = find_basis_set(filename)
        elif format == 'fchk':
            atoms = read_fchk(filename)
            convention = atoms.info['convention']
            atom_to_simplified_orbitals, atom_to_dftio_orbitals, atom_to_transform_indices = convention['atom_to_simplified_orbitals'], convention['atom_to_dftio_orbitals'], convention['atom_to_transform_indices']
            basis_name = atoms.info['functional_basis']['basis_set']
        else:
            raise NotImplementedError

        # Set the basis name from the first file and check for consistency in others.
        if merged_convention['basis_name'] is None:
            merged_convention['basis_name'] = basis_name
        # Merge the dictionaries. The update() method adds new elements and overwrites existing ones.
        merged_convention['atom_to_simplified_orbitals'].update(atom_to_simplified_orbitals)
        merged_convention['atom_to_dftio_orbitals'].update(atom_to_dftio_orbitals)
        merged_convention['atom_to_transform_indices'].update(atom_to_transform_indices)

    # If a dump_file path is provided, save the result as a JSON file.
    if dump_file:
        print(f"\nDumping merged convention to {dump_file}...")
        with open(dump_file, 'w') as f:
            json.dump(merged_convention, f, indent=4)

    print("\nFinal Merged Convention:")
    pprint(merged_convention, compact=True)

    return merged_convention


def check_eigenvalue_consistency(matrix, manipulated_matrix):
    original_eigenvalues = np.linalg.eigvals(matrix)
    manipulated_eigenvalues = np.linalg.eigvals(manipulated_matrix)
    original_eigenvalues.sort()
    manipulated_eigenvalues.sort()
    tolerance = 1e-10
    is_consistent = np.allclose(original_eigenvalues, manipulated_eigenvalues, atol=tolerance)
    return is_consistent


def check_transform(filepath):
    convention = get_convention(filename='gau.log')
    nbasis, atoms = get_basic_info(filepath)
    molecule_transform_indices, _ = generate_molecule_transform_indices(atom_types=atoms.symbols,
                                                                        atom_to_transform_indices=convention[
                                                                            'atom_to_transform_indices'])
    hamiltonian_matrix = read_int1e_from_gau_log(filepath, matrix_type=3, nbf=nbasis)
    new_hamiltonian_matrix = transform_matrix(hamiltonian_matrix, molecule_transform_indices)
    consistent_flag = check_eigenvalue_consistency(hamiltonian_matrix, new_hamiltonian_matrix)
    if consistent_flag:
        print('Hamiltonian matrix transform is consistent for eigenvalues.')
    else:
        print('Hamiltonian matrix transform is non-consistent for eigenvalues.')

    overlap_matrix = read_int1e_from_gau_log(filepath, matrix_type=0, nbf=nbasis)
    new_overlap_matrix = transform_matrix(overlap_matrix, molecule_transform_indices)
    consistent_flag = check_eigenvalue_consistency(overlap_matrix, new_overlap_matrix)
    if consistent_flag:
        print('Overlap matrix transform is consistent for eigenvalues.')
    else:
        print('Overlap matrix transform is non-consistent for eigenvalues.')

    density_matrix = read_density_from_gau_log(filepath, nbf=nbasis)
    new_density_matrix = transform_matrix(density_matrix, molecule_transform_indices)
    consistent_flag = check_eigenvalue_consistency(density_matrix, new_density_matrix)
    if consistent_flag:
        print('Density matrix transform is consistent for eigenvalues.')
    else:
        print('Density matrix transform is non-consistent for eigenvalues.')


# traverse root folder to find log files that named with logname
# copy them out to the dst folder and name them with their parent folder name
def traverse_cp_log(root_folder, logname, dst_folder):
    root_folder = os.path.abspath(root_folder)
    dst_folder = os.path.abspath(dst_folder)
    common_path = os.path.commonpath([root_folder, dst_folder])
    assert common_path != root_folder, f"Error: {dst_folder} is a subfolder of {root_folder}"
    os.makedirs(dst_folder, exist_ok=True)
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file == logname:
                folder_name = os.path.basename(subdir)
                shutil.copy(src=os.path.join(subdir, file),
                            dst=os.path.join(dst_folder, folder_name + '.log'))


def traverse_find_log(root_folder, logname):
    root_folder = os.path.abspath(root_folder)
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file == logname:
                yield os.path.join(subdir, file)


def get_gau_logs(valid_file_path: str):
    gau_log_list = []
    with open(valid_file_path, 'r') as file:
        for line in file.readlines():
            gau_log_list.append(line.strip())
    return gau_log_list


def convert_to_sorted_orbitals(atom_to_dftio_orbitals):
    atom_to_sorted_orbitals = {}

    for atom, orbitals in atom_to_dftio_orbitals.items():
        expanded_orbitals = ""
        i = 0
        while i < len(orbitals):
            if orbitals[i].isdigit():
                count = int(orbitals[i])
                i += 1
                orbital_type = orbitals[i]
                expanded_orbitals += orbital_type * count
            i += 1
        atom_to_sorted_orbitals[atom] = expanded_orbitals

    return atom_to_sorted_orbitals


def get_phase_sign_list(atomic_symbols, atom_to_sorted_orbitals, orbital_sign_map):
    phase_sign_list = []

    for symbol in atomic_symbols:
        if symbol in atom_to_sorted_orbitals:
            simplified_orbitals = atom_to_sorted_orbitals[symbol]

            for orbital in simplified_orbitals:
                if orbital in orbital_sign_map:
                    phase_sign_list.extend(orbital_sign_map[orbital])
                else:
                    raise ValueError(f"Unknown orbital type: {orbital}")
        else:
            raise ValueError(f"Unknown atomic symbol: {symbol}")

    return phase_sign_list


def apply_phase_signs_to_matrix(matrix, phase_sign_list):
    # Ensure the dimensions match
    if matrix.shape[0] != len(phase_sign_list) or matrix.shape[1] != len(phase_sign_list):
        raise ValueError("Hamiltonian dimensions do not match the phase sign list length")

    # Create a 2D array of phase signs
    phase_sign_matrix = np.outer(phase_sign_list, phase_sign_list)
    # Apply the phase signs to the Hamiltonian
    modified_matrix = matrix * phase_sign_matrix

    return modified_matrix
