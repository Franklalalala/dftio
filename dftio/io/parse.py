import os
import glob
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import h5py
import ase
import pickle
import lmdb
from dftio.data import _keys
from dftio.utils import j_must_have
from dftio.register import Register
from ase.io.trajectory import Trajectory
import logging

log = logging.getLogger(__name__)


def find_target_line(f, target):
    line = f.readline()
    while line:
        if target in line:
            return line
        line = f.readline()
    return None


class ParserRegister:
    _register = Register()

    @staticmethod
    def register(target):
        return ParserRegister._register.register(target)

    def __new__(cls, mode: str, **kwargs):
        if mode in ParserRegister._register.keys():
            return ParserRegister._register[mode](**kwargs)
        else:
            raise Exception(f"Descriptor mode: {mode} is not registered!")


class Parser(ABC):
    def __init__(
            self,
            root,
            prefix,
            **kwargs
    ):
        """All DFT parser need to inherit this class and implement the abstract methods."""
        self.root = root
        self.prefix = prefix
        if '/' in prefix:
            log.warning(f"There are '/' in the prefix: {prefix}, which should not appear. You should consider use '-r' "
                        f"to change the root rather than include path in the prefix, which might cause error.")

        if isinstance(root, list) and all(isinstance(item, str) for item in root):
            self.raw_datas = root
        else:
            self.raw_datas = glob.glob(os.path.join(root, '*' + prefix + '*'))

        assert (len(self.raw_datas) != 0), 'There are no folders that meet the requirements in the directory!'

    def __len__(self):
        return len(self.raw_datas)

    def __getitem__(self, idx):
        return self.raw_datas[idx]

    @abstractmethod
    def get_structure(self, idx):
        pass  # return a dict of structure information

    def formula(self, idx):
        structure = self.get_structure(idx)
        atomic_number = j_must_have(structure, _keys.ATOMIC_NUMBERS_KEY)
        return ase.Atoms(numbers=atomic_number).get_chemical_formula()

    def structure_to_ase(self, structure):
        self.check_structure(idx=None, structure=structure)
        cell = structure[_keys.CELL_KEY]
        nframe = cell.shape[0]
        ase_lists = []
        for i in range(nframe):
            ase_lists.append(ase.Atoms(
                numbers=structure[_keys.ATOMIC_NUMBERS_KEY],
                positions=structure[_keys.POSITIONS_KEY][i],
                pbc=structure[_keys.PBC_KEY],
                cell=cell[i]
            ))
        return ase_lists

    def ase_to_structure(self, sys):
        if isinstance(sys, list):
            pos = []
            cell = []
            pbc = sys[0].pbc
            atom_numbs = sys[0].get_atomic_numbers()
            for s in sys:
                assert isinstance(s, ase.Atoms), "The input system is not a list of ase.Atoms object!"
                pos.append(s.positions)
                cell.append(s.cell)
                assert np.all(s.pbc == pbc), "The input system is not a list of ase.Atoms object with same PBC!"
                assert np.all(
                    s.get_atomic_numbers() == atom_numbs), "The input system is not a list of ase.Atoms object with same atomic numbers!"

            structure = {
                _keys.ATOMIC_NUMBERS_KEY: atom_numbs.astype(np.int32),
                _keys.PBC_KEY: pbc,
                _keys.POSITIONS_KEY: np.array(pos).astype(np.float32),
                _keys.CELL_KEY: np.array(cell).astype(np.float32)
            }

        elif isinstance(sys, ase.Atoms):
            structure = {
                _keys.ATOMIC_NUMBERS_KEY: sys.get_atomic_numbers().astype(np.int32),
                _keys.PBC_KEY: sys.pbc,
                _keys.POSITIONS_KEY: sys.positions[np.newaxis, :, :].astype(np.float32),
                _keys.CELL_KEY: sys.cell[np.newaxis, :, :].astype(np.float32)
            }
        else:
            raise ValueError("The input system is not a list or ase.Atoms object!")

        self.check_structure(idx=None, structure=structure)
        return structure

    @abstractmethod
    def get_eigenvalue(self, idx, band_index_min):
        pass

    @abstractmethod
    def get_blocks(self, idx, hamiltonian: bool = False, overlap: bool = False, density_matrix: bool = False):
        pass

    def check_structure(self, idx, structure=None):
        if structure is None:
            structure = self.get_structure(idx)
        atomic_number = j_must_have(structure, _keys.ATOMIC_NUMBERS_KEY)
        pos = j_must_have(structure, _keys.POSITIONS_KEY)
        cell = j_must_have(structure, _keys.CELL_KEY)
        pbc = j_must_have(structure, _keys.PBC_KEY)

        # check the shape
        assert len(pos.shape) == 3 and pos.shape[2] == 3, "The shape of structure should be (n_frame, natom, 3)"
        n_frame, natom, _ = pos.shape
        assert len(cell.shape) == 3 and cell.shape[0] == n_frame and cell.shape[1] == 3 and cell.shape[
            2] == 3, "The shape of cell should be (n_frame, 3, 3)"
        assert len(atomic_number.shape) == 1 and atomic_number.shape[
            0] == natom, "The shape of atomic_number should be (natom,)"
        assert len(pbc.shape) == 1 and pbc.shape[0] == 3, "The shape of pbc should be (3)"

        # check dtype
        assert atomic_number.dtype == np.int32, "The dtype of atomic_number should be int"
        assert cell.dtype == np.float32, "The dtype of cell should be float"
        assert pos.dtype == np.float32, "The dtype of pos should be float"
        assert pbc.dtype == np.bool8 or pbc.dtype == np.bool_, "The dtype of pbc should be bool"

        return True

    def check_eigenvalue(self, idx, eigstatus=None):
        if eigstatus is None:
            eigstatus = self.get_eigenvalue(idx)
        eigs = j_must_have(eigstatus, _keys.ENERGY_EIGENVALUE_KEY)
        kpts = j_must_have(eigstatus, _keys.KPOINT_KEY)

        # check the shape
        assert len(eigs.shape) == 3, "The shape of eigenvalue should be (n_frame, n_kpt, n_band)"
        nf, nk, nb = eigs.shape
        assert len(kpts.shape) == 2 and kpts.shape[0] == nk and kpts.shape[
            1] == 3, "The shape of kpoint should be (n_kpt, 3)"

        # check dtype
        assert eigs.dtype == np.float32, "The dtype of eigenvalue should be float"
        assert kpts.dtype == np.float32, "The dtype of kpoint should be float"

        return True

    def check_blocks(self, idx, hamiltonian: bool = False, overlap: bool = False, density_matrix: bool = False):
        ham, ovp, dm = self.get_blocks(idx, hamiltonian, overlap, density_matrix)
        structure = self.get_structure(idx)
        if hamiltonian:
            assert ham is not None, "Hamiltonian should not be None"
            assert len(ham) == structure[_keys.POSITIONS_KEY].shape[
                0], "The number of hamiltonian blocks should be equal to the number of frames"
            assert all([isinstance(h, dict) for h in ham]), "Hamiltonian should be a list of dict"
            assert ham[0].get("0_0_0_0_0") is not None, "Hamiltonian should at least have key 0_0_0_0_0"
            assert ham[0].get("0_0_0_0_0").dtype in [np.float32,
                                                     np.complex64], "The dtype of hamiltonian block should be float real or complex"

        if overlap:
            assert ovp is not None, "Overlap should not be None"
            assert len(ovp) == structure[_keys.POSITIONS_KEY].shape[
                0], "The number of overlap blocks should be equal to the number of frames"
        if density_matrix:
            assert dm is not None, "Density matrix should not be None"
            assert len(dm) == structure[_keys.POSITIONS_KEY].shape[
                0], "The number of density matrix blocks should be equal to the number of frames"

        return True

    def output_dir_path(self, output_file="output_dir_path.txt"):
        with open(output_file, "w") as f:
            for path in self.raw_datas:
                f.write(os.path.abspath(path) + "\n")

    def write(self, idx, outroot, format, eigenvalue, hamiltonian, overlap, density_matrix, band_index_min,
              energy=False, **kwargs):
        if format == "hdf5":
            self.write_hdf5(idx=idx, outroot=outroot, eigenvalue=eigenvalue, hamiltonian=hamiltonian, overlap=overlap,
                            density_matrix=density_matrix, band_index_min=band_index_min, energy=energy)
        elif format in ["dat", "ase"]:
            self.write_dat(idx=idx, outroot=outroot, fmt=format, eigenvalue=eigenvalue, hamiltonian=hamiltonian,
                           overlap=overlap, density_matrix=density_matrix, band_index_min=band_index_min, energy=energy,
                           **kwargs)
        elif format == "lmdb":
            self.write_lmdb(idx=idx, outroot=outroot, eigenvalue=eigenvalue, hamiltonian=hamiltonian, overlap=overlap,
                            density_matrix=density_matrix, band_index_min=band_index_min, energy=energy, **kwargs)
        else:
            raise NotImplementedError(f"Format: {format} is not implemented!")

    def write_hdf5(self, idx, outroot, eigenvalue: bool = False, hamiltonian: bool = False, overlap: bool = False,
                   density_matrix: bool = False, band_index_min=0):
        pass

    def write_struct(self, structure, out_dir, fmt='dat'):
        # write structure
        if fmt == 'dat':
            # The abacus must have PBC, so here we save cell by default
            np.savetxt(os.path.join(out_dir, "cell.dat"), structure[_keys.CELL_KEY].reshape(-1, 3))
            np.savetxt(os.path.join(out_dir, "positions.dat"), structure[_keys.POSITIONS_KEY].reshape(-1, 3))
            np.savetxt(os.path.join(out_dir, "atomic_numbers.dat"), structure[_keys.ATOMIC_NUMBERS_KEY], fmt='%d')
            np.savetxt(os.path.join(out_dir, "pbc.dat"), structure[_keys.PBC_KEY])

        elif fmt == 'ase':
            ase_list = self.structure_to_ase(structure)
            trajfile = Trajectory(os.path.join(out_dir, "xdat.traj"), 'w')
            for istr in ase_list:
                trajfile.write(istr)
            trajfile.close()
        else:
            raise NotImplementedError(f"Format: {fmt} is not implemented!")

    def _write_hdf5_blocks(self, filepath, blocks_list):
        """
        Helper function to write a list of block dicts to HDF5.
        Format style matches original code: Group(frame_idx) -> Dataset(key).
        HDF5 handles complex numbers automatically.
        """
        with h5py.File(filepath, 'w') as fid:
            for i in range(len(blocks_list)):
                default_group = fid.create_group(str(i))
                for key_str, value in blocks_list[i].items():
                    default_group.create_dataset(key_str, data=value)

    def write_dat(self, idx, outroot, fmt='dat', eigenvalue=False, hamiltonian=False, overlap=False,
                  density_matrix=False, band_index_min=0, energy=False, hamiltonian_storage=None, **kwargs):
        # 1. 解析 Hamiltonian 存储模式
        if hamiltonian_storage is None:
            storage_components = {'full'}
        else:
            storage_components = set(hamiltonian_storage) if isinstance(hamiltonian_storage, (list, tuple)) else {
                hamiltonian_storage}
        is_delta_mode = 'delta' in storage_components

        # 2. 准备目录
        os.makedirs(outroot, exist_ok=True)
        structure = self.get_structure(idx)
        out_dir = os.path.join(outroot, self.formula(idx=idx) + ".{}".format(idx))
        os.makedirs(out_dir, exist_ok=True)

        # 3. 写入结构
        self.write_struct(structure, out_dir, fmt=fmt)

        # 4. 写入本征值 (Original style: .npy binary)
        if eigenvalue:
            eigstatus = self.get_eigenvalue(idx=idx, band_index_min=band_index_min)
            self.check_eigenvalue(idx=idx, eigstatus=eigstatus)
            np.save(os.path.join(out_dir, "kpoints.npy"), eigstatus[_keys.KPOINT_KEY])
            np.save(os.path.join(out_dir, "eigenvalues.npy"), eigstatus[_keys.ENERGY_EIGENVALUE_KEY])

        # 5. 写入能量 (Text format)
        if energy:
            if hasattr(self, 'get_etot'):
                energy_data = self.get_etot(idx)
                if energy_data is not None:
                    np.savetxt(os.path.join(out_dir, "total_energy.dat"), energy_data[_keys.TOTAL_ENERGY_KEY])
                    # Write unconverged frame indices if present
                    if _keys.UNCONVERGED_FRAME_INDICES_KEY in energy_data:
                        unconverged_indices = energy_data[_keys.UNCONVERGED_FRAME_INDICES_KEY]
                        if len(unconverged_indices) > 0:
                            with open(os.path.join(out_dir, "unconverged_frames.dat"), 'w') as f:
                                f.write("# Frame indices that did not converge during MD/RELAX\n")
                                for idx_frame in unconverged_indices:
                                    f.write(f"{idx_frame}\n")
                else:
                    log.warning(f"Failed to extract energy for structure {idx}")
            else:
                log.warning(f"Parser does not implement get_etot method")

        # 6. 写入矩阵块 (Original style: HDF5, but logic enhanced for Delta/H0)
        ham = ovp = dm = None
        ham0 = None

        if any([hamiltonian, overlap, density_matrix]):
            # 写入 basis 信息
            with open(os.path.join(out_dir, "basis.dat"), 'w') as f:
                f.write(str(self.get_basis(idx)))

            ham, ovp, dm = self.get_blocks(idx, hamiltonian, overlap, density_matrix)

            # Hamiltonian 处理
            # Hamiltonian 处理
            if hamiltonian:
                # 只要涉及 delta 或 h0，就需要获取 h0_blocks
                if is_delta_mode or 'h0' in storage_components:
                    if not hasattr(self, "get_h0_blocks"):
                        raise NotImplementedError(
                            "hamiltonian_storage requires 'delta' or 'h0' but parser does not implement get_h0_blocks().")
                    ham0, _, _ = self.get_h0_blocks(idx, hamiltonian=True, overlap=False, density_matrix=False)

                # 1. 存储 Delta H -> hamiltonians.h5 (如果是 delta 模式，它作为主输出)
                if is_delta_mode:
                    delta_ham = []
                    for i in range(len(ham)):
                        h_full_frame = ham[i]
                        h0_frame = ham0[i]
                        delta_frame = {}
                        for key, full_block in h_full_frame.items():
                            if key not in h0_frame:
                                raise KeyError(f"H0 Hamiltonian is missing block {key} when computing delta H.")
                            delta_frame[key] = full_block - h0_frame[key]
                        delta_ham.append(delta_frame)

                    self._write_hdf5_blocks(os.path.join(out_dir, "hamiltonians.h5"), delta_ham)

                # 2. 存储 Full H
                if 'full' in storage_components:
                    # 如果 delta 是主输出，full 就叫 hamiltonians_full.h5；否则 full 作为主输出叫 hamiltonians.h5
                    full_name = "hamiltonians_full.h5" if is_delta_mode else "hamiltonians.h5"
                    self._write_hdf5_blocks(os.path.join(out_dir, full_name), ham)

                # 3. 存储 H0 -> hamiltonians_0.h5
                if 'h0' in storage_components:
                    self._write_hdf5_blocks(os.path.join(out_dir, "hamiltonians_0.h5"), ham0)

            if overlap:
                self._write_hdf5_blocks(os.path.join(out_dir, "overlaps.h5"), ovp)

            if density_matrix:
                self._write_hdf5_blocks(os.path.join(out_dir, "density_matrices.h5"), dm)

            # 清理内存
            del ham, ovp, dm, ham0

        return True

    def write_lmdb(self, idx, outroot, eigenvalue: bool = False, hamiltonian: bool = False, overlap: bool = False,
                   density_matrix: bool = False, band_index_min=0, energy: bool = False, hamiltonian_storage=None,
                   **kwargs):
        """
        hamiltonian_storage: list of strings or None
            控制哈密顿量的存储模式。
            - 默认 (None): 等同于 ['full']。存储全哈密顿量到 "hamiltonian"。
            - 'delta': 计算 (H_full - H0) 并存储到 "hamiltonian"。
            - 'full' (当与 'delta' 混用时): 存储 H_full 到 "hamiltonian_full"。
            - 'h0': 存储 H0 到 "hamiltonian_0"。
        """
        if hamiltonian_storage is None:
            storage_components = {'full'}
        else:
            storage_components = set(hamiltonian_storage) if isinstance(hamiltonian_storage, (list, tuple)) else {
                hamiltonian_storage}
        is_delta_mode = 'delta' in storage_components

        os.makedirs(outroot, exist_ok=True)
        out_dir = os.path.join(outroot, "data.{}.lmdb".format(os.getpid()))
        structure = self.get_structure(idx)

        ham = ovp = dm = None
        ham0 = None

        if any([hamiltonian, overlap, density_matrix]):
            ham, ovp, dm = self.get_blocks(idx, hamiltonian, overlap, density_matrix)

        eigstatus = None
        if eigenvalue:
            eigstatus = self.get_eigenvalue(idx=idx, band_index_min=band_index_min)

        energy_data = None
        if energy:
            if hasattr(self, 'get_etot'):
                energy_data = self.get_etot(idx)
            else:
                log.warning(f"Parser does not implement get_etot method")

        log_info = None
        if hasattr(self, 'get_log_info') and callable(getattr(self, 'get_log_info')):
            try:
                log_info = self.get_log_info(idx)
            except Exception:
                log_info = None

        # Build frame index mapping for energy data if unconverged
        energy_frame_mapping = None
        if energy and energy_data is not None:
            unconverged_indices = energy_data.get(_keys.UNCONVERGED_FRAME_INDICES_KEY, [])
            if len(unconverged_indices) > 0:
                energy_frame_mapping = {}
                energy_idx = 0
                n_frames_total = structure[_keys.POSITIONS_KEY].shape[0]
                for frame_idx in range(n_frames_total):
                    if frame_idx not in unconverged_indices:
                        energy_frame_mapping[frame_idx] = energy_idx
                        energy_idx += 1

        # 如果开启了 Delta 模式，读取 H0
        if hamiltonian and is_delta_mode:
            if not hasattr(self, "get_h0_blocks"):
                raise NotImplementedError(
                    "hamiltonian_storage=['delta'] requested but parser does not implement get_h0_blocks().")
            ham0, _, _ = self.get_h0_blocks(idx, hamiltonian=True, overlap=False, density_matrix=False)

        n_frames = structure[_keys.POSITIONS_KEY].shape[0]
        lmdb_env = lmdb.open(out_dir, map_size=1048576000000, lock=True)

        for nf in range(n_frames):
            data_dict = {}
            data_dict[_keys.ATOMIC_NUMBERS_KEY] = structure[_keys.ATOMIC_NUMBERS_KEY]
            data_dict[_keys.CELL_KEY] = structure[_keys.CELL_KEY][nf]
            data_dict[_keys.POSITIONS_KEY] = structure[_keys.POSITIONS_KEY][nf]
            data_dict[_keys.PBC_KEY] = structure[_keys.PBC_KEY]

            if eigenvalue:
                data_dict[_keys.ENERGY_EIGENVALUE_KEY] = eigstatus[_keys.ENERGY_EIGENVALUE_KEY][nf]
                data_dict[_keys.KPOINT_KEY] = eigstatus[_keys.KPOINT_KEY]

            if energy and energy_data is not None:
                if energy_data[_keys.TOTAL_ENERGY_KEY].shape[0] == 1:
                    data_dict[_keys.TOTAL_ENERGY_KEY] = energy_data[_keys.TOTAL_ENERGY_KEY][0]
                else:
                    if energy_frame_mapping is not None:
                        if nf in energy_frame_mapping:
                            energy_idx = energy_frame_mapping[nf]
                            data_dict[_keys.TOTAL_ENERGY_KEY] = energy_data[_keys.TOTAL_ENERGY_KEY][energy_idx]
                    else:
                        data_dict[_keys.TOTAL_ENERGY_KEY] = energy_data[_keys.TOTAL_ENERGY_KEY][nf]

            if hamiltonian:
                if is_delta_mode:
                    h_full_frame = ham[nf]
                    h0_frame = ham0[nf]

                    delta_frame = {}
                    for key, full_block in h_full_frame.items():
                        if key not in h0_frame:
                            raise KeyError(f"H0 Hamiltonian is missing block {key} when computing delta H.")
                        delta_frame[key] = full_block - h0_frame[key]

                    data_dict["hamiltonian"] = delta_frame

                    if 'full' in storage_components:
                        data_dict["hamiltonian_full"] = h_full_frame
                    if 'h0' in storage_components:
                        data_dict["hamiltonian_0"] = h0_frame
                else:
                    data_dict["hamiltonian"] = ham[nf]

            if overlap:
                data_dict["overlap"] = ovp[nf]
            if density_matrix:
                data_dict["density_matrix"] = dm[nf]

            if log_info is not None:
                data_dict["iterations"] = log_info[0]
                data_dict["total_time"] = log_info[1]

            data_dict["idx"] = idx
            data_dict["nf"] = nf

            data_dict = pickle.dumps(data_dict)

            with lmdb_env.begin(write=True) as txn:
                entries = lmdb_env.stat()["entries"]
                txn.put(entries.to_bytes(length=4, byteorder='big'), data_dict)

        lmdb_env.close()