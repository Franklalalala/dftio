from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
import re
from tqdm import tqdm
from collections import Counter
from dftio.constants import orbitalId, ABACUS2DFTIO
import ase
from ase.io import read
import dpdata
import os
import numpy as np
from dftio.io.parse import Parser, ParserRegister, find_target_line
from dftio.data import _keys
from dftio.register import Register
import lmdb
import logging
import glob
import pickle
import shutil
import subprocess

log = logging.getLogger(__name__)


@ParserRegister.register("abacus")
class AbacusParser(Parser):
    def __init__(
            self,
            root,
            prefix,
            **kwargs
            ):
        super(AbacusParser, self).__init__(root, prefix)
        mode = self.get_mode(idx=0)
        if mode in ['nscf', "scf"]:
            self.raw_sys = [
                dpdata.System(
                    read(os.path.join(self._get_output_dir(idx), "STRU.cif")),
                    fmt="ase/structure"
                )
                for idx in range(len(self.raw_datas))
            ]
        else:
            self.raw_sys = [
                dpdata.LabeledSystem(self.raw_datas[idx], fmt='abacus/' + self.get_mode(idx))
                for idx in range(len(self.raw_datas))
            ]

    # ---------------------- 输出目录定位 ---------------------- #
    def _get_output_dir(self, idx):
        """
        优先使用 OUT.ABACUS；如果不存在，则匹配 OUT.* 中最新修改时间的目录
        """
        base_path = self.raw_datas[idx]

        out_abacus_path = os.path.join(base_path, "OUT.ABACUS")
        if os.path.exists(out_abacus_path):
            return out_abacus_path

        search_pattern = os.path.join(base_path, "OUT.*")
        out_dirs = glob.glob(search_pattern)
        out_dirs = [d for d in out_dirs if os.path.isdir(d)]

        if len(out_dirs) == 0:
            raise FileNotFoundError(
                f"No ABACUS output directory found in {base_path}. "
                f"Looked for OUT.ABACUS or OUT.* pattern."
            )
        elif len(out_dirs) == 1:
            log.info(f"Using output directory: {out_dirs[0]} (OUT.ABACUS not found, using OUT.* pattern)")
            return out_dirs[0]
        else:
            log.warning(f"Multiple output directories found: {out_dirs}. Using the most recently modified one.")
            out_dirs.sort(key=os.path.getmtime, reverse=True)
            return out_dirs[0]

    # ---------------------- 结构与模式 ---------------------- #
    def get_structure(self, idx):
        sys = self.raw_sys[idx]

        structure = {
            _keys.ATOMIC_NUMBERS_KEY: np.array(
                [ase.atom.atomic_numbers[i] for i in sys.data["atom_names"]],
                dtype=np.int32
            )[sys.data["atom_types"]],
            # abacus 不允许非 PBC
            _keys.PBC_KEY: np.array([True, True, True])
        }
        structure[_keys.POSITIONS_KEY] = sys.data["coords"].astype(np.float32)
        structure[_keys.CELL_KEY] = sys.data["cells"].astype(np.float32)

        return structure

    def get_mode(self, idx):
        output_dir = self._get_output_dir(idx)
        with open(os.path.join(output_dir, "INPUT"), 'r') as f:
            line = find_target_line(f, "calculation")
            assert line is not None, 'Cannot find "calculation" in INPUT file'
            mode = line.split()[1]
        return mode

    def get_log_info(self, idx):
        """
        从 run.log 解析：
        - SCF 迭代轮数（GE1, GE2, ...）
        - TOTAL Time
        """
        iterations = 0
        total_time = 0.0

        runlog = os.path.join(self.raw_datas[idx], "run.log")
        if not os.path.exists(runlog):
            return iterations, total_time

        with open(runlog, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("GE"):
                    parts = line.split()
                    step_str = parts[0][2:]
                    if step_str.isdigit():
                        iterations = int(step_str)

                elif "TOTAL  Time" in line and ":" in line:
                    try:
                        time_str = line.split(":")[-1]
                        total_time = float(time_str)
                    except ValueError:
                        pass

        return iterations, total_time

    # ---------------------- 本征值 ---------------------- #
    def get_eigenvalue(self, idx, band_index_min=0):
        mode = self.get_mode(idx)
        output_dir = self._get_output_dir(idx)

        if mode not in ["scf", "nscf"]:
            if mode in ["md", "relax"]:
                raise NotImplementedError("output eigenvalues from MD / relax trajectory is not supported by ABACUS.")
            else:
                raise NotImplementedError(f"mode {mode} is not supported.")

        # 能带文件：优先 band.txt，其次 BANDS_1.dat
        band_candidates = ["band.txt", "BANDS_1.dat"]
        band_path = None
        for name in band_candidates:
            path = os.path.join(output_dir, name)
            if os.path.exists(path):
                band_path = path
                break
        if band_path is None:
            raise FileNotFoundError(
                f"Cannot find band file in {output_dir}. Tried: {band_candidates}"
            )

        eigs_raw = np.loadtxt(band_path)
        eigs = eigs_raw[np.newaxis, :, 2 + band_index_min:]

        # k 点文件：优先 KPT.info，其次 kpoints
        kpt_candidates = ["KPT.info", "kpoints"]
        kpt_path = None
        for name in kpt_candidates:
            path = os.path.join(output_dir, name)
            if os.path.exists(path):
                kpt_path = path
                break
        if kpt_path is None:
            raise FileNotFoundError(
                f"Cannot find k-point file in {output_dir}. Tried: {kpt_candidates}"
            )

        kpts = []
        with open(kpt_path, "r") as f:
            line = find_target_line(f, "nkstot now")
            assert line is not None, "Cannot find 'nkstot now' in k-point file"
            nkstot = int(line.strip().split()[-1])

            line = find_target_line(f, "KPOINTS ")
            if not line:
                f.seek(0)
                line = find_target_line(f, "KPT ")
            if not line:
                raise Exception("Cannot find KPT or KPOINTS in k-point file")

            for _ in range(nkstot):
                line = f.readline().strip().split()
                kpts.append([float(line[1]), float(line[2]), float(line[3])])
        kpts = np.array(kpts)

        return {
            _keys.ENERGY_EIGENVALUE_KEY: eigs.astype(np.float32),
            _keys.KPOINT_KEY: kpts.astype(np.float32)
        }

    # ---------------------- 基组信息 ---------------------- #
    def get_basis(self, idx):
        """
        返回类似 {"Si": "2s2p1d"} 的基组描述
        """
        mode = self.get_mode(idx)
        logfile = "running_" + mode + ".log"
        output_dir = self._get_output_dir(idx)
        sys = self.raw_sys[idx]

        orbital_types_dict = {}
        with open(os.path.join(output_dir, logfile), 'r') as f:
            for index_type in range(len(sys.data["atom_numbs"])):
                tmp = find_target_line(f, "READING ATOM TYPE")
                assert tmp is not None, 'Cannot find "ATOM TYPE" in log file'
                assert tmp.split()[-1] == str(index_type + 1)

                line = f.readline()
                assert "atom label =" in line.lower()
                atom_label = line.split()[-1]
                atom_label = ''.join(re.findall(r'[A-Za-z]', atom_label))
                assert atom_label in ase.data.atomic_numbers, "Atom label should be in periodic table"

                current_orbital_types = []
                while True:
                    line = f.readline()
                    if "number of zeta" in line:
                        tmp = line.split()
                        L = int(tmp[0][2:-1])
                        num_L = int(tmp[-1])
                        current_orbital_types.extend([L] * num_L)
                    else:
                        break
                orbital_types_dict[atom_label] = current_orbital_types

        basis = {}
        for k, v in orbital_types_dict.items():
            counter = Counter(v)
            basis[k] = [
                str(counter[l]) + orbitalId[l]
                for l in range(max(counter.keys()) + 1)
                if counter.get(l, 0) > 0
            ]
            basis[k] = "".join(basis[k])
        return basis

    # ---------------------- 内部工具：原子标签解析 ---------------------- #
    @staticmethod
    def _parse_atom_label_from_coord_token(token: str) -> str:
        """
        从 COORDINATES 段中出现的 token 抽取元素符号。
        兼容形式：
          - "Si"
          - "Zn1"
          - "tau1_Zn"
          - "taucZn" 等
        """
        # 优先用大写开头 + 可选小写的原子符号模式
        candidates = re.findall(r'[A-Z][a-z]?', token)
        for symbol in reversed(candidates):
            if symbol in ase.data.atomic_numbers:
                return symbol

        # 如果没有或都不是合法元素，再做一次保守处理
        if candidates:
            cand = candidates[-1]
        else:
            letters = re.findall(r'[A-Za-z]+', token)
            if not letters:
                raise ValueError(f"Cannot extract element symbol from token '{token}'")
            cand = letters[-1]

        for L in (2, 1):
            if len(cand) >= L:
                symbol = cand[-L:].title()
                if symbol in ase.data.atomic_numbers:
                    return symbol

        raise ValueError(f"Cannot map token '{token}' to a valid element symbol")

    # ---------------------- 内部工具：轨道与自旋信息 ---------------------- #
    def _get_orbital_and_spin_info(self, idx):
        """
        从 running_*.log 里解析：
        - 每个原子类型的轨道数与角动量类型 (site_norbits_dict, orbital_types_dict)
        - 每个 site 的轨道数 (site_norbits)
        - 每个 site 的元素 (element 原子序数)
        - NSPIN, 并判断是否为 SOC (spinful=True 表示 nspin==4)
        """
        mode = self.get_mode(idx)
        logfile = "running_" + mode + ".log"
        output_dir = self._get_output_dir(idx)
        sys = self.raw_sys[idx]
        nsites = sys.data["atom_types"].shape[0]

        log_path = os.path.join(output_dir, logfile)

        site_norbits_dict = {}
        orbital_types_dict = {}
        site_norbits = np.zeros(nsites, dtype=int)
        element = np.zeros(nsites, dtype=int)
        spinful = False

        with open(log_path, 'r') as f:
            # 1) 原子类型基组
            for index_type in range(len(sys.data["atom_numbs"])):
                tmp = find_target_line(f, "READING ATOM TYPE")
                assert tmp is not None, 'Cannot find "ATOM TYPE" in log file'
                assert tmp.split()[-1] == str(index_type + 1)

                line = f.readline()
                assert "atom label =" in line.lower()
                atom_label = line.split()[-1]
                atom_label = ''.join(re.findall(r'[A-Za-z]', atom_label))
                assert atom_label in ase.data.atomic_numbers, "Atom label should be in periodic table"
                atom_type = ase.data.atomic_numbers[atom_label]

                current_site_norbits = 0
                current_orbital_types = []
                while True:
                    line = f.readline()
                    if "number of zeta" in line:
                        tmp = line.split()
                        L = int(tmp[0][2:-1])
                        num_L = int(tmp[-1])
                        current_site_norbits += (2 * L + 1) * num_L
                        current_orbital_types.extend([L] * num_L)
                    else:
                        break
                site_norbits_dict[atom_type] = current_site_norbits
                orbital_types_dict[atom_type] = current_orbital_types

            # 2) COORDINATES 区，建立 site -> 元素映射
            line = find_target_line(f, " COORDINATES")
            header = f.readline().lower()
            assert "atom" in header, "Cannot find atom header after COORDINATES"

            for index_site in range(nsites):
                line = f.readline()
                tmp = line.split()
                if not tmp:
                    raise ValueError(f"Unexpected empty line when reading COORDINATES, index_site={index_site}")
                token = tmp[0]
                atom_label = self._parse_atom_label_from_coord_token(token)
                assert atom_label in ase.data.atomic_numbers, f"Atom label {atom_label} should be in periodic table"
                element[index_site] = ase.data.atomic_numbers[atom_label]
                site_norbits[index_site] = site_norbits_dict[element[index_site]]

            # 3) 自旋信息 NSPIN
            f.seek(0)
            line = find_target_line(f, "nspin")
            if line is None:
                line = find_target_line(f, "NSPIN")

            assert line is not None, 'Cannot find "NSPIN" in log file'
            line_lower = line.lower()

            if ("nspin == 1" in line_lower) or ("nspin = 1" in line_lower):
                spinful = False
            elif ("nspin == 4" in line_lower) or ("nspin = 4" in line_lower):
                spinful = True
            elif ("nspin == 2" in line_lower) or ("nspin = 2" in line_lower):
                # 共线自旋，矩阵通常为实数，这里不当作 SOC
                spinful = False
            else:
                # 兜底再尝试直接解析数值
                try:
                    val = int(line.split()[-1])
                    spinful = (val == 4)
                except Exception:
                    raise ValueError(f'{line} is not supported')

        return mode, sys, nsites, site_norbits, element, orbital_types_dict, spinful

    # ---------------------- 内部工具：静态矩阵文件查找 ---------------------- #
    def _find_static_matrix_file(self, idx, kind, spin_index=0, h0=False):
        """
        kind: 'H' | 'S' | 'DM'
        - 旧命名: hrs1_nao.csr / srs1_nao.csr / dmrs1_nao.csr
        - 新命名: data-HR-sparse_SPIN0.csr / data-SR-sparse_SPIN0.csr / data-DMR-sparse_SPIN0.csr
        - H0: data-HR0_SPIN0.csr
        """
        output_dir = self._get_output_dir(idx)
        candidates = []

        if h0:
            if kind != "H":
                raise NotImplementedError("Only Hamiltonian H0 is supported for now.")
            candidates.append(f"data-HR0_SPIN{spin_index}.csr")
            candidates.append("hrs0_nao.csr")
        else:
            if kind == "H":
                candidates.extend([
                    "hrs1_nao.csr",
                    f"data-HR-sparse_SPIN{spin_index}.csr",
                    f"data-HR_SPIN{spin_index}.csr",
                ])
            elif kind == "S":
                candidates.extend([
                    "srs1_nao.csr",
                    f"data-SR-sparse_SPIN{spin_index}.csr",
                    f"data-SR_SPIN{spin_index}.csr",
                ])
            elif kind == "DM":
                candidates.extend([
                    "dmrs1_nao.csr",
                    f"data-DMR-sparse_SPIN{spin_index}.csr",
                    f"data-DMR_SPIN{spin_index}.csr",
                ])
            else:
                raise ValueError(f"Unknown matrix kind: {kind}")

        for name in candidates:
            path = os.path.join(output_dir, name)
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"Cannot find {kind} matrix file for idx={idx} in {output_dir}. "
            f"Tried: {candidates}"
        )

    # ---------------------- 内部工具：MD 矩阵文件查找 ---------------------- #
    def _find_md_matrix_file(self, idx, kind, frame, spin_index=0):
        """
        MD 轨迹中按帧存放在 matrix/ 子目录下的 csr:
        e.g. matrix/0_data-HR-sparse_SPIN0.csr 或 matrix/0_hrs1_nao.csr
        """
        output_dir = self._get_output_dir(idx)
        matrix_dir = os.path.join(output_dir, "matrix")

        candidates = []
        if kind == "H":
            candidates.extend([
                os.path.join(matrix_dir, f"{frame}_data-HR-sparse_SPIN{spin_index}.csr"),
                os.path.join(matrix_dir, f"{frame}_hrs1_nao.csr"),
                os.path.join(matrix_dir, f"{frame}_HR-sparse_SPIN{spin_index}.csr"),
            ])
        elif kind == "S":
            candidates.extend([
                os.path.join(matrix_dir, f"{frame}_data-SR-sparse_SPIN{spin_index}.csr"),
                os.path.join(matrix_dir, f"{frame}_srs1_nao.csr"),
                os.path.join(matrix_dir, f"{frame}_SR-sparse_SPIN{spin_index}.csr"),
            ])
        elif kind == "DM":
            candidates.extend([
                os.path.join(matrix_dir, f"{frame}_data-DMR-sparse_SPIN{spin_index}.csr"),
                os.path.join(matrix_dir, f"{frame}_dmrs1_nao.csr"),
                os.path.join(matrix_dir, f"{frame}_DMR-sparse_SPIN{spin_index}.csr"),
            ])
        else:
            raise ValueError(f"Unknown matrix kind: {kind}")

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"Cannot find {kind} MD matrix file for idx={idx}, frame={frame} in {matrix_dir}. "
            f"Tried: {candidates}"
        )

    # ---------------------- H / S / DM Blocks（最终收敛 H） ---------------------- #
    def get_blocks(self, idx, hamiltonian=True, overlap=False, density_matrix=False):
        """
        读取最终收敛 H / S / DM：
        - 静态 (scf/nscf)：兼容旧 / 新 csr 文件名
        - MD：兼容 matrix/ 子目录逐帧文件 & 单文件多 STEP 格式
        """
        output_dir = self._get_output_dir(idx)
        mode, sys, nsites, site_norbits, element, orbital_types_dict, spinful = \
            self._get_orbital_and_spin_info(idx)

        # 如有 hscsr.tgz 则先解压到计算根目录
        hscsr_tgz = os.path.join(output_dir, "hscsr.tgz")
        if os.path.exists(hscsr_tgz):
            subprocess.run(
                ["tar", "-xzf", hscsr_tgz, "-C", self.raw_datas[idx]],
                check=True
            )

        hamiltonian_dict, overlap_dict, density_matrix_dict = None, None, None

        # ---------- 静态：scf / nscf ---------- #
        if mode in ["scf", "nscf"]:
            if hamiltonian:
                matrix_path = self._find_static_matrix_file(idx, kind="H", spin_index=0, h0=False)
                hamiltonian_dict, tmp = self.parse_matrix(
                    matrix_path=matrix_path,
                    nsites=nsites,
                    site_norbits=site_norbits,
                    orbital_types_dict=orbital_types_dict,
                    element=element,
                    factor=13.605698,  # Ryd->eV
                    spinful=spinful,
                    step=None
                )
                assert tmp == int(np.sum(site_norbits)) * (1 + spinful)
                hamiltonian_dict = [hamiltonian_dict]

            if overlap:
                matrix_path = self._find_static_matrix_file(idx, kind="S", spin_index=0, h0=False)
                ovp_dict, tmp = self.parse_matrix(
                    matrix_path=matrix_path,
                    nsites=nsites,
                    site_norbits=site_norbits,
                    orbital_types_dict=orbital_types_dict,
                    element=element,
                    factor=1.0,
                    spinful=spinful,
                    step=None
                )
                assert tmp == int(np.sum(site_norbits)) * (1 + spinful)

                # SOC 情况下，沿用旧逻辑：截半并取实部作为返回结果
                if spinful:
                    ovp_spinless = {}
                    for k, v in ovp_dict.items():
                        ovp_spinless[k] = v[:v.shape[0] // 2, :v.shape[1] // 2].real
                    ovp_dict = ovp_spinless

                overlap_dict = [ovp_dict]

            if density_matrix:
                matrix_path = self._find_static_matrix_file(idx, kind="DM", spin_index=0, h0=False)
                density_matrix_dict, tmp = self.parse_matrix(
                    matrix_path=matrix_path,
                    nsites=nsites,
                    site_norbits=site_norbits,
                    orbital_types_dict=orbital_types_dict,
                    element=element,
                    factor=1.0,
                    spinful=spinful,
                    step=None
                )
                assert tmp == int(np.sum(site_norbits)) * (1 + spinful)
                density_matrix_dict = [density_matrix_dict]

        # ---------- MD 模式 ---------- #
        elif mode == "md":
            nframes = sys.get_nframes()
            matrix_dir = os.path.join(output_dir, "matrix")
            has_matrix_dir = os.path.isdir(matrix_dir)

            if hamiltonian:
                hamiltonian_dict = []
                for i in range(nframes):
                    if has_matrix_dir:
                        matrix_path = self._find_md_matrix_file(idx, kind="H", frame=i, spin_index=0)
                        step_arg = None
                    else:
                        matrix_path = self._find_static_matrix_file(idx, kind="H", spin_index=0, h0=False)
                        step_arg = i

                    hamil, tmp = self.parse_matrix(
                        matrix_path=matrix_path,
                        nsites=nsites,
                        site_norbits=site_norbits,
                        orbital_types_dict=orbital_types_dict,
                        element=element,
                        factor=13.605698,
                        spinful=spinful,
                        step=step_arg
                    )
                    assert tmp == int(np.sum(site_norbits)) * (1 + spinful)
                    hamiltonian_dict.append(hamil)

            if overlap:
                overlap_dict = []
                for i in range(nframes):
                    if has_matrix_dir:
                        matrix_path = self._find_md_matrix_file(idx, kind="S", frame=i, spin_index=0)
                        step_arg = None
                    else:
                        matrix_path = self._find_static_matrix_file(idx, kind="S", spin_index=0, h0=False)
                        step_arg = i

                    ovp, tmp = self.parse_matrix(
                        matrix_path=matrix_path,
                        nsites=nsites,
                        site_norbits=site_norbits,
                        orbital_types_dict=orbital_types_dict,
                        element=element,
                        factor=1.0,
                        spinful=spinful,
                        step=step_arg
                    )
                    assert tmp == int(np.sum(site_norbits)) * (1 + spinful)

                    if spinful:
                        ovp_spinless = {}
                        for k, v in ovp.items():
                            ovp_spinless[k] = v[:v.shape[0] // 2, :v.shape[1] // 2].real
                        ovp = ovp_spinless
                    overlap_dict.append(ovp)

            if density_matrix:
                density_matrix_dict = []
                for i in range(nframes):
                    if has_matrix_dir:
                        matrix_path = self._find_md_matrix_file(idx, kind="DM", frame=i, spin_index=0)
                        step_arg = None
                    else:
                        matrix_path = self._find_static_matrix_file(idx, kind="DM", spin_index=0, h0=False)
                        step_arg = i

                    dm, tmp = self.parse_matrix(
                        matrix_path=matrix_path,
                        nsites=nsites,
                        site_norbits=site_norbits,
                        orbital_types_dict=orbital_types_dict,
                        element=element,
                        factor=1.0,
                        spinful=spinful,
                        step=step_arg
                    )
                    assert tmp == int(np.sum(site_norbits)) * (1 + spinful)
                    density_matrix_dict.append(dm)

        else:
            raise NotImplementedError(f"mode {mode} is not supported.")

        return hamiltonian_dict, overlap_dict, density_matrix_dict

    # ---------------------- 新增：H0 Blocks（第一步 H0） ---------------------- #
    def get_h0_blocks(self, idx, hamiltonian=True, overlap=False, density_matrix=False):
        """
        使用同目录下 data-HR0_SPIN0.csr 读取 H0。
        仅实现 Hamiltonian H0，其它暂不支持。
        返回形式：list[dict]，静态计算时长度为 1
        """
        if overlap or density_matrix:
            raise NotImplementedError("H0 for overlap / density_matrix is not supported currently.")
        if not hamiltonian:
            return None, None, None

        output_dir = self._get_output_dir(idx)
        mode, sys, nsites, site_norbits, element, orbital_types_dict, spinful = \
            self._get_orbital_and_spin_info(idx)

        # 如有 hscsr.tgz 则先解压
        hscsr_tgz = os.path.join(output_dir, "hscsr.tgz")
        if os.path.exists(hscsr_tgz):
            subprocess.run(
                ["tar", "-xzf", hscsr_tgz, "-C", self.raw_datas[idx]],
                check=True
            )

        if mode not in ["scf", "nscf"]:
            raise NotImplementedError(f"get_h0_blocks is only supported for scf/nscf, got mode={mode}")

        matrix_path = self._find_static_matrix_file(idx, kind="H", spin_index=0, h0=True)
        h0_dict, tmp = self.parse_matrix(
            matrix_path=matrix_path,
            nsites=nsites,
            site_norbits=site_norbits,
            orbital_types_dict=orbital_types_dict,
            element=element,
            factor=13.605698,
            spinful=spinful,
            step=None
        )
        assert tmp == int(np.sum(site_norbits)) * (1 + spinful)

        return [h0_dict], None, None

    # ---------------------- CSR 解析与轨道变换 ---------------------- #
    def parse_matrix(self, matrix_path, nsites, site_norbits, orbital_types_dict,
                     element, factor, spinful=False, step=None):
        """
        通用 CSR 解析：
        - 支持无 STEP 的静态文件
        - 支持带 STEP 的 MD 单文件（通过 step 参数选择第几步）
        """
        site_norbits_cumsum = np.cumsum(site_norbits)
        matrix_dict = {}

        with open(matrix_path, 'r') as f:
            lines = f.readlines()

        # ------ 定位 "Matrix Dimension of" 行 ------ #
        dim_line_idx = None

        if step is not None:
            # 先试着按 STEP 定位对应块
            step_line_idx = None
            for i, line in enumerate(lines):
                if "STEP" in line:
                    parts = line.split()
                    try:
                        stp = int(parts[-1])
                    except Exception:
                        continue
                    if stp == step:
                        step_line_idx = i
                        break
            if step_line_idx is not None:
                for j in range(step_line_idx + 1, len(lines)):
                    if "Matrix Dimension of" in lines[j]:
                        dim_line_idx = j
                        break

        if dim_line_idx is None:
            # 静态文件 / 或找不到 STEP：从头搜索第一个 "Matrix Dimension of"
            for i, line in enumerate(lines):
                if "Matrix Dimension of" in line:
                    dim_line_idx = i
                    break

        if dim_line_idx is None:
            raise ValueError(f"Cannot find 'Matrix Dimension of' in {matrix_path}")

        dim_line = lines[dim_line_idx]
        try:
            norbits = int(dim_line.split()[-1])
        except Exception as e:
            raise ValueError(f"Cannot parse matrix dimension from line '{dim_line}' in {matrix_path}") from e

        # 下一行是 "Matrix number of ..."，从再下一行开始是数据
        i = dim_line_idx + 2

        while i < len(lines):
            line = lines[i]
            line1 = line.split()
            if len(line1) == 0 or len(line1) == 2:
                # 原版代码以此条件作为结束标记，保持兼容
                break

            num_element = int(line1[3])
            if num_element != 0:
                R_cur = np.array(line1[:3]).astype(int)

                if i + 3 >= len(lines):
                    break

                line2 = lines[i + 1].split()
                line3 = lines[i + 2].split()
                line4 = lines[i + 3].split()

                if not spinful:
                    data_vals = np.array(line2).astype(np.float32)
                    row_idx = np.array(line3).astype(int)
                    indptr = np.array(line4).astype(np.int32)
                    mat_full = csr_matrix(
                        (data_vals, row_idx, indptr),
                        shape=(norbits, norbits),
                        dtype=np.float32
                    ).toarray()
                else:
                    data_vals = np.char.replace(line2, '(', '')
                    data_vals = np.char.replace(data_vals, ')', 'j')
                    data_vals = np.char.replace(data_vals, ',', '+')
                    data_vals = np.char.replace(data_vals, '+-', '-')
                    data_vals = np.array(data_vals).astype(np.complex64)
                    row_idx = np.array(line3).astype(int)
                    indptr = np.array(line4).astype(np.int32)
                    mat_full = csr_matrix(
                        (data_vals, row_idx, indptr),
                        shape=(norbits, norbits),
                        dtype=np.complex64
                    ).toarray()

                for index_site_i in range(nsites):
                    for index_site_j in range(nsites):
                        key_str = f"{index_site_i}_{index_site_j}_{R_cur[0]}_{R_cur[1]}_{R_cur[2]}"
                        i_start = (site_norbits_cumsum[index_site_i] - site_norbits[index_site_i]) * (1 + spinful)
                        i_end = site_norbits_cumsum[index_site_i] * (1 + spinful)
                        j_start = (site_norbits_cumsum[index_site_j] - site_norbits[index_site_j]) * (1 + spinful)
                        j_end = site_norbits_cumsum[index_site_j] * (1 + spinful)

                        mat = mat_full[i_start:i_end, j_start:j_end]
                        if np.abs(mat).max() < 1e-10:
                            continue

                        if not spinful:
                            mat = self.transform(
                                mat,
                                orbital_types_dict[element[index_site_i]],
                                orbital_types_dict[element[index_site_j]]
                            )
                        else:
                            mat = mat.reshape(
                                (site_norbits[index_site_i], 2,
                                 site_norbits[index_site_j], 2)
                            )
                            mat = mat.transpose((1, 0, 3, 2)).reshape(
                                (2 * site_norbits[index_site_i],
                                 2 * site_norbits[index_site_j])
                            )
                            mat = self.transform(
                                mat,
                                orbital_types_dict[element[index_site_i]] * 2,
                                orbital_types_dict[element[index_site_j]] * 2
                            )

                        matrix_dict[key_str] = mat * factor

                i += 4
            else:
                i += 1

        return matrix_dict, norbits

    def transform(self, mat, l_lefts, l_rights):
        if max(*l_lefts, *l_rights) > 5:
            raise NotImplementedError("Only support l = s, p, d, f, g, h.")
        block_lefts = block_diag(*[ABACUS2DFTIO[l_left] for l_left in l_lefts])
        block_rights = block_diag(*[ABACUS2DFTIO[l_right] for l_right in l_rights])

        return block_lefts @ mat @ block_rights.T

    # ---------------------- Etot 提取 ---------------------- #
    @staticmethod
    def _extract_energy_from_log(loglines, mode, dump_freq=1):
        """
        从 ABACUS log 提取总能：
        - SCF/NSCF：最终一帧
        - MD：按 dump 频率提取，过滤未收敛帧
        - RELAX：所有收敛结构
        只在 ABACUS 3.9.0 测过，其他版本请谨慎。
        """
        energy = []

        if mode in ["scf", "nscf"]:
            for line in reversed(loglines):
                if "final etot is" in line:
                    Etot = float(line.split()[-2])
                    return (np.array([Etot], dtype=np.float64), [])
                elif "TOTAL ENERGY" in line:
                    Etot = float(line.split()[-2])
                    return (np.array([Etot], dtype=np.float64), [])
                elif "convergence has NOT been achieved!" in line or \
                     "convergence has not been achieved" in line:
                    return None
            return None

        elif mode == "md":
            nenergy = 0
            for line in loglines:
                if "final etot is" in line:
                    if nenergy % dump_freq == 0:
                        energy.append(float(line.split()[-2]))
                    nenergy += 1
                elif "!! convergence has not been achieved" in line:
                    if nenergy % dump_freq == 0:
                        energy.append(np.nan)
                    nenergy += 1

            if len(energy) == 0:
                return None

            energy = np.array(energy, dtype=np.float64)
            valid_mask = ~np.isnan(energy)
            if not valid_mask.any():
                return None

            unconverged_indices = np.where(~valid_mask)[0].tolist()
            energy = energy[valid_mask]
            return (energy, unconverged_indices)

        elif mode == "relax":
            relax_success = False
            for line in loglines:
                if "Relaxation is converged!" in line:
                    relax_success = True
                if "!FINAL_ETOT_IS" in line:
                    energy.append(float(line.split()[-2]))

            if relax_success and len(energy) > 0:
                return (np.array(energy, dtype=np.float64), [])
            else:
                raise ValueError("Relaxation did not converge or no energies found.")

        else:
            return None

    def get_etot(self, idx):
        """
        从 running_*.log 提取 Etot。
        返回:
        {
          _keys.TOTAL_ENERGY_KEY: 能量数组 (eV),
          _keys.UNCONVERGED_FRAME_INDICES_KEY: 未收敛帧索引列表
        }
        """
        mode = self.get_mode(idx)
        logfile = "running_" + mode + ".log"
        logpath = os.path.join(self._get_output_dir(idx), logfile)

        if not os.path.exists(logpath):
            raise FileNotFoundError(f"Log file {logpath} does not exist.")

        with open(logpath, 'r') as f:
            loglines = f.readlines()

        dump_freq = 1
        if mode == "md":
            input_path = os.path.join(self.raw_datas[idx], "INPUT")
            if os.path.exists(input_path):
                with open(input_path, 'r') as f_in:
                    for line in f_in:
                        if len(line) > 0 and "md_dumpfreq" in line and "md_dumpfreq" == line.split()[0]:
                            dump_freq = int(line.split()[1])
                            break

        result = self._extract_energy_from_log(loglines, mode, dump_freq)

        if result is None:
            return None

        energy, unconverged_indices = result

        if len(unconverged_indices) > 0:
            log.warning(f"Energy extraction: frames {unconverged_indices} did not converge")

        return {
            _keys.TOTAL_ENERGY_KEY: energy,
            _keys.UNCONVERGED_FRAME_INDICES_KEY: unconverged_indices
        }

    # ---------------------- H0 目录映射 + LMDB 后处理 ---------------------- #
    def get_abs_h0_folders(self, h0_root):
        folder_path_map = {}
        for sub_root, dirs, _ in os.walk(h0_root):
            for dirname in dirs:
                folder_path_map[dirname] = os.path.join(sub_root, dirname)

        abs_h0_folders = []
        valid_idx_list = []
        for valid_idx, a_H_folder in enumerate(self.raw_datas):
            a_leaf_folder_name = os.path.split(a_H_folder)[-1]
            a_leaf_folder_path = folder_path_map.get(a_leaf_folder_name)
            found_flag = a_leaf_folder_path is not None

            if found_flag:
                abs_h0_folders.append(a_leaf_folder_path)
                valid_idx_list.append(valid_idx)
            else:
                abs_h0_folders.append(None)
                valid_idx_list.append(None)
        return abs_h0_folders, valid_idx_list

    def add_h0_delta_h(self, h0_src_root, old_lmdb_path, new_lmdb_path,
                       keep_old_lmdb: bool = True,
                       keep_delta_ham_only: bool = True):
        """
        旧逻辑：H 与 H0 分别来自两套 DFT 输出目录，通过 LMDB 做后处理：
        - 读 old_lmdb 中已有的 full H
        - 用当前 parser 对 h0_root 下对应结构调用 get_blocks() 得到 H0
        - 存 delta H (以及可选 full H / H0) 到 new_lmdb
        """
        h0_root = os.path.abspath(h0_src_root)
        self.raw_datas, valid_idx_list = self.get_abs_h0_folders(h0_root=h0_root)

        os.makedirs(new_lmdb_path, exist_ok=True)
        counter = 0
        batch_size = 50

        for batch_start in tqdm(range(0, len(valid_idx_list), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(valid_idx_list))
            batch_indices = valid_idx_list[batch_start:batch_end]

            old_db_env = lmdb.open(old_lmdb_path, readonly=True, lock=False)
            new_db_env = lmdb.open(new_lmdb_path, map_size=1048576000000, lock=True)

            with old_db_env.begin() as old_txn, new_db_env.begin(write=True) as new_txn:
                for idx in batch_indices:
                    if idx is None:
                        continue

                    data_dict = old_txn.get(idx.to_bytes(length=4, byteorder='big'))
                    data_dict = pickle.loads(data_dict)

                    h0_block, _, _ = self.get_blocks(idx, hamiltonian=True, overlap=False, density_matrix=False)
                    h0_block = h0_block[0]
                    old_ham_block = data_dict['hamiltonian']

                    delta_block = {}
                    for a_block_name in old_ham_block.keys():
                        a_delta_block = old_ham_block[a_block_name] - h0_block[a_block_name]
                        delta_block[a_block_name] = a_delta_block

                    data_dict['hamiltonian'] = delta_block
                    if not keep_delta_ham_only:
                        data_dict['hamiltonian_full'] = old_ham_block
                        data_dict['hamiltonian_0'] = h0_block

                    data_bytes = pickle.dumps(data_dict)
                    new_txn.put(counter.to_bytes(length=4, byteorder='big'), data_bytes)
                    counter += 1

            old_db_env.close()
            new_db_env.close()

        if not keep_old_lmdb:
            shutil.rmtree(old_lmdb_path)