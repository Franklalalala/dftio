import re
import json
import numpy as np
from ase import Atoms, Atom
from dftio.data import _keys
from dftio.register import Register
from dftio.io.parse import Parser, ParserRegister
from dftio.io.gaussian.gaussian_tools import *
from dftio.io.gaussian.gaussian_conventions import orbital_sign_map
from emoles.fchk_parser import read_fchk  # 只保留 read_fchk，overlap 改为本地实现


def overlap_from_orthonormal(X, rcond=1e-10, return_pinv=True):
    """
    从 AO->OAO 变换矩阵 X 构造 AO 重叠矩阵 S。
    - 若 X 为非方阵（nbf x nif, nbf >= nif），使用 S = X^{+T} X^+（SVD 伪逆）。
    - 若 X 为方阵且条件数好，也可退化为精确逆的公式。
    返回:
      S: (nbf x nbf) 对称半正定、秩=nif 的重叠矩阵
      X_plus: (nif x nbf) 的 Moore-Penrose 伪逆（用于把 OAO 矩阵推回 AO）
    """
    X = np.asarray(X)
    n, m = X.shape

    # 统一走 SVD 路径，更稳健。遇到近线性相关会自动截断奇异值。
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if s.size == 0:
        raise ValueError("orthonormal_basis(X) seems empty.")
    cutoff = rcond * s.max()
    keep = s > cutoff
    if not np.any(keep):
        raise ValueError("orthonormal_basis(X) is numerically rank-deficient under rcond; no singular values kept.")

    U1 = U[:, keep]          # n x r
    s1 = s[keep]             # r
    V1 = Vt.T[:, keep]       # m x r

    # X^+ = V diag(1/s) U^T （截断后的最小范数解）
    X_plus = V1 @ (U1.T / s1[:, None])  # (m x r) @ (r x n) = m x n

    # S = X^{+T} X^+ = U1 diag(1/s^2) U1^T，同值但更省事也更稳
    inv_s_sq = (1.0 / s1)**2
    S = (U1 * inv_s_sq) @ U1.T
    S = 0.5 * (S + S.T)

    if return_pinv:
        return S, X_plus
    return S


def _invsqrt_symmetric(A, rcond=1e-12):
    """
    对称阵 A 的稳定逆平方根。用于 MO 的 S-正交白化。
    """
    w, U = np.linalg.eigh((A + A.T) * 0.5)
    cutoff = rcond * np.max(np.abs(w))
    keep = w > cutoff
    if not np.any(keep):
        raise ValueError("Matrix is (nearly) singular in invsqrt_symmetric.")
    W_inv_sqrt = U[:, keep] @ np.diag(1.0 / np.sqrt(w[keep])) @ U[:, keep].T
    return W_inv_sqrt


def _build_fock_from_mo_AO(S, M, eig, tol_orth=1e-6):
    """
    用 AO 基（M 为 nbf x nmo）上的 MO 系数与本征值构建 Fock：
      F = S M E M^T S
    若 M^T S M 与 I 偏差较大，则对白化 M（并相应相似变换 E），以提高数值稳定性。
    """
    k = eig.shape[0]
    if M.shape[1] != k:
        k = min(M.shape[1], k)
        M = M[:, :k]
        eig = eig[:k]
    E = np.diag(eig)

    G = M.T @ S @ M
    dev = np.linalg.norm(G - np.eye(k), ord='fro') / np.sqrt(k)
    if dev > tol_orth:
        # 白化并相应相似变换 E
        W = _invsqrt_symmetric(G)
        M = M @ W
        E = W.T @ E @ W

    F = S @ (M @ (E @ (M.T @ S)))
    F = 0.5 * (F + F.T)
    return F


def _to_ao_from_oao(mat_oao, X_plus):
    """
    把 OAO 基的矩阵（如 F_o、D_o）推回 AO：
      A_AO = X^{+T} A_OAO X^+
    """
    A = X_plus.T @ mat_oao @ X_plus
    return 0.5 * (A + A.T)


@ParserRegister.register("fchk_gaussian")
class GaussianFchkParser(Parser):
    def __init__(self, root, prefix, convention_file=None, valid_gau_info_path=None,
                 add_phase_transfer=False, orthonormal_rcond=1e-10, **kwargs):
        super(GaussianFchkParser, self).__init__(root, prefix)
        self.add_phase_transfer = add_phase_transfer
        self.is_fixed_convention = False
        self.orthonormal_rcond = orthonormal_rcond

        if convention_file:
            self.is_fixed_convention = True
            with open(convention_file, 'r') as f:
                self.convention = json.load(f)
        else:
            self.convention = None

        self.atomic_symbols = {}
        self.nbasis = {}
        self.raw_datas = []

        if valid_gau_info_path:
            self.raw_datas = get_gau_logs(valid_gau_info_path)
        else:
            # 若外部没有提供有效列表，就依赖 Parser 父类扫描到的文件清单（若有），并写出一份。
            if hasattr(self, 'raw_datas') and self.raw_datas:
                pass
            with open('valid_gaussian_logs.txt', 'w') as f:
                for a_raw_datapath in self.raw_datas:
                    f.write(a_raw_datapath + '\n')

    def get_structure(self, idx):
        file_path = self.raw_datas[idx]
        self.atoms = read_fchk(file_path)
        self.nbasis[idx] = self.atoms.info['fchk_meta']['nbf']
        self.atomic_symbols[idx] = self.atoms.symbols
        structure = {
            _keys.ATOMIC_NUMBERS_KEY: self.atoms.numbers,
            _keys.PBC_KEY: np.array([False, False, False]),
            _keys.POSITIONS_KEY: self.atoms.positions.reshape(1, -1, 3).astype(np.float32),
            _keys.CELL_KEY: self.atoms.cell.reshape(1,3,3).astype(np.float32)
        }
        return structure

    def get_eigenvalue(self, idx=None):
        mats = self.atoms.info.get('fchk_matrices', {})
        return {
            'alpha': mats.get('alpha_eigenvalues', None),
            'beta': mats.get('beta_eigenvalues', None)
        }

    def get_basis(self, idx):
        return self.atoms.info['convention']

    def get_blocks(self, idx, hamiltonian=True, overlap=False, density_matrix=False):
        # Cache basic info
        if idx not in self.nbasis.keys():
            self.nbasis[idx], atoms = self.atoms.info['fchk_meta']['nbf'], self.atoms
            self.atomic_symbols[idx] = self.atoms.symbols
        nbasis = self.nbasis[idx]
        atomic_symbols = self.atomic_symbols[idx]

        if self.convention is None:
            self.convention = self.atoms.info['convention']
        else:
            pass
            # print('using convention file')

        # Generate indices
        molecule_transform_indices, atom_in_mo_indices = generate_molecule_transform_indices(
            atom_types=atomic_symbols,
            atom_to_transform_indices=self.convention['atom_to_transform_indices']
        )

        # Get phase sign list if needed
        phase_sign_list = None
        if self.add_phase_transfer:
            atom_to_sorted_orbitals = convert_to_sorted_orbitals(self.convention['atom_to_dftio_orbitals'])
            phase_sign_list = get_phase_sign_list(
                atomic_symbols=atomic_symbols,
                atom_to_sorted_orbitals=atom_to_sorted_orbitals,
                orbital_sign_map=orbital_sign_map
            )

        mats = self.atoms.info['fchk_matrices']

        # 1) 构造/获取 S 与 X^+（尽量与 fchk 一致；若无 S 则从 X 重构）
        X = mats.get('orthonormal_basis', None)
        S_direct = mats.get('overlap', None)
        if S_direct is not None:
            S = 0.5 * (S_direct + S_direct.T)
            X_plus = None
        elif X is not None:
            S, X_plus = overlap_from_orthonormal(X, rcond=self.orthonormal_rcond, return_pinv=True)
        else:
            raise ValueError("Neither overlap nor orthonormal_basis found in fchk matrices.")

        results = []
        for matrix_type, should_compute in [
            ('ham', hamiltonian),
            ('overlap', overlap),
            ('density', density_matrix)
        ]:
            if not should_compute:
                results.append([None])
                continue

            if matrix_type == 'ham':
                mo = mats['alpha_mo']
                eig = mats['alpha_eigenvalues']
                # 根据 mo 行数判断是在 AO 还是 OAO 基
                n_S = S.shape[0]
                nbf_x = n_S
                nmo_rows = mo.shape[0]

                if nmo_rows == nbf_x:
                    # AO 基的 MO：F = S M E M^T S
                    matrix = _build_fock_from_mo_AO(S, mo, eig)
                else:
                    if X is None:
                        raise ValueError("alpha_mo does not match AO dimension and orthonormal_basis is missing.")
                    # OAO 基的 MO：F_o = M E M^T；再推回 AO：F = X^{+T} F_o X^+
                    # 先截断 eig
                    nmo = mo.shape[1]
                    if eig.shape[0] != nmo:
                        nuse = min(nmo, eig.shape[0])
                        mo = mo[:, :nuse]
                        eig = eig[:nuse]
                    F_oao = mo @ (np.diag(eig) @ mo.T)
                    if X_plus is None:
                        # 如果之前用的是 S 直接给定，则需要从 X 再求一次 X^+ 才能推回 AO
                        _, X_plus = overlap_from_orthonormal(X, rcond=self.orthonormal_rcond, return_pinv=True)
                    matrix = _to_ao_from_oao(F_oao, X_plus)

            elif matrix_type == 'overlap':
                matrix = S

            else:  # density
                D = mats.get('total_scf_density', None)
                if D is None:
                    # 若没有密度矩阵，可在需要时补：Closed-shell 近似 P = 2 * C_occ C_occ^T
                    # 这里保守起见返回 None，避免假设自旋/占据信息
                    matrix = None
                else:
                    nD = D.shape[0]
                    if nD == S.shape[0]:
                        matrix = 0.5 * (D + D.T)
                    else:
                        # 若密度在 OAO 基，则推回 AO
                        if X is None:
                            raise ValueError("Density matrix shape does not match AO and orthonormal_basis missing.")
                        if X_plus is None:
                            _, X_plus = overlap_from_orthonormal(X, rcond=self.orthonormal_rcond, return_pinv=True)
                        matrix = _to_ao_from_oao(D, X_plus)

            if matrix is not None:
                matrix = transform_matrix(matrix=matrix, transform_indices=molecule_transform_indices)
                if self.add_phase_transfer:
                    matrix = apply_phase_signs_to_matrix(matrix, phase_sign_list)
                result = cut_matrix(full_matrix=matrix, atom_in_mo_indices=atom_in_mo_indices)
            else:
                result = None

            results.append([result])

        return results