"""Build cyclo(Gly-Val-Pro-Val-Trp-Ala) (segetalin A) with all-trans peptide bonds."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms as T

SEQ = "GVPVWA"
lin = Chem.MolFromSequence(SEQ)
info = lambda a: a.GetPDBResidueInfo()
find = lambda res, name: next(a.GetIdx() for a in lin.GetAtoms()
                              if info(a).GetResidueNumber() == res and info(a).GetName().strip() == name)
n1, c6, oxt = find(1, "N"), find(len(SEQ), "C"), find(len(SEQ), "OXT")
rw = Chem.RWMol(lin)
rw.AddBond(c6, n1, Chem.BondType.SINGLE)
rw.RemoveAtom(oxt)
cyc = rw.GetMol()
Chem.SanitizeMol(cyc)
molH = Chem.AddHs(cyc, addCoords=False)

p = AllChem.ETKDGv3(); p.useMacrocycleTorsions = True; p.randomSeed = 7
cids = AllChem.EmbedMultipleConfs(molH, 60, p)
AllChem.MMFFOptimizeMoleculeConfs(molH, maxIters=2000)
props = AllChem.MMFFGetMoleculeProperties(molH)

def omegas(conf):
    at = lambda r, n: next(a.GetIdx() for a in molH.GetAtoms() if a.GetPDBResidueInfo()
                           and a.GetPDBResidueInfo().GetResidueNumber() == r and a.GetPDBResidueInfo().GetName().strip() == n)
    res = range(1, len(SEQ) + 1)
    nxt = lambda r: r % len(SEQ) + 1
    return [T.GetDihedralDeg(conf, at(r, "CA"), at(r, "C"), at(nxt(r), "N"), at(nxt(r), "CA")) for r in res]

energy = lambda cid: AllChem.MMFFGetMoleculeForceField(molH, props, confId=cid).CalcEnergy()
trans = [c for c in cids if all(abs(w) > 150 for w in omegas(molH.GetConformer(c)))]
best = min(trans, key=energy)
print(f"{len(trans)}/{len(cids)} all-trans conformers; best {best} E={energy(best):.1f} kcal/mol",
      "omegas", np.round(omegas(molH.GetConformer(best))).tolist())
heavy = Chem.RemoveHs(molH)
Chem.MolToPDBFile(heavy, "segetalinA_heavy.pdb", confId=best)
