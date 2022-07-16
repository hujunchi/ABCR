from os import path
from copy import copy
from functools import reduce
from multiprocessing.dummy import Array
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

#Single bonds and rotatable bonds
def GetSingleBonds(self):
    return filter(lambda v: v.GetBondType() == Chem.rdchem.BondType.SINGLE, self.GetBonds())
def isRotatableBond(self):
    return not self.IsInRing() and self.GetBeginAtom().GetDegree() != 1 and self.GetEndAtom().GetDegree() != 1
def GetRotatableBonds(self):
    return list(filter(lambda v: v.isRotatableBond(), self.GetSingleBonds()))

Chem.Bond.isRotatableBond = isRotatableBond
Chem.Mol.GetSingleBonds = GetSingleBonds
Chem.Mol.GetRotatableBonds = GetRotatableBonds

#Rotate Molecule and Generate Conformation
def RotateOnce(self, bond, angle):
    rot = copy(self)
    batom = bond.GetBeginAtom()
    eatom = bond.GetEndAtom()
    bneighbors = map(lambda v: v.GetIdx(), batom.GetNeighbors())
    eneighbors = map(lambda v: v.GetIdx(), eatom.GetNeighbors())
    try:
        bid, eid = batom.GetIdx(), eatom.GetIdx()
        bnid, enid = list(filter(lambda x: x != eid, bneighbors))[0], list(filter(lambda x: x != bid, eneighbors))[0]
        conf = rot.GetConformer()
        angle += Chem.rdMolTransforms.GetDihedralDeg(conf, bnid, bid, eid, enid)
        Chem.rdMolTransforms.SetDihedralDeg(rot.GetConformer(), bnid, bid, eid, enid, angle)
    except:
        pass
    return rot
def Rotate(self, bond, angle):
    rots = {self}
    rangle = angle = int(angle)
    while rangle != 0:
        rots.add(self.RotateOnce(bond, angle))
        rangle = (rangle + angle) % 360
    return rots
def GenerateConformation(self, bonds, angle, once = False, cutoff = 1):
    mols = {self}
    for bond in bonds:
        for mol in list(mols):
            mols.update({mol.RotateOnce(bond, angle), mol.RotateOnce(bond, -angle)} if once else mol.Rotate(bond, angle))
    return filter(lambda v: v.CheckDistanceMatrix(cutoff), mols)

Chem.Mol.RotateOnce = RotateOnce
Chem.Mol.Rotate = Rotate
Chem.Mol.GenerateConformation = GenerateConformation

#Check Molecule By Atom Distance Matrix
def CheckDistanceMatrix(self, cutoff):
    matrix = AllChem.Get3DDistanceMatrix(self)
    return min(matrix[~np.eye(matrix.shape[0], dtype = bool)]) >= cutoff

Chem.Mol.CheckDistanceMatrix = CheckDistanceMatrix

#RMS Calculation and sort
def GetRMS(self, mol):
    return Chem.rdMolAlign.GetBestRMS(self, mol)
    #return Chem.rdMolAlign.AlignMol(self, mol)
def GetRotateRMS(self, bond, angle, fun = max):
    #max instead of average was used for RotateRMS generation and sort
    return fun({GetRMS(self, rot) for rot in self.Rotate(bond, angle)})
def GetSortedRotatableBonds(self, bonds, angle):
    return sorted(bonds, key = lambda x: self.GetRotateRMS(x, angle), reverse = True)

Chem.Mol.GetRMS = GetRMS
Chem.Mol.GetRotateRMS = GetRotateRMS
Chem.Mol.GetSortedRotatableBonds = GetSortedRotatableBonds

#Split Bonds
def SplitBondsByNum(self, bonds, angle, num, overlap):
    bonds = self.GetSortedRotatableBonds(bonds, angle)
    group = []
    num = max(1, num)
    overlap = min(overlap, num - 1) if num > 1 else 0
    start = overlap
    '''for end in range(num, len(bonds), num - overlap):
        group.append(bonds[start:end])
        start = end - overlap
    group.append(bonds[start:])'''
    for end in range(num, len(bonds), num - overlap):
        group.append(bonds[:overlap] + bonds[start: end])
        start = end
    group.append(bonds[:overlap] + bonds[start:])
    return group
def SplitBondsByPer(self, bonds, angle, per, overlap):
    num = int(len(bonds) * min(per, 1))
    overlap = int(len(bonds) * max(overlap, 0))
    return self.SplitBondsByNum(bonds, angle, num, overlap)
def SplitRotatableBondsByNum(self, angle, num, overlap):
    return self.SplitBondsByNum(self.GetRotatableBonds(), angle, num, overlap)
def SplitRotatableBondsByPer(self, angle, per, overlap):
    return self.SplitBondsByPer(self.GetRotatableBonds(), angle, per, overlap)

Chem.Mol.SplitBondsByNum = SplitBondsByNum
Chem.Mol.SplitBondsByPer = SplitBondsByPer
Chem.Mol.SplitRotatableBondsByNum = SplitRotatableBondsByNum
Chem.Mol.SplitRotatableBondsByPer = SplitRotatableBondsByPer



#File Reader & Writer
def Read(filename):
    if path.splitext(filename)[1][1:].lower() == "sdf":
        mol = Chem.SDMolSupplier(filename)[0]
    else:
        mol = Chem.__dict__["MolFrom%sFile" % path.splitext(filename)[1][1:].title()](filename)
    Chem.RemoveHs(mol)
    return mol
def Write(self, filename):
    value = copy(self)
    if path.splitext(filename)[1][1:].lower() == "sdf":
        value.SetProp("_Name", path.splitext(path.basename(filename))[0])
        writer = Chem.SDWriter(filename)
        writer.write(value)
        writer.close()
    else:
        Chem.__dict__["MolTo%sFile" % path.splitext(filename)[1][1:].title()](self, filename)
def write(mols, filename):
    if type(mols) in (list, set):
        if path.splitext(filename)[1][1:].lower() == "sdf":
            writer = Chem.SDWriter(filename)
            for mol in mols:
                writer.write(mol)
            writer.close()
        else:
            raise TypeError("Only .sdf format supported but '%s' was set here." % filename)
    else:
        mols.Write(filename)

Chem.Mol.Write = Write

#Magic Function for Output
def StrAtom(self):
    return "%d" % (self.GetIdx() + 1)
def StrBond(self):
    return "%s-%s" % (self.GetBeginAtom(), self.GetEndAtom())

Chem.Atom.__str__ = StrAtom
Chem.Bond.__str__ = StrBond