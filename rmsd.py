from rdkit import Chem
from rdkit.Chem import AllChem
import abcr

#Generate Input Coordinate
def RandomTransform(self, seed):
    mol = Chem.AddHs(self)
    AllChem.EmbedMolecule(mol, randomSeed = seed)
    return Chem.RemoveHs(mol)

Chem.Mol.RandomTransform = RandomTransform

#Calculate RMS Score
def GetScore(self, ref):
    return Chem.rdMolAlign.GetBestRMS(self, ref)

Chem.Mol.GetScore = GetScore

#Output Group Info
def GroupInfo(group):
    return ",".join(map(str, group))
def GroupsInfo(groups):
    return "], [".join(map(GroupInfo, groups))

#Comformations Generated by bgroups and angle & Reduced by scoreFunc
def ConfReduce(self, bgroups, angle, once = False, cutoff = 1, file = None, step = "", round = 0, **kwargs):
    #Different once, in Flower Step as False, while in Swing Step as True
    conf = self
    score = float("inf")
    count = 0
    if file:
        self.Write("%s_0%s" % (path.splitext(file)[0], path.splitext(file)[1]))
    for group in bgroups:
        confs = conf.GenerateConformation(group, angle, once, cutoff)
        for tconf in confs:
            count += 1
            marker = ""
            tconf.SetProp("_Name", "%s_%d_%d" % (step, round, count))
            tscore = tconf.GetScore(**kwargs)
            if tscore < score:
                marker = "_b"
                score = tscore
                conf = tconf
            if file:
                tconf.Write("%s_%d_%d%s%s" % (path.splitext(file)[0], round, count, marker, path.splitext(file)[1]))
    return (conf, count, score)

Chem.Mol.ConfReduce = ConfReduce


#====================== Main ======================
from os import path
import sys, getopt
def main(argv):
    version = "3.0.0"
    #==================== Hyperparameters ====================
    parameters = {
        "flowerAngle": 180,
        "flowerPerct": 0.6,
        "flowerOverlap": 0,
        "swingAngle": 180,
        "swingDescent": 0.7,
        "swingPerct": 0.3,
        "swingOverlap": 0,
        "stopAngle": 5,
        "stopDelta": 0.1,
        "seed": 900825,
        "cutoffDistance": 1.0,
        "flowerTemp": None,
        "swingTemp": None
    }

    #Get Parameters from Command-Line Options
    try:
        opts, _ = getopt.getopt(argv, "i:c:o:v", ["flowerAngle=", "flowerPerct=", "flowerOverlap=", "swingAngle=", "swingDescent=", "swingPerct=", "swingOverlap=", "stopAngle=", "stopDelta=", "seed=", "cutoffDistance=", "flowerTemp=", "swingTemp="])
    except getopt.GetoptError:
        print("Command error.")
        sys.exit(2)

    ofile = None
    for opt, arg in opts:
        if opt in ("-v"):
            print("ABCR, version %s" % version)
            return
        elif opt in ("-i"):
            ifile = arg
        elif opt in ("-c"):
            rfile = arg
        elif opt in ("-o"):
            ofile = arg
        elif opt in ("--flowerTemp", "--swingTemp"):
            parameters[opt[2:]] = arg
        elif opt in ("--seed"):
            parameters[opt[2:]] = int(arg)
        else:
            try:
                parameters[opt[2:]] = float(arg)
            except:
                print("Option value error: %s = %s" % (opt, arg))
                sys.exit(2)
    
    #zzh: 
    parameters["swingAngle"]=parameters["flowerAngle"]
    print("*Parameters")
    try:
        for key, value in parameters.items():
            print("   "+key.ljust(15),value,sep="| ")
    except:
        raise
    
    #Reader
    print("*File Reader")
    try:
        print("   File: %s" % path.basename(ifile))
    except NameError:
        print("Undefined input filename.")
        sys.exit(2)

    try:
        #Read start 3D Conformation
        start = abcr.Read(ifile)
    except:
        print("Input file format error.")
        sys.exit(2)

    try:
        #Read result 3D Conformation if Exist rfile
        reslt = abcr.Read(rfile)
    except:
        #Use start File as result Conformation and Regenerate start Conformation if NOT Exist rfile
        print("   @ Warning: Start Conformation was Generated by '%s'" % path.basename(ifile))
        reslt = start
        start = reslt.RandomTransform(parameters["seed"])
    
    print("   Rotatable Bonds: %d" % len(start.GetRotatableBonds()))
    rms = start.GetRMS(reslt)
    print("   Current RMSD: %.9f" % rms)
    tcount = 1

    #Flower
    print("*Flower Started")
    #Split Bonds into Groups According to Hyperparameters
    groups = start.SplitRotatableBondsByPer(parameters["flowerAngle"], parameters["flowerPerct"], parameters["flowerOverlap"])
    print("   Current Group Info: [%s]" % GroupsInfo(groups))
    #Flower the Confofrmation
    flower, count, rms = start.ConfReduce(groups, parameters["flowerAngle"], ref = reslt, cutoff = parameters["cutoffDistance"], file = parameters["flowerTemp"], step = "flower")
    tcount += count
    print("   Current Angle: %6.2f  RMSD: %.9f  Confs: %6d" % (parameters["flowerAngle"], rms, count))
    print("*Flower Finished")

    #Swing
    print("*Swing Started")
    #Init Input Values
    delta = rms = float("inf")
    swingAngle = parameters["swingAngle"]
    swing = flower
    round = 0
    #Main Cicle of Swing
    while delta > parameters["stopDelta"] or swingAngle > parameters["stopAngle"]:
        #Update swingAngle
        swingAngle *= parameters["swingDescent"]
        #Split Bonds into Groups According to Hyperparameters
        groups = swing.SplitRotatableBondsByPer(swingAngle, parameters["swingPerct"], parameters["swingOverlap"])
        print("   Current Group Info: [%s]" % GroupsInfo(groups))
        #Swing the Conformation
        tswing, count, rms = swing.ConfReduce(groups, swingAngle, once = True, ref = reslt, cutoff = parameters["cutoffDistance"], file = parameters["swingTemp"], step = "swing", round = round)
        round += 1
        tcount += count
        #Update delta
        delta = swing.GetRMS(tswing)
        swing = tswing
        print("   Current Angle: %6.2f  RMSD: %.9f  Confs: %6d" % (swingAngle, rms, count))
    print("*Swing Finished")
    print("   Final RMSD: %.9f  Total Confs: %d" % (rms, tcount))

    #Writer
    print("*File Writer")
    #Write Final Structure
    try:
        ofile = ofile if ofile else ".out".join(path.splitext(ifile))
        swing.Write(ofile)
    except:
        ofile = "%s.out.mol" % path.splitext(ifile)[0]
        swing.Write(ofile)
    finally:
        print("   Output file saved: %s" % ofile)




if __name__ == "__main__":
    import time
    start = time.process_time()
    print("#========== Start ==========")
    try:
        main(sys.argv[1:])
    except:
        print("Running error.")
    print("#=========== End ===========")
    end = time.process_time()
    print("Running Time: %.9fs" % (end - start))