import glob
import numpy as np
import os
from astropy.coordinates import ICRS
from astropy import units as u

DAT_DIR = "/home/dalp/data/rosat/zhou06_60min"
WD = "/home/dalp/Dropbox/science/rosmm/survey_rosat"
SRC_RAD = "120\""
BKG_IN = "240\""
BKG_OUT = "600\""
TOL = 45/60.

def conv2icrs(coords):
    return coords.replace(" ","h",1).replace(" ","m",1).replace(",","s",1).replace(" ",",",1).replace(" ","d",1).replace(" ","m",1).replace(" ","s",1).replace(","," ",1)

os.chdir(WD) #Move to designated directory
OBSERVATIONS = glob.glob(DAT_DIR+'/r*') #Find all files. 
output = open('aux/meta_data.dat', 'w')
zhou06 = [conv2icrs(line.strip()) for line in open("../sample/coords_zhou06.txt", 'r')]
zhou06_ra = np.array([line[:12] for line in zhou06])
zhou06_dec = np.array([line[13:24] for line in zhou06])
#zhou06_z = np.array([line[25:] for line in zhou06])
zhou06 = ICRS(zhou06_ra, zhou06_dec)

accepted = 0
rejected = 0
for obs in OBSERVATIONS:
    rosid = obs[-11:]
    fname = DAT_DIR + "/" + rosid + "/" + rosid + ".public_contents"
    with open(fname) as f:
        for line in f:
            if "REVISION" in line:
                if not line[12]=='2':
                    print "WARNING: Not revision 2: " + line[12]
            if "FILTER" in line:
                if "NONE" not in line:
                    print "WARNING: " + line
            if "INSTRUMENT_NAME" in line:
                pspc = line[19:25]
            if "RIGHT_ASCENSION" in line:
                ra = line[19:34].replace(" ","").replace("\"","").strip()
            if "DECLINATION" in line:
                dec = line[15:30].replace(" ","").replace("\"","").strip()
            if "UT_START_TIME" in line:
                start = line[17:37]
                if pspc == "PSPC C":
                    rmf = "pspcc_gain1_256.rmf"
                    crf = "pspcc_v1.spec_resp"
                elif int(start[9:11]) > 91:
                    rmf = "pspcb_gain2_256.rmf"
                    crf = "pspcb_v2.spec_resp"
                elif int(start[9:11]) < 91:
                    rmf = "pspcb_gain1_256.rmf"
                    crf = "pspcb_v1.spec_resp"
                elif start[3:6] in ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP"]:
                    rmf = "pspcb_gain1_256.rmf"
                    crf = "pspcb_v1.spec_resp"
                elif start[3:6] == "OCT" and int(start[0:2]) < 14:
                    rmf = "pspcb_gain1_256.rmf"
                    crf = "pspcb_v1.spec_resp"
                else:
                    rmf = "pspcb_gain2_256.rmf"
                    crf = "pspcb_v2.spec_resp"
            if "INSTRUMENT_NAME" in line:
                instr = line[19:25].replace(" ","")
            if "TOTAL_ACCEPTED_SECONDS" in line:
                expot = line[26:-2]

        coords = ra + " " + dec
        coords = ICRS(coords)
        idx, d2d, d3d = coords.match_to_catalog_sky(zhou06)

        if d2d.deg[0] > TOL:
            print "Rejected: ", d2d.deg[0], obs
            rejected += 1
        else:
            print "Accepted: ", d2d.deg[0], obs
            accepted += 1
            coords = zhou06[idx].to_string()
            coords = coords.replace(" ",",").replace("h",":").replace("m",":").replace("s","").replace("d",":")
            src = "circle({0},{1})".format(coords, SRC_RAD)
            bkg = "annulus({0},{1},{2})".format(coords, BKG_IN, BKG_OUT)
            output.write("{0} {1} z nh {2} {3} {4} {5} {6}\n".format(coords, rosid, start, rmf, crf, src, bkg))

output.close()
print accepted, rejected
