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

ros_ra = []
ros_dec = []
rmf = []
crf = []
times = []
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
                    rmf.append("pspcc_gain1_256.rmf")
                    crf.append("pspcc_v1.spec_resp")
                elif int(start[9:11]) > 91:
                    rmf.append("pspcb_gain2_256.rmf")
                    crf.append("pspcb_v2.spec_resp")
                elif int(start[9:11]) < 91:
                    rmf.append("pspcb_gain1_256.rmf")
                    crf.append("pspcb_v1.spec_resp")
                elif start[3:6] in ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP"]:
                    rmf.append("pspcb_gain1_256.rmf")
                    crf.append("pspcb_v1.spec_resp")
                elif start[3:6] == "OCT" and int(start[0:2]) < 14:
                    rmf.append("pspcb_gain1_256.rmf")
                    crf.append("pspcb_v1.spec_resp")
                else:
                    rmf.append("pspcb_gain2_256.rmf")
                    crf.append("pspcb_v2.spec_resp")
        ros_ra.append(ra)
        ros_dec.append(dec)
        times.append(start)

xmm = [conv2icrs(line.strip()) for line in open('../sample/xmm_coords.dat', 'r')]
xmm_f = [line for line in open('../sample/xmm_coords.dat', 'r')]
xmm_ra = np.array([line[:12] for line in xmm])
xmm_dec = np.array([line[13:24] for line in xmm])
xmm = ICRS(xmm_ra, xmm_dec)

zhou06 = [conv2icrs(line.strip()) for line in open("../sample/coords_zhou06.txt", 'r')]
zhou06_ra = np.array([line[:12] for line in zhou06])
zhou06_dec = np.array([line[13:24] for line in zhou06])
zhou06_z = np.array([line[25:] for line in zhou06])
zhou06 = ICRS(zhou06_ra, zhou06_dec)

help_ra = list(ros_ra)
help_dec = list(ros_dec)
help_obs = list(OBSERVATIONS)
help_t = list(times)
help_rmf = list(rmf)
help_crf = list(crf)
rosco = ICRS(help_ra, help_dec)

good = []
for i in range(0,zhou06_z.size):
    xidx, d2d, d3d = zhou06[i].match_to_catalog_sky(xmm)
    if not d2d.deg[0] < 1e-6:
        continue
    idx, d2d, d3d = zhou06[i].match_to_catalog_sky(rosco)
    obses = 0
    while d2d.deg[0] < TOL:
        rosid = help_obs[idx][-11:]
        print "Accepted: ", d2d.deg[0], rosid, zhou06[i].to_string(), obses
        if xmm_f[xidx] not in good:
            good.append(xmm_f[xidx])
        obses += 1
        hcoo = zhou06[i].to_string()
        hcoo = hcoo.replace(" ",",").replace("h",":").replace("m",":").replace("s","").replace("d",":")
        src = "circle({0},{1})".format(hcoo, SRC_RAD)
        bkg = "annulus({0},{1},{2})".format(hcoo, BKG_IN, BKG_OUT)
        output.write("{0}_{1} {2} {3} nh {4} {5} {6} {7} {8}\n".format(hcoo.replace(",","+").replace("+-","-").replace(":","").replace(".",""), obses, rosid, zhou06_z[i], help_t[idx], help_rmf[idx], help_crf[idx], src, bkg))
        
        del help_ra[idx]
        del help_dec[idx]
        del help_obs[idx]
        del help_t[idx]
        del help_rmf[idx]
        del help_crf[idx]
        idx, d2d, d3d = zhou06[i].match_to_catalog_sky(ICRS(help_ra, help_dec))
    help_ra = list(ros_ra)
    help_dec = list(ros_dec)
    help_obs = list(OBSERVATIONS)
    help_t = list(times)
    help_rmf = list(rmf)
    help_crf = list(crf)

output.close()
np.savetxt('../sample/good.dat',np.array(good), fmt="%s")
