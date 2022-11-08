#This script is not fail-safe, read comments!
import subprocess
import numpy as np
import sys
from scipy.interpolate import UnivariateSpline

rspname = sys.argv[1]
srcname = sys.argv[2]
outfile = sys.argv[3]
OVERSAMPLE = 3 #Parameter, defined analogously as to XMM SAS specgroup
MIN_CTS = 25

class Group():
    start_chan = -1
    end_chan = -1
    FWHM = -1
    tot_counts = 0

    def is_group_complete(self, chan, prev_chan):
        if self.tot_counts >= MIN_CTS:
            if prev_chan-self.start_chan+1 >= self.FWHM and not self.FWHM == -1:
                if not prev_chan == chan:
                    return True
        return False
    
    def update(self, COUNTS, chan, prev_FWHM):
        self.start_chan = chan if (self.start_chan == -1) else self.start_chan
        self.FWHM = prev_FWHM if (prev_FWHM > self.FWHM) else self.FWHM
        self.tot_counts += count_photons(self, COUNTS, chan)
        
    def my_print(self):
        print(str(int(self.start_chan))+' '+str(int(self.end_chan))+' '+str(int(self.FWHM)))
    
def count_photons(grp, COUNTS, chan):
    assert chan <= np.shape(COUNTS)[0], "Error: COUNTS index out of bounds: %i" % int(chan+1)
    photons = np.sum(COUNTS[grp.start_chan:chan+1])
    COUNTS[grp.start_chan:chan+1] = 0
    return photons

def make_binning(data, COUNTS):
    binning = []
    grp = Group()
    for row in range(0, np.shape(data)[0]):
        chan = data[row,0]
        prev_row = (not row == 0) * (row-1)
        prev_chan = data[prev_row,0]
        dummy = 1 if chan == prev_chan else 0

        if chan == prev_chan and not row == 0:
            prev_FWHM = data[prev_row,1]
            grp.update(COUNTS, chan, prev_FWHM)
            continue

        for implicit in range(int(prev_chan),int(chan+dummy)):
            prev_FWHM = data[prev_row,1] if implicit==prev_chan else -1 #Only if non-fictive channel, else nothing is known of FWHM
            grp.update(COUNTS, implicit, prev_FWHM)
            if grp.is_group_complete(chan, implicit):
                grp.FWHM = implicit-grp.start_chan+1
                grp.end_chan = implicit
                binning.append(grp)
                grp = Group()
    grp.FWHM = implicit-grp.start_chan+1
    grp.end_chan = implicit
    binning.append(grp)
    return binning
        
def print_file(binning):
    f = open(outfile, 'w')
    for grp in binning:
        f.write(str(int(grp.start_chan)+1)+' '+str(int(grp.end_chan)+1)+' '+str(int(grp.FWHM))+'\n')
    f.close()

rows = int(subprocess.check_output("hexdump -e '80/1 \"%_p\" \"\n\"' " + rspname + " | grep -m 1 NAXIS2 | awk '{print $3}'", shell=True)[0:-1])
rows = int(rows*0.9) #Because strange things happen with the response at too high energies making it difficult to find FWHM !SUZAKU SPECIFIC, CHOOSE WISELY!

#Find the FWHMs
FWHM = np.zeros(rows)
bins = np.zeros(rows)
subprocess.call(["fdump","infile=" + rspname + "[1]","outfile=F_CHAN.txt","columns=F_CHAN","rows=-","prhead=no","showcol=no","showunit=no","showrow=no","clobber=yes"], shell=False)
F_CHAN = np.loadtxt("F_CHAN.txt")
subprocess.call(["fdump","infile=" + srcname + "[1]","outfile=COUNTS.txt","columns=COUNTS","rows=-","prhead=no","showcol=no","showunit=no","showrow=no","clobber=yes"], shell=False)
COUNTS = np.loadtxt("COUNTS.txt")

for i in range(0,rows):
    subprocess.call(["fdump","infile=" + rspname + "[1]","outfile=resp.txt","columns=MATRIX","rows="+str(i+1),"prhead=no","showcol=no","showunit=no","showrow=no","clobber=yes"], shell=False)

    resp = np.loadtxt("resp.txt")
    x = np.linspace(1,np.size(resp),np.size(resp))
    spline = UnivariateSpline(x,resp-np.max(resp)/2,s=0)
    rts = spline.roots()
    if len(rts) == 1: #When left wing of response outside of channel 0
        r1=0
        r2=rts[0]
    elif len(rts) == 0: #When left wing of response outside of channel 0
        r1=0
        r2=0
    else:
        r1,r2=rts[0:2]
    FWHM[i] = np.ceil((r2-r1)/OVERSAMPLE)
    bins[i] = resp.argmax(axis=0)+F_CHAN[i]

data=np.column_stack((bins,np.ceil(FWHM)))
data=data[data[:,0].argsort()] #Sort by bin
            
#Make the file
binning = make_binning(data, COUNTS)
print_file(binning)

subprocess.call(["rm","resp.txt"])
subprocess.call(["rm","F_CHAN.txt"])
