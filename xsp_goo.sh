#!/bin/bash -x

################################################################
# Help functions
goo_all () {
    cd /Users/silver/dat/xmm/sbo/${oi}_repro/
    printf "data 1:1 ${oi}_pn_spec_grp_${1}.fits
data 1:2 ${oi}_m1_spec_grp_${1}.fits
data 1:3 ${oi}_m2_spec_grp_${1}.fits
ignore 1-3: **-0.3 10.-**
ignore bad
parallel goodness 3
query yes
statistic cstat
statistic test cvm
abund wilm

@${oi}_mo_pl_${1}.xcm
free 3
fit
goodness ${nn}

@${oi}_mo_bb_${1}.xcm
free 3
fit
goodness ${nn}" > ${oi}_goo_${1}.xcm
    xspec < ${oi}_goo_${1}.xcm > ${oi}_goo_${1}.log
}

goo_nom1 () {
    cd /Users/silver/dat/xmm/sbo/${oi}_repro/
    printf "data 1:1 ${oi}_pn_spec_grp_${1}.fits
data 1:2 ${oi}_m2_spec_grp_${1}.fits
ignore 1-2: **-0.3 10.-**
ignore bad
parallel goodness 3
query yes
statistic cstat
statistic test cvm
abund wilm

@${oi}_mo_pl_${1}.xcm
free 3
fit
goodness ${nn}

@${oi}_mo_bb_${1}.xcm
free 3
fit
goodness ${nn}" > ${oi}_goo_${1}.xcm
    xspec < ${oi}_goo_${1}.xcm > ${oi}_goo_${1}.log
}




################################################################
# Parameters
nn=${1}

oi=0149780101
goo_all during
goo_all first
goo_all second

oi=0203560201
goo_all during
goo_all first
goo_all second

oi=0300240501
goo_all during
goo_all first
goo_all second

oi=0300930301
goo_nom1 during
goo_nom1 first
goo_nom1 second

oi=0502020101
goo_all during
goo_all first
goo_all second

oi=0555780101
goo_all during
goo_all first
goo_all second

oi=0604740101
goo_nom1 during
goo_nom1 first
goo_nom1 second

oi=0651690101
goo_all during
goo_all first
goo_all second

oi=0675010401
goo_all during
goo_all first
goo_all second

oi=0743650701
goo_all during
goo_all first
goo_all second

oi=0760380201
goo_nom1 during
goo_nom1 first
goo_nom1 second

oi=0765041301
goo_all during
goo_all first
goo_all second

oi=0770380401
goo_nom1 during
goo_nom1 first
goo_nom1 second

oi=0781890401
goo_all during
goo_all first
goo_all second
