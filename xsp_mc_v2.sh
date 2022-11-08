#!/bin/bash -x

# https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/RelNotes_61.html
# - Users can manually re-seed XSPEC's pseudo random number generator
#   at any point in the program using the new "xset seed" command option.
#   The default initial seed is taken from the time at program start-up.




################################################################
# Help functions
print_load_all () {
    printf "data 1:1 ${oi}_pn_spec_grp_${1}.fits
data 1:2 ${oi}_m1_spec_grp_${1}.fits
data 1:3 ${oi}_m2_spec_grp_${1}.fits
ignore 1-3: **-0.3 10.-**
ignore bad
statistic cstat
xset delta 0.0001
method migrad
parallel error 3
query yes
abund wilm" > /Users/silver/dat/xmm/sbo/${oi}_repro/${oi}_load_${1}.xcm
    printf "data 1:1 ${oi}_pn_spec_src_${1}.fits
data 1:2 ${oi}_m1_spec_src_${1}.fits
data 1:3 ${oi}_m2_spec_src_${1}.fits
ignore 1-3: **-0.3 10.-**
ignore bad
statistic cstat
xset delta 0.0001
method migrad
parallel error 3
query yes
abund wilm" > /Users/silver/dat/xmm/sbo/${oi}_repro/${oi}_load_${1}_unb.xcm
    printf "data 1:1 ${oi}_pn_spec_grp_${1}.fak
data 1:2 ${oi}_m1_spec_grp_${1}.fak
data 1:3 ${oi}_m2_spec_grp_${1}.fak
ignore 1-3: **-0.3 10.-**
ignore bad
statistic cstat
xset delta 0.0001
method migrad
parallel error 3
query yes
abund wilm" > /Users/silver/dat/xmm/sbo/${oi}_repro/${oi}_load_${1}_fak.xcm
    printf "#!/bin/bash -x
grppha \"${oi}_pn_spec_src_${1}.fak\" \"${oi}_pn_spec_grp_${1}.fak\" \"group min 1\" \"exit\"
grppha \"${oi}_m1_spec_src_${1}.fak\" \"${oi}_m1_spec_grp_${1}.fak\" \"group min 1\" \"exit\"
grppha \"${oi}_m2_spec_src_${1}.fak\" \"${oi}_m2_spec_grp_${1}.fak\" \"group min 1\" \"exit\"
" > /Users/silver/dat/xmm/sbo/${oi}_repro/${oi}_grppha_${1}.sh
    chmod +x /Users/silver/dat/xmm/sbo/${oi}_repro/${oi}_grppha_${1}.sh

    printf "#!/bin/bash -x
    echo \"FIXING SEG FAULT\"
    printf \"200 0\n201 0\" > qq.txt
    printf \"200 1\n201 1\" > gg.txt
    fmodtab \${1}_pn_spec_grp_\${2}.fak+1 QUALITY qq.txt
    fmodtab \${1}_m1_spec_grp_\${2}.fak+1 QUALITY qq.txt
    fmodtab \${1}_m2_spec_grp_\${2}.fak+1 QUALITY qq.txt
    fmodtab \${1}_pn_spec_grp_\${2}.fak+1 GROUPING gg.txt
    fmodtab \${1}_m1_spec_grp_\${2}.fak+1 GROUPING gg.txt
    fmodtab \${1}_m2_spec_grp_\${2}.fak+1 GROUPING gg.txt
    rm qq.txt gg.txt" > /Users/silver/dat/xmm/sbo/${oi}_repro/fix_seg_fault.sh
    chmod +x /Users/silver/dat/xmm/sbo/${oi}_repro/fix_seg_fault.sh
}

print_load_nom1 () {
    printf "data 1:1 ${oi}_pn_spec_grp_${1}.fits
data 1:2 ${oi}_m2_spec_grp_${1}.fits
ignore 1-2: **-0.3 10.-**
ignore bad
statistic cstat
xset delta 0.0001
method migrad
parallel error 3
query yes
abund wilm" > /Users/silver/dat/xmm/sbo/${oi}_repro/${oi}_load_${1}.xcm
    printf "data 1:1 ${oi}_pn_spec_src_${1}.fits
data 1:2 ${oi}_m2_spec_src_${1}.fits
ignore 1-2: **-0.3 10.-**
ignore bad
statistic cstat
xset delta 0.0001
method migrad
parallel error 3
query yes
abund wilm" > /Users/silver/dat/xmm/sbo/${oi}_repro/${oi}_load_${1}_unb.xcm
    printf "data 1:1 ${oi}_pn_spec_grp_${1}.fak
data 1:2 ${oi}_m2_spec_grp_${1}.fak
ignore 1-2: **-0.3 10.-**
ignore bad
statistic cstat
xset delta 0.0001
method migrad
parallel error 3
query yes
abund wilm" > /Users/silver/dat/xmm/sbo/${oi}_repro/${oi}_load_${1}_fak.xcm
    printf "#!/bin/bash -x
grppha \"${oi}_pn_spec_src_${1}.fak\" \"${oi}_pn_spec_grp_${1}.fak\" \"group min 1\" \"exit\"
grppha \"${oi}_m2_spec_src_${1}.fak\" \"${oi}_m2_spec_grp_${1}.fak\" \"group min 1\" \"exit\"
" > /Users/silver/dat/xmm/sbo/${oi}_repro/${oi}_grppha_${1}.sh
    chmod +x /Users/silver/dat/xmm/sbo/${oi}_repro/${oi}_grppha_${1}.sh

    printf "#!/bin/bash -x
    echo \"FIXING SEG FAULT\"
    printf \"200 0\n201 0\" > qq.txt
    printf \"200 1\n201 1\" > gg.txt
    fmodtab \${1}_pn_spec_grp_\${2}.fak+1 QUALITY qq.txt
    fmodtab \${1}_m2_spec_grp_\${2}.fak+1 QUALITY qq.txt
    fmodtab \${1}_pn_spec_grp_\${2}.fak+1 GROUPING gg.txt
    fmodtab \${1}_m2_spec_grp_\${2}.fak+1 GROUPING gg.txt
    rm qq.txt gg.txt" > /Users/silver/dat/xmm/sbo/${oi}_repro/fix_seg_fault.sh
    chmod +x /Users/silver/dat/xmm/sbo/${oi}_repro/fix_seg_fault.sh
}

print_mc () {
    printf "@${oi}_load_${2}_unb.xcm
xset seed $(python -c "import random; print(random.randint(1,2147483647))")
@${oi}_mo_${1}_${2}.xcm
fakeit








./${oi}_grppha_${2}.sh
@${oi}_load_${2}_fak.xcm
@${oi}_set_${1}_${2}.xcm
fit
method leven
fit
" > ${oi}_mc_${1}_${2}.xcm
}
print_mc_post_fix () {
    printf "
@${oi}_load_${2}_fak.xcm
@${oi}_set_${1}_${2}.xcm
fit
method leven
fit
" > ${oi}_mc_${1}_${2}_post_fix.xcm
}

get_z () {
    echo $(python -c "ff=open('${oi}_mo_bb_during.xcm')
for ii,line in enumerate(ff):
    if ii==10: 
       print(line.split()[0])
       break")
}
get_znh () {
    echo $(python -c "ff=open('${oi}_mo_${1}_during.xcm')
for ii,line in enumerate(ff):
    if '${1}'=='pl' and ii==9: 
       print(line.split()[0])
    elif '${1}'=='bb' and ii==9: 
       print(line.split()[0])")
}

fit_mo () {
    # sts.chi2(3).isf(4*sts.norm(0,1).sf(3)) = 12.67311108710059
    # sts.chi2(2).isf(4*sts.norm(0,1).sf(3)) = 10.44286372078092
    # sts.chi2(3).isf(2*sts.norm(0,1).sf(1)) = 3.5267403802617308
    # sts.chi2(2).isf(2*sts.norm(0,1).sf(1)) = 2.295748928898636
    if [ "${2}" == "bb" ] && [ "${1}" == "during" ]; then
        printf "@${oi}_load_${1}.xcm

@${oi}_set_${2}_${1}.xcm
fit
method leven
fit
error 3.5267403802617308 2 3 8
error 12.67311108710059 2 3 8
method migrad
save model ${oi}_mo_${2}_${1}.xcm
data none
flux 0.3 10.
" > ${oi}_fit_${2}_${1}.xcm
    elif [ "${1}" == "during" ]; then
        printf "@${oi}_load_${1}.xcm

@${oi}_set_${2}_${1}.xcm
fit
method leven
fit
error 3.5267403802617308 2 7 8
error 12.67311108710059 2 7 8
method migrad
save model ${oi}_mo_${2}_${1}.xcm
data none
flux 0.3 10.
" > ${oi}_fit_${2}_${1}.xcm
    else
        printf "@${oi}_load_${1}.xcm

@${oi}_set_${2}_${1}.xcm
fit
method leven
fit
error 2.295748928898636 7 8
error 10.44286372078092 7 8
method migrad
save model ${oi}_mo_${2}_${1}.xcm
" > ${oi}_fit_${2}_${1}.xcm        
    fi
    xspec < ${oi}_fit_${2}_${1}.xcm > ${oi}_fit_${2}_${1}.log
}

plt_mod () {
    cp ${oi}_mo_bb_${1}.xcm ${oi}_mo_${1}.xcm
    sed 's/model /model 2:second/g' ${oi}_mo_pl_${1}.xcm >> ${oi}_mo_${1}.xcm
    printf "data ${oi}_ep_spec_grp_${1}.fits
response 2:1 ${oi}_ep_spec_rsp_${1}.fits
ignore **-0.3 10.-**
ignore bad
abund wilm
@${oi}_mo_${1}.xcm
cpd ${oi}_mo_${1}.ps/cps
setpl ene

ipl
col 0 on 2
col 2 on 3
col 3 on 4
log y
resc y 1e-4
time off
label top
label y N\dE\u (counts s\u-1\d keV\u-1\d)
label x E (keV)
Vie  0.2 0.2 0.7 0.9
csize 1.4
pl
q
exit" > ${oi}_plt_mo_${1}.xcm
    xspec < ${oi}_plt_mo_${1}.xcm > ${oi}_plt_mo_${1}.log

    ps2pdf ${oi}_mo_${1}.ps
    rm ${oi}_mo_${1}.ps
    pdftk ${oi}_mo_${1}.pdf cat r1 output tmp.pdf
    mv tmp.pdf ${oi}_mo_${1}.pdf
    pdfcrop --margins '4 4 4 4' ${oi}_mo_${1}.pdf ${oi}_mo_${1}.pdf
}



################################################################
# Main
main () {
# Create scripts that set the models with initial values for the full duration
cd "/Users/silver/dat/xmm/sbo/${oi}_repro/"

printf "model clear
xset TBABSVERSION 1
mo tbabs(ztbabs(clumin(zbb)))
${nh} 0
0.01 0.001 1.e-5 1.e-5 1000. 1000.
0.3 0.01 0.01 0.01 3. 3.
0.001
100.
=p3
=log(3.9984905221170706e+46)-log(${tt}/(1.+p3))
0.3 0.01 0.03 0.03 3. 3.
=p3
1. 0
" > ${oi}_set_bb_during.xcm

# Fit models initially to the full duration
rm ${oi}_fit_bb_during.xcm
rm ${oi}_mo_bb_during.xcm
fit_mo during bb

zz=$(get_z)
printf "model clear
xset TBABSVERSION 1
model tbabs(ztbabs(clumin(zpow)))
${nh} 0
0.01 0.001 1.e-5 1.e-5 1000. 1000.
${zz}
0.3
10.
=p3
=log(3.9984905221170706e+46)-log(${tt}/(1.+p3))
2. 0.01 -1. -1. 8. 8.
=p3
1. 0
tclout param 7
newpar 7 \$xspec_tclout
" > ${oi}_set_pl_during.xcm
rm ${oi}_fit_pl_during.xcm
rm ${oi}_mo_pl_during.xcm
fit_mo during pl




# Create scripts that set the initial values for the first and second intervals
znh=$(get_znh bb)
printf "model clear
@${oi}_mo_bb_during.xcm
tclout param 7
model clear
xset TBABSVERSION 1
mo tbabs(ztbabs(clumin(zbb)))
${nh} 0
${znh} 0
${zz}
0.3
10.
=p3

0.3 0.01 0.03 0.03 3. 3.
=p3
1. 0
newpar 7 \$xspec_tclout
" > ${oi}_set_bb_first.xcm

znh=$(get_znh pl)
printf "model clear
@${oi}_mo_bb_during.xcm
tclout param 7
model clear
xset TBABSVERSION 1
model tbabs(ztbabs(clumin(zpow)))
${nh} 0
${znh} 0
${zz}
0.3
10.
=p3

2. 0.01 -1. -1. 8. 8.
=p3
1. 0
newpar 7 \$xspec_tclout
" > ${oi}_set_pl_first.xcm


cp ${oi}_set_bb_first.xcm ${oi}_set_bb_second.xcm
cp ${oi}_set_pl_first.xcm ${oi}_set_pl_second.xcm

# Fit models initially to the full duration
rm ${oi}_fit_bb_first.xcm
rm ${oi}_fit_bb_second.xcm
rm ${oi}_fit_pl_first.xcm
rm ${oi}_fit_pl_second.xcm
rm ${oi}_mo_bb_first.xcm
rm ${oi}_mo_bb_second.xcm
rm ${oi}_mo_pl_first.xcm
rm ${oi}_mo_pl_second.xcm
fit_mo first bb
fit_mo first pl
fit_mo second bb
fit_mo second pl




# Plot the models over the combined EPIC spectra
plt_mod during
plt_mod first
plt_mod second



################################################################
# Run MC
rm *.fak
rm -rf bb_log
rm -rf pl_log
mkdir -p bb_log
mkdir -p pl_log
for i in $(seq 1 ${nn})
do
    ii=$(printf "%06d" ${i})
    print_mc bb during
    print_mc pl during
    print_mc bb first
    print_mc pl first
    print_mc bb second
    print_mc pl second
    print_mc_post_fix bb during
    print_mc_post_fix pl during
    print_mc_post_fix bb first
    print_mc_post_fix pl first
    print_mc_post_fix bb second
    print_mc_post_fix pl second

    xspec < ${oi}_mc_bb_during.xcm > bb_log/${oi}_bb_during_${ii}.log
    if [ "$(tail -1 bb_log/${oi}_bb_during_${ii}.log | awk '{print $1;}')" == "!XSPEC12>ignore" ]; then
        ./fix_seg_fault.sh ${oi} during
        xspec < ${oi}_mc_bb_during_post_fix.xcm >> bb_log/${oi}_bb_during_${ii}.log
    fi
    xspec < ${oi}_mc_bb_first.xcm > bb_log/${oi}_bb_first_${ii}.log
    if [ "$(tail -1 bb_log/${oi}_bb_first_${ii}.log | awk '{print $1;}')" == "!XSPEC12>ignore" ]; then
        ./fix_seg_fault.sh ${oi} first
    xspec < ${oi}_mc_bb_first_post_fix.xcm >> bb_log/${oi}_bb_first_${ii}.log
    fi
    xspec < ${oi}_mc_bb_second.xcm > bb_log/${oi}_bb_second_${ii}.log
    if [ "$(tail -1 bb_log/${oi}_bb_second_${ii}.log | awk '{print $1;}')" == "!XSPEC12>ignore" ]; then
        ./fix_seg_fault.sh ${oi} second
    xspec < ${oi}_mc_bb_second_post_fix.xcm >> bb_log/${oi}_bb_second_${ii}.log
    fi
    rm *.fak

    xspec < ${oi}_mc_pl_during.xcm > pl_log/${oi}_pl_during_${ii}.log
    if [ "$(tail -1 pl_log/${oi}_pl_during_${ii}.log | awk '{print $1;}')" == "!XSPEC12>ignore" ]; then
        ./fix_seg_fault.sh ${oi} during
        xspec < ${oi}_mc_pl_during_post_fix.xcm >> pl_log/${oi}_pl_during_${ii}.log
    fi
    xspec < ${oi}_mc_pl_first.xcm > pl_log/${oi}_pl_first_${ii}.log
    if [ "$(tail -1 pl_log/${oi}_pl_first_${ii}.log | awk '{print $1;}')" == "!XSPEC12>ignore" ]; then
        ./fix_seg_fault.sh ${oi} first
    xspec < ${oi}_mc_pl_first_post_fix.xcm >> pl_log/${oi}_pl_first_${ii}.log
    fi
    xspec < ${oi}_mc_pl_second.xcm > pl_log/${oi}_pl_second_${ii}.log
    if [ "$(tail -1 pl_log/${oi}_pl_second_${ii}.log | awk '{print $1;}')" == "!XSPEC12>ignore" ]; then
        ./fix_seg_fault.sh ${oi} second
    xspec < ${oi}_mc_pl_second_post_fix.xcm >> pl_log/${oi}_pl_second_${ii}.log
    fi
    rm *.fak
done
}








################################################################
# Parameters
nn=${1}

oi=0149780101
nh=0.0640
tt=1450.
print_load_all during
print_load_all first
print_load_all second
main

oi=0203560201
nh=0.04172
tt=27000.
print_load_all during
print_load_all first
print_load_all second
main

oi=0300240501
nh=0.1083
tt=240.
print_load_all during
print_load_all first
print_load_all second
main

oi=0300930301
nh=0.04346
tt=5000.
print_load_nom1 during
print_load_nom1 first
print_load_nom1 second
main

oi=0502020101
nh=0.016344
tt=165.
print_load_all during
print_load_all first
print_load_all second
main

oi=0604740101
nh=0.05078
tt=10000.
print_load_nom1 during
print_load_nom1 first
print_load_nom1 second
main

oi=0675010401
nh=0.02971
tt=1000.
print_load_all during
print_load_all first
print_load_all second
main

oi=0743650701
nh=0.112
tt=10000.
print_load_all during
print_load_all first
print_load_all second
main

oi=0760380201
nh=0.05084
tt=5500.
print_load_nom1 during
print_load_nom1 first
print_load_nom1 second
main

oi=0765041301
nh=0.0842
tt=6000.
print_load_all during
print_load_all first
print_load_all second
main

oi=0770380401
nh=0.02492
tt=900.
print_load_nom1 during
print_load_nom1 first
print_load_nom1 second
main

oi=0781890401
nh=0.01882
tt=175.
print_load_all during
print_load_all first
print_load_all second
main
