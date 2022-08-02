proc import 
out=df
datafile="\\Mac\Home\Desktop\PediatricPain_PatientDemographics.xlsx"  
dbms=xlsx replace;
getnames=yes;
run;

proc freq data=df;
tables sex*group/chisq;
run;

proc freq data=df;
tables pain_duration pain_freq pain_episode_duration;
run;

proc means data=df n mean std median p25 p75 range maxdec=2;
var age nrs_initial nrs_cpt nrs_change;
run;

proc means data=df n mean std median p25 p75 range maxdec=2;
var age nrs_initial nrs_cpt nrs_change;
by group;
run;

proc ttest data=df;
class group;
var age nrs_initial nrs_cpt nrs_change;
run;
