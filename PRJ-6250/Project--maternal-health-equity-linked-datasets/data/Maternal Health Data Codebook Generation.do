use "C:\Users\iwashbur\OneDrive - University of Oklahoma\1 Projects\Leremy Research Group\Maternal Health HCUP SV CENSUS TIME RAW.dta",replace
*wordcb using "C:\Users\iwashbur\OneDrive - University of Oklahoma\Leremy Research Group\Maternal Health Preliminary Data\Combined Codebook.docx",replace
codebook,compact
foreach var of varlist time* {
	ladder `var'
	gen sq`var' = sqrt(`var')
	gen ln`var' = ln(`var')
}
foreach var of varlist p1_003n p1_004n p1_005n p1_006n p1_007n p1_008n p1_009n {
	gen per_`var' = 10*`var'/p1_001n
	gen lnper_`var' = ln(per_`var')
}

/*
hist per_p1_003n
hist per_p1_004n
hist per_p1_005n
hist per_p1_006n
hist per_p1_007n
hist per_p1_008n
hist per_p1_009n
*/

gen Kp1_001n = p1_001n/1000
gen lnKpop = ln(Kp1_001n)
lnskew0 lnep_pov = ep_pov

*capture putdocx begin


foreach outcome of varlist lntime0 lntime1 lntime2 {
	foreach pred of varlist lnep_pov lnper* lnKpop rpl_themes {
			capture putdocx paragraph
			capture putdocx text (" Predicting `outcome' from `pred'."), bold
			regress `outcome' `pred'
			capture putdocx table mytable = etable
			estat hettest
			predict p`outcome'`pred'  if `outcome'!=.
			gen p1`outcome'`pred' = exp(p`outcome'`pred') if `outcome'!=.
	}
}

foreach outcome of varlist sqtime3 {
	foreach pred of varlist lnep_pov lnper* lnKpop rpl_themes {
			capture putdocx paragraph
			capture putdocx text (" Predicting `outcome' from `pred'."), bold
			regress `outcome' `pred'
			capture putdocx table mytable = etable
			estat hettest
			predict p`outcome'`pred'  if `outcome'!=.
			gen p1`outcome'`pred' = p`outcome'`pred'^2 if `outcome'!=.
	}
}


capture putdocx paragraph
regress lntime0 lnep_pov lnper* lnKpop
capture putdocx table mytable = etable
estat hettest
capture putdocx paragraph
regress lntime1 lnep_pov lnper_p1_004n lnper_p1_005n lnper_p1_006n lnper_p1_007n lnper_p1_008n lnper_p1_009n lnKpop
regress lntime1 lnep_pov lnper* lnKpop
capture putdocx table mytable = etable
estat hettest
capture putdocx paragraph
regress lntime2 lnep_pov lnper_p1_004n lnper_p1_005n lnper_p1_006n lnper_p1_007n lnper_p1_008n lnper_p1_009n lnKpop
regress lntime2 lnep_pov lnper* lnKpop
capture putdocx table mytable = etable
estat hettest
capture putdocx paragraph
regress sqtime3 lnep_pov lnper* lnKpop
capture putdocx table mytable = etable
estat hettest

mvreg lntime0 lntime1 lntime2 lntime3 = lnep_pov lnper* lnKpop,corr

foreach var of varlist lnper* {
	regress sqtime3 c.rpl_themes##c.`var'
}

regress lntime0 c.rpl_themes c.lnKpop
regress lntime1 c.rpl_themes c.lnKpop
regress lntime2 c.rpl_themes c.lnKpop
regress sqtime3 c.rpl_themes c.lnKpop

regress sqtime3 c.rpl_themes##c.lnKpop
predict mod_svi_pop if sqtime3 != .
gen sqmod_svi_pop = mod_svi_pop^2

*capture putdocx save "C:\Users\iwashbur\OneDrive - University of Oklahoma\Leremy Research Group\results doc.docx",replace


twoway (fpfit p1lntime0lnper_p1_003n per_p1_003n ) (scatter time0 per_p1_003n), legend(off)
graph export "C:\Users\iwashbur\OneDrive - University of Oklahoma\1 Projects\Leremy Research Group\time0white.png",replace
capture putdocx paragraph
capture putdocx image "C:\Users\iwashbur\OneDrive - University of Oklahoma\1 Projects\Leremy Research Group\time0white.png"


twoway (fpfit p1lntime0lnper_p1_008n per_p1_008n ) (scatter time0 per_p1_008n), legend(off)
twoway (fpfit p1lntime0lnper_p1_009n per_p1_009n ) (scatter time0 per_p1_009n), legend(off)
twoway (fpfit p1lntime0lnKpop p1_001n  if p1_001n < 800000) (scatter time0 p1_001n if p1_001n < 800000), legend(off)

twoway (fpfit p1lntime1lnper_p1_003n per_p1_003n ) (scatter time1 per_p1_003n), legend(off)
twoway (fpfit p1lntime1lnper_p1_004n per_p1_004n ) (scatter time1 per_p1_004n), legend(off)
twoway (fpfit p1lntime1lnper_p1_005n per_p1_005n ) (scatter time1 per_p1_005n), legend(off)
twoway (fpfit p1lntime1lnper_p1_006n per_p1_006n ) (scatter time1 per_p1_006n), legend(off)
twoway (fpfit p1lntime1lnper_p1_007n per_p1_007n ) (scatter time1 per_p1_007n), legend(off)
twoway (fpfit p1lntime1lnper_p1_009n per_p1_009n ) (scatter time1 per_p1_009n), legend(off)
twoway (fpfit p1lntime1lnKpop p1_001n  if p1_001n < 800000) (scatter time1 p1_001n if p1_001n < 800000), legend(off)

twoway (fpfit p1lntime2lnper_p1_004n per_p1_004n ) (scatter time2 per_p1_004n), legend(off)
twoway (fpfit p1lntime2lnper_p1_006n per_p1_006n ) (scatter time2 per_p1_006n), legend(off)
twoway (fpfit p1lntime2lnper_p1_007n per_p1_007n ) (scatter time2 per_p1_007n), legend(off)
twoway (fpfit p1lntime2lnper_p1_009n per_p1_009n ) (scatter time2 per_p1_009n), legend(off)
twoway (fpfit p1lntime2lnKpop p1_001n  if p1_001n < 800000) (scatter time2 p1_001n if p1_001n < 800000), legend(off)

twoway (fpfit p1sqtime3lnper_p1_004n per_p1_004n ) (scatter time3 per_p1_004n), legend(off)
twoway (fpfit p1sqtime3lnper_p1_005n per_p1_005n ) (scatter time3 per_p1_005n), legend(off)
twoway (fpfit p1sqtime3lnper_p1_006n per_p1_006n ) (scatter time3 per_p1_006n), legend(off)
twoway (fpfit p1sqtime3lnper_p1_008n per_p1_008n ) (scatter time3 per_p1_008n), legend(off)
twoway (fpfit p1sqtime3lnper_p1_009n per_p1_009n ) (scatter time3 per_p1_009n), legend(off)
twoway (fpfit p1sqtime3lnKpop p1_001n  if p1_001n < 800000) (scatter time3 p1_001n if p1_001n < 800000), legend(off)


regress sqtime3 lnKpop
predict psqtime3 if sqtime3 != .
gen ptime3 = psqtime3^2
summ sqtime3 psqtime3 time3 ptime3
twoway (qfit ptime3 lnKpop)
twoway (qfit ptime3 lnKpop) (qfit time3 lnKpop)
twoway (scatter time3 lnKpop if p1_001n < 800000), legend(off)
twoway (qfit ptime3 lnKpop) (qfit time3 lnKpop) (scatter time3 lnKpop if p1_001n < 800000), legend(off)
twoway (fpfit ptime3 p1_001n  if p1_001n < 800000) (fpfit time3 p1_001n if p1_001n < 800000) (scatter time3 p1_001n if p1_001n < 800000), legend(off)