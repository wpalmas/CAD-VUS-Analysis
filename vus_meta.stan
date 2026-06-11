const {
  Document,Packer,Paragraph,TextRun,Table,TableRow,TableCell,
  ImageRun,HeadingLevel,AlignmentType,BorderStyle,WidthType,
  ShadingType,LevelFormat,PageBreak
} = require('docx');
const fs=require('fs');

const R=JSON.parse(fs.readFileSync('/home/claude/paper_figures/results.json','utf8'));
const s1=R.sim1; const s2=R.sim2;

const ne='n/e\u2020';
const f3=x=>(x==null||x!==x)?ne:(+x).toFixed(3);
const fp=x=>(x==null||x!==x)?ne:(+x).toFixed(3);
const fci=(m,lo,hi,pct=false)=>{
  if(m==null||m!==m) return ne;
  if(pct) return `${(+m*100).toFixed(1)}%\u2002[${(+lo*100).toFixed(1)}%, ${(+hi*100).toFixed(1)}%]`;
  return `${(+m).toFixed(3)}\u2002[${(+lo).toFixed(3)}, ${(+hi).toFixed(3)}]`;
};
const fpvus=arr=>{
  if(!arr||arr[0]==null||arr[0]!==arr[0]) return ne;
  return `${(+arr[0]).toFixed(3)}\u2002[${(+arr[1]).toFixed(3)}, ${(+arr[2]).toFixed(3)}]`;
};
const fsgi=(m,lo,hi)=>(m==null||m!==m)?ne:`${(+m).toFixed(1)}%\u2002[${(+lo).toFixed(1)}%, ${(+hi).toFixed(1)}%]`;

const NAVY='1F4E79',BLUE='2E75B6',LTBLUE='DEEAF1',LTGREY='F2F2F2',
      WHITE='FFFFFF',DKGRN='1D6B2E',GREEN='E2EFDA',AMBER='FFF2CC',RED='FCE4D6';
const bd={style:BorderStyle.SINGLE,size:1,color:'CCCCCC'};
const bds={top:bd,bottom:bd,left:bd,right:bd};
const W=9360;

const run=(t,o={})=>new TextRun({text:String(t),font:'Arial',
  bold:o.bold||false,italics:o.italics||false,size:o.size||22,color:o.color||'000000'});
const p=(children,o={})=>new Paragraph({
  alignment:o.align||AlignmentType.LEFT,
  spacing:{before:o.before||0,after:o.after||140},
  children:Array.isArray(children)?children:[run(children,o)]});
const h1=t=>new Paragraph({heading:HeadingLevel.HEADING_1,spacing:{before:320,after:140},
  children:[run(t,{bold:true,size:28,color:NAVY})]});
const h3=t=>new Paragraph({heading:HeadingLevel.HEADING_3,spacing:{before:180,after:90},
  children:[run(t,{bold:true,size:22,color:'2F5496'})]});
const rule=()=>new Paragraph({spacing:{before:40,after:60},
  border:{bottom:{style:BorderStyle.SINGLE,size:5,color:BLUE}},children:[]});
const blank=(n=1)=>Array(n).fill(new Paragraph({children:[]}));
const eq=t=>new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:100,after:100},
  children:[new TextRun({text:t,font:'Courier New',size:22})]});
const bul=t=>new Paragraph({numbering:{reference:'bullets',level:0},spacing:{after:90},
  children:[run(t)]});
const img=(f,w,h)=>new Paragraph({alignment:AlignmentType.CENTER,
  spacing:{before:130,after:80},
  children:[new ImageRun({
    data:fs.readFileSync(`/home/claude/paper_figures/${f}`),
    transformation:{width:Math.round(w*0.103),height:Math.round(h*0.103)},type:'png'})]});
const cap=t=>new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:40,after:200},
  children:[run(t,{italics:true,size:19,color:'444444'})]});
const hc=(t,w,shade=NAVY)=>new TableCell({borders:bds,
  width:{size:w,type:WidthType.DXA},shading:{fill:shade,type:ShadingType.CLEAR},
  margins:{top:80,bottom:80,left:100,right:100},
  children:[new Paragraph({alignment:AlignmentType.CENTER,
    children:[run(t,{bold:true,size:18,color:WHITE})]})]});
const dc=(t,w,shade=WHITE,bold=false,col='000000')=>new TableCell({borders:bds,
  width:{size:w,type:WidthType.DXA},shading:{fill:shade,type:ShadingType.CLEAR},
  margins:{top:65,bottom:65,left:100,right:100},
  children:[new Paragraph({alignment:AlignmentType.CENTER,
    children:[run(String(t),{bold,size:18,color:col})]})]});
const dl=(t,w,shade=WHITE,bold=false,col='000000')=>new TableCell({borders:bds,
  width:{size:w,type:WidthType.DXA},shading:{fill:shade,type:ShadingType.CLEAR},
  margins:{top:65,bottom:65,left:100,right:100},
  children:[new Paragraph({alignment:AlignmentType.LEFT,
    children:[run(String(t),{bold,size:18,color:col})]})]});

// ── Table helpers ─────────────────────────────────────────────────────────────
// Simulation 1: columns [label, T1, T2, dVUS, P(T1>T2)]
const W2=[2200,2000,2000,1980,1180];
const row2=(lbl,t1,t2,dv,pp,shade=WHITE)=>new TableRow({children:[
  dl(lbl,W2[0],shade,true),dc(t1,W2[1],shade),dc(t2,W2[2],shade),
  dc(dv,W2[3],shade,false,DKGRN),dc(pp,W2[4],shade,true,DKGRN)]});

const pvus_label=['Mild (0\u201322)','Intermediate (23\u201332)','Severe (\u226533)'];

const t2rows=[
  new TableRow({children:[hc('Measure',W2[0]),hc('Test 1',W2[1]),hc('Test 2',W2[2]),
    hc('\u0394 (T1\u2212T2)',W2[3]),hc('P(T1>T2)',W2[4])]}),
  row2('n patients','1,500','1,500','\u2014','\u2014',LTBLUE),
  row2('SYNTAX range','1\u201360','1\u201360','\u2014','\u2014'),
  row2('Naive AUC',f3(s1.nauc1),f3(s1.nauc2),'\u2014','\u2014',LTBLUE),
  row2('Global VUS  [95% CI]',
    fci(s1.vus1,s1.vus1_lo,s1.vus1_hi),
    fci(s1.vus2,s1.vus2_lo,s1.vus2_hi),
    fci(s1.dvus,s1.dvus_lo,s1.dvus_hi),
    fp(s1.p_vus1_gt)),
  ...pvus_label.map((lbl,j)=>
    row2(`PVUS \u2014 ${lbl}  [95% CI]`,
      fpvus(s1.pvus1[j]), fpvus(s1.pvus2[j]),
      fpvus(s1.dpvus[j]), fp(s1.p_pvus1_gt[j]),
      j%2===0?LTBLUE:WHITE)),
  row2('MVF  [95% CI]',
    fci(s1.mvf1,s1.mvf1_lo,s1.mvf1_hi,true),
    fci(s1.mvf2,s1.mvf2_lo,s1.mvf2_hi,true),'\u2014','\u2014',LTBLUE),
  row2('ICV  [95% CI]',
    fci(s1.icv1,s1.icv1_lo,s1.icv1_hi,true),
    fci(s1.icv2,s1.icv2_lo,s1.icv2_hi,true),'\u2014','\u2014'),
  row2('SGI  [95% CI]',
    fsgi(s1.sgi1,s1.sgi1_lo,s1.sgi1_hi),
    fsgi(s1.sgi2,s1.sgi2_lo,s1.sgi2_hi),'\u2014','\u2014',LTBLUE),
];

// Simulation 2: columns [label, TestA, TestB, dPVUS, P(A>B)]
const W3=[2200,2000,2000,1980,1180];
const row3=(lbl,tA,tB,dv,pp,shade=WHITE)=>new TableRow({children:[
  dl(lbl,W3[0],shade,true),dc(tA,W3[1],shade),dc(tB,W3[2],shade),
  dc(dv,W3[3],shade,false,DKGRN),dc(pp,W3[4],shade,true,DKGRN)]});

const t3rows=[
  new TableRow({children:[hc('Measure',W3[0]),hc('Test A\n(Full spectrum)',W3[1]),
    hc('Test B\n(Severe only)',W3[2]),hc('\u0394 (A\u2212B)',W3[3]),hc('P(A>B)',W3[4])]}),
  row3('n patients','1,500','500','\u2014','\u2014',LTBLUE),
  row3('SYNTAX range','1\u201360','\u226533 only','\u2014','\u2014'),
  row3('Naive AUC',f3(s2.naucA),f3(s2.naucB),'\u2014','\u2014',LTBLUE),
  row3('Global VUS  [95% CI]',
    fci(s2.vusA,s2.vusA_lo,s2.vusA_hi), ne, ne, ne),
  ...pvus_label.map((lbl,j)=>{
    const dpv=s2.dpvusAB[j]; const pp=s2.p_pvusA_gt[j];
    return row3(`PVUS \u2014 ${lbl}  [95% CI]`,
      fpvus(s2.pvusA[j]), fpvus(s2.pvusB[j]),
      fpvus(dpv), fp(pp),
      j%2===0?LTBLUE:WHITE);
  }),
  row3('MVF  [95% CI]',
    fci(s2.mvfA,s2.mvfA_lo,s2.mvfA_hi,true),
    fci(s2.mvfB,s2.mvfB_lo,s2.mvfB_hi,true),'\u2014','\u2014',LTBLUE),
  row3('ICV  [95% CI]',
    fci(s2.icvA,s2.icvA_lo,s2.icvA_hi,true),
    fci(s2.icvB,s2.icvB_lo,s2.icvB_hi,true),'\u2014','\u2014'),
  row3('SGI  [95% CI]',
    fsgi(s2.sgiA,s2.sgiA_lo,s2.sgiA_hi), ne,'\u2014','\u2014',LTBLUE),
];

const doc=new Document({
  styles:{default:{document:{run:{font:'Arial',size:22}}},
    paragraphStyles:[
      {id:'Heading1',name:'Heading 1',basedOn:'Normal',next:'Normal',
       run:{size:28,bold:true,font:'Arial'},
       paragraph:{spacing:{before:320,after:140},outlineLevel:0}},
      {id:'Heading3',name:'Heading 3',basedOn:'Normal',next:'Normal',
       run:{size:22,bold:true,font:'Arial'},
       paragraph:{spacing:{before:180,after:90},outlineLevel:2}},
    ]},
  numbering:{config:[{reference:'bullets',levels:[{level:0,format:LevelFormat.BULLET,
    text:'\u2022',alignment:AlignmentType.LEFT,
    style:{paragraph:{indent:{left:720,hanging:360}}}}]}]},
  sections:[{
    properties:{page:{size:{width:12240,height:15840},
      margin:{top:1440,right:1440,bottom:1440,left:1440}}},
    children:[

// TITLE
new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:300,after:140},
  children:[run('The Volume Under the ROC Surface: A Three-Dimensional Statistic to Address Spectrum Bias',
    {bold:true,size:28,color:NAVY})]}),
new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:0,after:300},
  children:[run('By Walter R. Palmas, MD, MS',{italics:true,size:22})]}),

// INTRODUCTION
h1('Introduction'),rule(),
p('The disease severity spectrum is a modifier of test sensitivity. In patients with severe disease the pathological changes tend to be more pronounced, and the signal-to-noise ratio is higher, pushing the sensitivity upwards. At the other end of the disease spectrum the pathological changes of mild disease are harder to detect, the signal-to-noise ratio declines, and the test sensitivity is lowered. These effects on test sensitivity are a universal phenomenon and they often cause spectrum bias, a well-recognized threat to the validity and applicability of diagnostic test accuracy estimates.'),
p('Ideally diagnostic tests should be evaluated in a representative sample of the entire spectrum of disease, providing accuracy estimates for disease severity sub-groups. However, this is not the case in many studies. Studies that recruit participants in tertiary referral centers are bound to sample preferentially sicker patients with more severe disease. Similarly, samples referred for testing because of recurring symptoms are severity-enriched by design. Recruitment at a primary care network is more likely to accrue patients with milder disease. Finally, case-control designs pooling patients with severe disease against normal controls represent the most egregious form of spectrum bias.'),
p('Receiver operating characteristic (ROC) curves are among the most frequently used statistics to assess diagnostic accuracy. The Area Under the Curve (AUC) measures overall test accuracy. ROC curves are clinically misleading when compromised by spectrum bias, and the pernicious consequences of bias persist when the data are meta-analyzed.'),
p('This manuscript introduces the Volume Under the ROC Surface (VUS), which adds disease severity to the two-dimensional ROC curve, creating a three-dimensional representation of test performance that accounts for disease severity and how thoroughly the study sample represents the overall patient population.'),

// METHODS
h1('Methods'),rule(),

h3('Adding a Third Dimension to ROC Analysis'),
p('The VUS inherits three properties of the AUC: it is a single number, it runs from 0.5 (uninformative) to 1.0 (perfect), and it allows comparison between tests \u2014 while adding a fourth: it accounts for disease severity. The conceptual bridge from AUC to VUS has three steps:'),
p('Step 1. At a fixed severity s\u2080, compute AUC(s\u2080) = \u222b\u2080\u00b9 TPR(FPR, s\u2080) dFPR \u2014 the familiar AUC.'),
p('Step 2. Compute AUC(s) across the full range of disease severity levels, adding the severity axis. This produces a dome-like surface.'),
p('Step 3. The VUS is the mean height of the dome above the diagonal floor:'),
eq('VUS = (1/N\u209b) \u00d7 \u03a3\u209b AUC(s)'),
p('where AUC(s) is the trapezoidal AUC of the ROC cross-section at severity level s, and N\u209b is the number of severity grid points.'),

h3('Coverage-First Principle'),
p('Before computing any summary statistic, the code scans the severity grid to determine which regions have adequate data (\u2265 n* = 15 diseased patients within a local window). Statistics are computed only over covered regions. The following rules govern estimability:'),
bul('Global VUS is estimable only when the observed severity range covers at least 50% of the full possible severity range. Otherwise the message \u201cGlobal VUS cannot be estimated: coverage of the full severity range is insufficient\u201d is returned. This prevents an inflated VUS being computed from a severity-enriched subsample.'),
bul('Partial VUS (PVUS) is estimated independently for each severity region. A test with data only in the severe region returns PVUS_severe with a credible interval, and n/e for mild and intermediate regions.'),
bul('Global \u0394VUS requires both tests to have estimable global VUS values. If either returns n/e, global \u0394VUS is also n/e.'),
bul('Region-specific \u0394PVUS is estimable for each region independently, where both tests have data in that region.'),
...blank(1),

h3('VUS Estimation: Pooled Non-Diseased Approach'),
p('Non-diseased patients have no disease severity score (SYNTAX = 0 by definition). The false positive rate at any threshold is estimated from the entire non-diseased population pooled, and used as the shared FPR reference at every severity grid point. For each grid point s\u2096, the window of diseased patients is additionally clipped to the same PVUS region as the grid point, so no patient contributes to AUC estimates across a region boundary.'),

h3('\u0394VUS and \u0394PVUS: Comparing Two Tests'),
p('In the paired design, both test scores are available for every patient in the same cohort. In each of M = 1,000 bootstrap draws, the same resampled patient indices are used to compute VUS\u2081 and VUS\u2082, preserving within-patient correlation. \u0394VUS = VUS\u2081 \u2212 VUS\u2082 and its 95% credible interval are derived from the resulting paired posterior distribution.'),
p('In the unpaired design, test scores come from two separate cohorts. Bootstrap resampling is performed independently for each cohort in each draw. Global \u0394VUS is computed only when both tests have estimable global VUS values. Region-specific \u0394PVUS is computed independently for each region where both tests have data, using matched bootstrap draws across the two independent posteriors.'),
p('For every comparison \u2014 global VUS, global \u0394VUS, and each region-specific PVUS and \u0394PVUS \u2014 the Bayesian posterior probability P(Test 1 > Test 2) is reported directly as the fraction of bootstrap draws in which the first test exceeds the second. This probability provides the clinical conclusion even when credible intervals show modest overlap: a \u0394PVUS whose 95% CI just crosses zero but whose P(A>B) = 0.98 represents strong evidence of superiority.'),

h3('Partial VUS by Disease Severity Region (PVUS)'),
p('PVUS(s_a, s_b) is the mean AUC(s) across severity grid points within a contiguous sub-range, on the same [0.5, 1.0] scale as the full VUS. Three regions use the SYNTAX trial tertiles:'),
bul('Mild CAD: SYNTAX 0\u201322'), bul('Intermediate CAD: SYNTAX 23\u201332'), bul('Severe CAD: SYNTAX \u226533'),
...blank(1),
p('Region boundaries are half-open [lo, hi) so every grid point belongs to exactly one region. The identity VUS = weighted mean of PVUS values holds exactly when all regions are estimable. 95% CI: bootstrap percentile method.'),

h3('Spectrum Gradient Index (SGI)'),
p('SGI = [AUC_fit(s_max_obs) \u2212 AUC_fit(s_min_obs)] / AUC_fit(s_max_obs) \u00d7 100%, using the fitted binormal surface and the severe-end AUC as denominator. Bounded in [0%, 100%]. Requires \u226550% severity coverage; otherwise n/e. 95% CI: bootstrap percentile.'),

h3('Quality Measures: MVF and ICV'),
p('The MVF (Missing Volume Fraction) is the fraction of the N\u209b = 50 severity grid points with no diseased patients, reported with Wilson 95% CI. MVF measures missingness across the full SYNTAX 1\u201360 grid. Because the three clinical tertile boundaries do not divide the grid into equal thirds \u2014 the severe region spans more grid points \u2014 a severe-only study yields MVF \u2248 56%, not 67%. The ICV (Imprecision of Covered Volume) is the fraction of characterised bins with CI Width Ratio > 0.25, also with Wilson 95% CI.'),

// TABLE 1
p([run('Table 1.',{bold:true,italics:true}),
   run(' Quality and severity-gradient measures of the VUS.',{italics:true})],{after:80}),
new Table({width:{size:W,type:WidthType.DXA},columnWidths:[1800,2000,2060,1700,1800],
  rows:[
    new TableRow({children:[hc('Measure',1800),hc('Quantity / failure mode',2000),
      hc('Statistical basis',2060),hc('CI method',1700),hc('Tuning parameter',1800)]}),
    new TableRow({children:[dl('MVF',1800,LTBLUE,true),
      dl('Fraction of severity grid with no data',2000,LTBLUE),
      dl('Bin presence/absence',2060,LTBLUE),dl('Wilson exact',1700,LTBLUE),dl('None',1800,LTBLUE)]}),
    new TableRow({children:[dl('ICV',1800,WHITE,true),
      dl('Fraction of characterised bins with imprecise AUC',2000),
      dl('Bootstrap CWR',2060),dl('Wilson exact',1700),dl('CWR threshold (0.25)',1800)]}),
    new TableRow({children:[dl('SGI',1800,LTBLUE,true),
      dl('% AUC lost at mild vs severe extreme',2000,LTBLUE),
      dl('Fitted surface AUC; \u226550% coverage',2060,LTBLUE),
      dl('Bootstrap percentile',1700,LTBLUE),dl('SYNTAX range; 50% coverage',1800,LTBLUE)]}),
    new TableRow({children:[dl('PVUS',1800,WHITE,true),
      dl('Region-specific performance + P(A>B)',2000),
      dl('Mean AUC(s) per SYNTAX tertile region',2060),
      dl('Bootstrap percentile',1700),dl('Region bounds (0\u201322, 23\u201332, \u226533)',1800)]}),
  ]}),
...blank(2),

// RESULTS
h1('Results'),rule(),

h3('Simulation 1: Paired Comparison'),
p(`Two hypothetical tests compared in the same full-spectrum cohort of 1,500 patients (SYNTAX 1\u201360). Naive AUC: Test 1 = ${f3(s1.nauc1)}, Test 2 = ${f3(s1.nauc2)}.`),
p('Figure 1 shows the distribution of diseased patients across SYNTAX bins for all study samples, making the spectrum gap of Test B immediately visible.'),
img('fig0_syntax_distribution.png',9200,4000),
cap('Figure 1. Diseased patients per SYNTAX bin (width 5 units). Background shading: blue = mild, yellow = intermediate, green = severe. Dashed line marks n* = 15. Red asterisks mark bins below threshold. Tests 1 and 2 span the full spectrum. Test B is entirely absent below SYNTAX 33.'),
...blank(1),
p([run('Table 2.',{bold:true,italics:true}),
   run(' Simulation 1: Paired comparison. All 95% CIs are bootstrap credible intervals. P(T1>T2) is the Bayesian posterior probability that Test 1 exceeds Test 2 in each bootstrap draw.',{italics:true})],{after:80}),
new Table({width:{size:W,type:WidthType.DXA},columnWidths:W2,rows:t2rows}),
...blank(1),
p([run('\u2020 ',{bold:true}),run('n/e = not estimable.',{size:19,italics:true})]),
...blank(1),
p(`Global VUS: Test 1 = ${fci(s1.vus1,s1.vus1_lo,s1.vus1_hi)}, Test 2 = ${fci(s1.vus2,s1.vus2_lo,s1.vus2_hi)}, \u0394VUS = ${fci(s1.dvus,s1.dvus_lo,s1.dvus_hi)}, P(T1>T2) = ${fp(s1.p_vus1_gt)}. Both SGI values indicate the tests lose approximately one third of their severe-end performance when applied at the mild extreme. The three PVUS values confirm Test 1\u2019s advantage is present and consistent across all severity regions, with P(T1>T2) = 1.000 in each region.`),
new Paragraph({children:[new PageBreak()]}),
img('fig1_sim1_domes.png',9200,4800),
cap(`Figure 2. VUS domes, Simulation 1. Left: Test 1 (VUS = ${f3(s1.vus1)} [${f3(s1.vus1_lo)}, ${f3(s1.vus1_hi)}]). Right: Test 2 (VUS = ${f3(s1.vus2)} [${f3(s1.vus2_lo)}, ${f3(s1.vus2_hi)}]). Coloured slices show ROC cross-sections at mild, intermediate, and severe severity. The translucent waterline marks the VUS. The back-wall curve traces the AUC(s) profile.`),

new Paragraph({children:[new PageBreak()]}),
h3('Simulation 2: Unpaired Comparison \u2014 Coverage-Gated Analysis'),
p(`Test A (full spectrum, n=1,500) vs Test B (severe only, n=500). Naive AUC: A = ${f3(s2.naucA)}, B = ${f3(s2.naucB)} \u2014 similar values concealing the severity-coverage difference.`),
img('fig2_sim2_naive_roc.png',9200,4000),
cap(`Figure 3. Naive ROC curves, Simulation 2. Test A (blue): AUC = ${f3(s2.naucA)}, n=1,500, full spectrum. Test B (red): AUC = ${f3(s2.naucB)}, n=500, severe only. Similar AUC values mask fundamentally different severity coverage.`),
...blank(1),
p([run('Table 3.',{bold:true,italics:true}),
   run(' Simulation 2: Unpaired coverage-gated comparison. Global VUS and \u0394VUS are n/e for Test B because its observed SYNTAX range covers only 43% of the full 0\u201360 range, below the 50% minimum. The comparison is restricted to the severe region where both tests have data.',{italics:true})],{after:80}),
new Table({width:{size:W,type:WidthType.DXA},columnWidths:W3,rows:t3rows}),
...blank(1),
p([run('\u2020 ',{bold:true}),run(`n/e = not estimable. Global VUS for Test B: ${s2.vusB_msg}. PVUS_mild and PVUS_intermediate for Test B: no diseased patients in those regions. Global \u0394VUS: not computed because Test B global VUS is n/e. MVF for Test B = ${(s2.mvfB*100).toFixed(0)}%: reflects the proportion of the SYNTAX 1\u201360 grid that is uncharacterised. Approximately 54% of the 50 grid points lie outside the severe region; one additional grid point at the upper extreme is also below n*, giving MVF = 56% rather than 67%.`,{size:19,italics:true})]),
...blank(1),

p(`The coverage-gated analysis produces a clear and honest result. Test A\u2019s global VUS = ${fci(s2.vusA,s2.vusA_lo,s2.vusA_hi)} characterises its performance across the full severity spectrum. Test B returns n/e for its global VUS because its observed SYNTAX range covers only 43% of the full range \u2014 this is the correct and transparent behaviour: a global VUS computed from severe-only data would be inflated and misleading.`),
p(`The comparison is restricted to the only region where both tests have data: the severe region. Here, PVUS_severe(A) = ${fpvus(s2.pvusA[2])} versus PVUS_severe(B) = ${fpvus(s2.pvusB[2])}, \u0394PVUS_severe = ${fpvus(s2.dpvusAB[2])}, with P(A>B) = ${fp(s2.p_pvusA_gt[2])}. The 95% credible interval for \u0394PVUS_severe barely excludes zero on one side, yet P(A>B) = ${fp(s2.p_pvusA_gt[2])} indicates strong evidence that Test A outperforms Test B even in the severe region where Test B has its best data. This is precisely the situation where reporting a credible interval alone would be misleading \u2014 the Bayesian posterior probability conveys the clinical conclusion clearly.`),
p('Test A\u2019s additional advantage in mild and intermediate disease \u2014 PVUS_mild = '
 +`${fpvus(s2.pvusA[0])}, PVUS_intermediate = ${fpvus(s2.pvusA[1])} \u2014 cannot be `
 +'compared against Test B at all because Test B has no data there. Rather than imputing '
 +'a value or computing a biased comparison, the framework returns n/e and makes this '
 +'absence of evidence explicit.'),
img('fig3_sim2_domes.png',9200,4800),
cap(`Figure 4. VUS domes, Simulation 2. Left: Test A (VUS = ${f3(s2.vusA)}), full spectrum. Right: Test B (VUS = n/e), severe only. Test B\u2019s dome covers only the high-SYNTAX region; mild and intermediate portions are absent. Global \u0394VUS cannot be computed. Only \u0394PVUS_severe is estimable.`),

new Paragraph({children:[new PageBreak()]}),
// DISCUSSION
h1('Discussion'),rule(),
p('This manuscript introduces the Volume Under the ROC Surface (VUS) as a three-dimensional extension of the ROC curve that explicitly accounts for disease severity. Three features distinguish the VUS from the conventional AUC.'),
p('First, the VUS shares the AUC\u2019s scale. It ranges from 0.5 (uninformative test) to 1.0 (perfect test), with the same intermediate zones that clinicians already use to interpret AUC. No new interpretive vocabulary is required.'),
p('Second, the VUS is accompanied by two quality measures and two severity-gradient descriptors. The Missing Volume Fraction (MVF) and the Imprecision of Covered Volume (ICV) quantify distinct failure modes in VUS estimation. The Spectrum Gradient Index (SGI) quantifies how much discriminatory ability the test loses at the mild end relative to the severe end, bounded in [0%, 100%]. The three Partial VUS values (PVUS_mild, PVUS_intermediate, PVUS_severe) decompose the overall VUS into severity-region-specific components, each reported with a credible interval and a Bayesian posterior probability of superiority.'),
p('Third, the coverage-first principle makes spectrum bias visible and honest rather than silently absorbed. When a test has been evaluated only in severe disease, the framework returns n/e for global VUS and for any PVUS region lacking data, and restricts comparisons between tests to regions where both have data. A severe-only study cannot make claims about mild disease performance \u2014 not because the statistic is unreliable, but because the data to estimate it do not exist.'),
p('The Bayesian posterior probability P(A>B) is reported alongside every credible interval for \u0394VUS and \u0394PVUS. This is particularly valuable when credible intervals show modest overlap: in Simulation 2, the 95% CI for \u0394PVUS_severe '
 +`[${fpvus(s2.dpvusAB[2])}] just reaches near zero at one boundary, yet P(A>B) = `
 +`${fp(s2.p_pvusA_gt[2])}, providing a clear and unambiguous clinical conclusion. `
 +'Relying on whether a CI crosses zero would be a binary and potentially misleading summary of the posterior evidence.'),
p(`A limitation of the current presentation is that the demonstration uses simulated data. Validation in real-world datasets remains an important next step. The code for VUS estimation is available on GitHub.`),

...blank(2),rule(),
p('Code and simulated datasets available on GitHub.  n* = 15.  M = 1,000 bootstrap draws.  Seed = 42.',
  {align:AlignmentType.CENTER,size:18,after:60}),

]}]});

Packer.toBuffer(doc).then(buf=>{
  fs.writeFileSync('/home/claude/VUS_Paper_v5.docx',buf);
  console.log('Done.');
});
