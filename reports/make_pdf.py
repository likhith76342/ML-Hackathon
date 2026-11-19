from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'PathSnap_Final_Report.pdf'
md = ROOT / 'reports' / 'PathSnap_Final_Report.md'
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportTitle', parent=styles['Title'], fontSize=24, leading=29, alignment=TA_CENTER, textColor=colors.HexColor('#17365D'), spaceAfter=12))
styles.add(ParagraphStyle(name='SubTitle', parent=styles['Normal'], fontSize=14, leading=18, alignment=TA_CENTER, textColor=colors.HexColor('#2F5597'), spaceAfter=18))
styles.add(ParagraphStyle(name='H1x', parent=styles['Heading1'], fontSize=16, leading=20, textColor=colors.HexColor('#17365D'), spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle(name='H2x', parent=styles['Heading2'], fontSize=12, leading=15, textColor=colors.HexColor('#2F5597'), spaceBefore=8, spaceAfter=4))
styles['BodyText'].fontSize=9.5; styles['BodyText'].leading=13

def esc(s):
    return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

story=[]
story += [Spacer(1, 1.0*inch), Paragraph('PathSnap', styles['ReportTitle']), Paragraph('GPS Map Matching and Route Reconstruction', styles['SubTitle']), Paragraph('Final Project Report', styles['SubTitle']), Paragraph('PES University | Machine Learning Hackathon', styles['SubTitle']), PageBreak()]
lines=md.read_text(encoding='utf-8').splitlines(); i=0
while i<len(lines):
    line=lines[i].strip()
    if not line or line.startswith('# PathSnap') or line.startswith('**Team members**') or line.startswith('---'):
        i+=1; continue
    if line.startswith('## '): story += [Spacer(1,5), Paragraph(esc(line[3:]), styles['H1x'])]; i+=1; continue
    if line.startswith('### '): story += [Paragraph(esc(line[4:]), styles['H2x'])]; i+=1; continue
    if line.startswith('|'):
        block=[]
        while i<len(lines) and lines[i].strip().startswith('|'): block.append([esc(x.strip()) for x in lines[i].strip().strip('|').split('|')]); i+=1
        if len(block)>1:
            block=[r for r in block if not all(set(x)<=set('-: ') for x in r)]
            t=Table(block, repeatRows=1, hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17365D')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),7.5),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#D9D9D9')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F3F6F9')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)])); story += [t,Spacer(1,8)]
        continue
    if line.startswith('**Screenshot insertion points:**'):
        story.append(Paragraph('Screenshot placeholders', styles['H1x']))
        for n,title in [(1,'API real-mode startup'),(2,'Trip 15 result'),(3,'Run-all evaluator table'),(4,'API prediction log')]:
            story += [Paragraph(f'Screenshot {n}: {title}', styles['H2x']), Spacer(1,18), Paragraph('PASTE SCREENSHOT HERE', styles['BodyText']), Spacer(1,18)]
        i+=1; continue
    text=esc(line).replace('`','')
    if text.startswith('1. ') or text.startswith('2. ') or text.startswith('3. ') or text.startswith('4. ') or text.startswith('5. '): story.append(Paragraph(text, styles['BodyText']))
    elif text: story.append(Paragraph(text, styles['BodyText']))
    story.append(Spacer(1,4)); i+=1
for fn,caption in [('eda_trip_size_distribution.png','Trip-size distribution'),('eda_geographic_coverage.png','Geographic coverage'),('model_comparison_cv_top1.png','Cross-validation Top-1 comparison'),('model_lightgbm_feature_importance.png','LightGBM feature importance'),('model_lightgbm_loss_curve.png','LightGBM loss curve'),('hmm_vs_argmax_lightgbm.png','HMM versus classifier-only route reconstruction')]:
    p=ROOT/'reports'/'figures'/fn
    if p.exists(): story += [PageBreak(), Image(str(p), width=6.4*inch, height=4.1*inch), Paragraph(caption, styles['BodyText'])]
doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=.65*inch,leftMargin=.65*inch,topMargin=.65*inch,bottomMargin=.65*inch)
doc.build(story)
print(OUT)
