import os
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()
section = doc.sections[0]
section.top_margin = Inches(0.7)
section.bottom_margin = Inches(0.7)
section.left_margin = Inches(0.8)
section.right_margin = Inches(0.8)
doc.styles['Normal'].font.name = 'Aptos'
doc.styles['Normal'].font.size = Pt(10)
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('PathSnap\nGPS Map Matching and Route Reconstruction\nFinal Project Report')
run.bold = True
run.font.size = Pt(24)
doc.add_paragraph('PES University | Machine Learning Hackathon').alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_page_break()
for line in open('reports/PathSnap_Final_Report.md', encoding='utf-8'):
    line = line.rstrip()
    if line.startswith('## '):
        doc.add_heading(line[3:], 1)
    elif line.startswith('### '):
        doc.add_heading(line[4:], 2)
    elif line.startswith('|') or line.startswith('---'):
        continue
    elif line.strip() and not line.startswith('# '):
        doc.add_paragraph(line.replace('`', ''))
doc.add_page_break()
doc.add_heading('Committed Result Figures', 1)
doc.add_paragraph('The following figures are generated from committed project outputs and are included as evidence for the corresponding analyses. No figures were manually recreated or numerically edited.')
figures = [
    ('reports/figures/eda_trip_size_distribution.png', 'Figure 1. GPS point-count distribution by analysis partition.'),
    ('reports/figures/eda_geographic_coverage.png', 'Figure 2. Geographic coverage of the train and student_test partitions.'),
    ('reports/figures/model_comparison_cv_top1.png', 'Figure 3. Grouped cross-validation Top-1 comparison across the five models.'),
    ('reports/figures/model_lightgbm_feature_importance.png', 'Figure 4. LightGBM feature importance for the primary model run.'),
    ('reports/figures/model_lightgbm_loss_curve.png', 'Figure 5. LightGBM validation loss across boosting iterations.'),
    ('reports/figures/hmm_vs_argmax_lightgbm.png', 'Figure 6. Final LightGBM classifier-only versus HMM route reconstruction.'),
    ('reports/figures/hmm_beta_sensitivity_lightgbm.png', 'Figure 7. HMM beta sensitivity using LightGBM emissions.'),
]
for path, caption in figures:
    if os.path.exists(path):
        p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(path, width=Inches(6.15))
        p=doc.add_paragraph(caption); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs: r.italic=True; r.font.size=Pt(9)
for title, text in [
    ('Screenshot 1: API real-mode startup', 'Paste the terminal showing REAL mode, final_model.joblib, evaluator_test, and 10 trips.'),
    ('Screenshot 2: Trip 15 result', 'Paste the Trip 15 UI showing metrics, map, 6,854 points, and 4.53 seconds.'),
    ('Screenshot 3: Run-all table', 'Paste the completed evaluator table showing all 10 trips with status ok.'),
    ('Screenshot 4: API prediction log', 'Paste the terminal showing HTTP 200 POST /predict lines and timings.'),
]:
    doc.add_heading(title, 2)
    doc.add_paragraph('PASTE SCREENSHOT HERE')
    doc.add_paragraph(text)
doc.save('reports/PathSnap_Final_Report.docx')
