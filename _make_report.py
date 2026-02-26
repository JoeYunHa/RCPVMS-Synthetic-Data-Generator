"""
RCPVMS 합성 데이터 생성 방법론 — PDF 보고서 생성 스크립트
"""
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ─────────────────────────────────────────────────────────────────────────────
# Korean font registration
# ─────────────────────────────────────────────────────────────────────────────
try:
    pdfmetrics.registerFont(TTFont('KR', 'C:/Windows/Fonts/malgun.ttf'))
    pdfmetrics.registerFont(TTFont('KR-B', 'C:/Windows/Fonts/malgunbd.ttf'))
    FONT_R, FONT_B = 'KR', 'KR-B'
except Exception:
    FONT_R, FONT_B = 'Helvetica', 'Helvetica-Bold'

# ─────────────────────────────────────────────────────────────────────────────
# Palette & geometry
# ─────────────────────────────────────────────────────────────────────────────
C_NAVY   = HexColor('#1a3a5c')
C_BLUE   = HexColor('#2c7bb6')
C_LBLUE  = HexColor('#d0e4f7')
C_FORM   = HexColor('#eef4fb')
C_GRAY   = HexColor('#f5f5f5')
C_BORDER = HexColor('#c0c8d4')
C_TEXT   = HexColor('#1a1a2e')
C_DGRAY  = HexColor('#555555')

PAGE_W, PAGE_H = A4
MARGIN    = 2.2 * cm
CW        = PAGE_W - 2 * MARGIN   # content width


# ─────────────────────────────────────────────────────────────────────────────
# Style factory
# ─────────────────────────────────────────────────────────────────────────────
def S(name, **kw):
    defaults = dict(fontName=FONT_R, fontSize=10, textColor=C_TEXT, leading=16)
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)

STYLES = {
    'cover_main' : S('cm', fontName=FONT_B, fontSize=24, textColor=white,
                     alignment=TA_CENTER, leading=30, spaceAfter=4),
    'cover_sub'  : S('cs', fontName=FONT_R, fontSize=13, textColor=C_LBLUE,
                     alignment=TA_CENTER, leading=20, spaceAfter=4),
    'cover_meta' : S('cmt', fontName=FONT_R, fontSize=9, textColor=HexColor('#90bde0'),
                     alignment=TA_CENTER, leading=14),
    'sec_h'      : S('sh', fontName=FONT_B, fontSize=15, textColor=white,
                     leading=22),
    'h2'         : S('h2', fontName=FONT_B, fontSize=12, textColor=C_NAVY,
                     spaceBefore=10, spaceAfter=4, leading=18),
    'h3'         : S('h3', fontName=FONT_B, fontSize=10, textColor=C_BLUE,
                     spaceBefore=8, spaceAfter=3, leading=16),
    'body'       : S('bd', fontName=FONT_R, fontSize=10, textColor=C_TEXT,
                     alignment=TA_JUSTIFY, leading=17, spaceAfter=5),
    'formula'    : S('fm', fontName=FONT_B, fontSize=9.5, textColor=C_NAVY,
                     alignment=TA_LEFT, leading=17, spaceBefore=2, spaceAfter=2),
    'flabel'     : S('fl', fontName=FONT_R, fontSize=8.5, textColor=C_DGRAY,
                     alignment=TA_RIGHT, leading=13),
    'th'         : S('th', fontName=FONT_B, fontSize=9.5, textColor=white,
                     alignment=TA_CENTER, leading=14),
    'td'         : S('td', fontName=FONT_R, fontSize=9, textColor=C_TEXT,
                     alignment=TA_CENTER, leading=14),
    'td_l'       : S('tdl', fontName=FONT_R, fontSize=9, textColor=C_TEXT,
                     alignment=TA_LEFT, leading=14),
    'note'       : S('nt', fontName=FONT_R, fontSize=8, textColor=C_DGRAY,
                     alignment=TA_CENTER, leading=12),
    'bullet'     : S('bl', fontName=FONT_R, fontSize=10, textColor=C_TEXT,
                     leading=16, spaceAfter=3, leftIndent=10),
}

# ─────────────────────────────────────────────────────────────────────────────
# Flowable helpers
# ─────────────────────────────────────────────────────────────────────────────
def p(text, style='body'):
    return Paragraph(text, STYLES[style])

def sp(h=0.35):
    return Spacer(1, h * cm)

def hr_line():
    return HRFlowable(width=CW, thickness=0.5, color=C_BORDER,
                      spaceBefore=4, spaceAfter=4)

def sec_header(num, title):
    """Navy full-width section header."""
    inner = Table([[p(f'{num}.  {title}', 'sec_h')]], colWidths=[CW])
    inner.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), C_NAVY),
        ('TOPPADDING', (0,0),(-1,-1), 10),
        ('BOTTOMPADDING', (0,0),(-1,-1), 10),
        ('LEFTPADDING', (0,0),(-1,-1), 14),
        ('RIGHTPADDING', (0,0),(-1,-1), 14),
    ]))
    return inner

def h2_bar(title):
    """Left-bar subsection heading."""
    bar   = Table([['']], colWidths=[5])
    bar.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), C_BLUE),
        ('TOPPADDING', (0,0),(-1,-1), 0),
        ('BOTTOMPADDING', (0,0),(-1,-1), 0),
    ]))
    text  = Table([[p(title, 'h2')]], colWidths=[CW-5])
    text.setStyle(TableStyle([
        ('LEFTPADDING', (0,0),(-1,-1), 10),
        ('TOPPADDING', (0,0),(-1,-1), 4),
        ('BOTTOMPADDING', (0,0),(-1,-1), 4),
    ]))
    row = Table([[bar, text]], colWidths=[5, CW-5])
    row.setStyle(TableStyle([
        ('VALIGN', (0,0),(-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0),(-1,-1), 0),
        ('BOTTOMPADDING', (0,0),(-1,-1), 0),
        ('LEFTPADDING', (0,0),(-1,-1), 0),
        ('RIGHTPADDING', (0,0),(-1,-1), 0),
    ]))
    return row

def fbox(lines, label=None):
    """Formula box with light blue background."""
    rows = [[p(ln, 'formula')] for ln in lines]
    if label:
        rows.append([p(label, 'flabel')])
    inner = Table(rows, colWidths=[CW - 2.4*cm])
    ts = [
        ('BACKGROUND', (0,0),(-1,-1), C_FORM),
        ('TOPPADDING', (0,0),(-1,-1), 7),
        ('BOTTOMPADDING', (0,0),(-1,-1), 7),
        ('LEFTPADDING', (0,0),(-1,-1), 18),
        ('RIGHTPADDING', (0,0),(-1,-1), 18),
        ('BOX', (0,0),(-1,-1), 1, C_BLUE),
    ]
    if label:
        ts.append(('LINEABOVE', (0,-1),(-1,-1), 0.5, C_BORDER))
    inner.setStyle(TableStyle(ts))
    outer = Table([[inner]], colWidths=[CW])
    outer.setStyle(TableStyle([
        ('ALIGN', (0,0),(-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
    ]))
    return outer

def ptable(rows, headers=None, col_widths=None):
    """Styled data table. rows = list of lists of strings."""
    def _cell(txt, style):
        return Paragraph(str(txt), STYLES[style])

    data = []
    if headers:
        data.append([_cell(h, 'th') for h in headers])
    for row in rows:
        data.append([_cell(c, 'td') for c in row])

    n_cols = len(data[0])
    if col_widths is None:
        col_widths = [CW / n_cols] * n_cols

    t = Table(data, colWidths=col_widths, repeatRows=1 if headers else 0)
    ts = [
        ('TOPPADDING', (0,0),(-1,-1), 6),
        ('BOTTOMPADDING', (0,0),(-1,-1), 6),
        ('LEFTPADDING', (0,0),(-1,-1), 8),
        ('RIGHTPADDING', (0,0),(-1,-1), 8),
        ('GRID', (0,0),(-1,-1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [white, C_GRAY]),
        ('VALIGN', (0,0),(-1,-1), 'MIDDLE'),
    ]
    if headers:
        ts.append(('BACKGROUND', (0,0),(-1,0), C_NAVY))
    t.setStyle(TableStyle(ts))
    return t

def ptable_l(rows, headers=None, col_widths=None):
    """Left-aligned data table."""
    def _cell(txt, style):
        return Paragraph(str(txt), STYLES[style])
    data = []
    if headers:
        data.append([_cell(h, 'th') for h in headers])
    for row in rows:
        data.append([_cell(c, 'td_l') for c in row])
    n_cols = len(data[0])
    if col_widths is None:
        col_widths = [CW / n_cols] * n_cols
    t = Table(data, colWidths=col_widths, repeatRows=1 if headers else 0)
    ts = [
        ('TOPPADDING', (0,0),(-1,-1), 6),
        ('BOTTOMPADDING', (0,0),(-1,-1), 6),
        ('LEFTPADDING', (0,0),(-1,-1), 8),
        ('RIGHTPADDING', (0,0),(-1,-1), 8),
        ('GRID', (0,0),(-1,-1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [white, C_GRAY]),
        ('VALIGN', (0,0),(-1,-1), 'MIDDLE'),
    ]
    if headers:
        ts.append(('BACKGROUND', (0,0),(-1,0), C_NAVY))
    t.setStyle(TableStyle(ts))
    return t

# ─────────────────────────────────────────────────────────────────────────────
# Page template (header/footer)
# ─────────────────────────────────────────────────────────────────────────────
TODAY = datetime.date.today().strftime('%Y-%m-%d')

def on_page(canvas, doc):
    canvas.saveState()
    # header line
    canvas.setStrokeColor(C_NAVY)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_H - 1.5*cm, PAGE_W - MARGIN, PAGE_H - 1.5*cm)
    canvas.setFont(FONT_R, 8)
    canvas.setFillColor(C_DGRAY)
    canvas.drawString(MARGIN, PAGE_H - 1.3*cm,
                      'RCPVMS 합성 진동 데이터 생성 방법론')
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 1.3*cm, TODAY)
    # footer
    canvas.line(MARGIN, 1.5*cm, PAGE_W - MARGIN, 1.5*cm)
    canvas.drawCentredString(PAGE_W / 2, 1.1*cm, f'{doc.page}')
    canvas.restoreState()

def on_first_page(canvas, doc):
    canvas.saveState()
    canvas.restoreState()

# ─────────────────────────────────────────────────────────────────────────────
# Document builder
# ─────────────────────────────────────────────────────────────────────────────
def build():
    OUT = 'rcpvms_synthetic_data_report.pdf'
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=2.4*cm, bottomMargin=2.0*cm,
        title='RCPVMS 합성 진동 데이터 생성 방법론',
        author='rcpvms-gen',
    )

    story = []

    # =========================================================================
    # COVER
    # =========================================================================
    cover_rows = [
        [p('RCPVMS  합성 진동 데이터 생성 방법론', 'cover_main')],
        [p('Synthetic Vibration Data Generation Methodology', 'cover_sub')],
        [p('Jeffcott 로터 물리 모델 기반 결함 주입 시스템', 'cover_sub')],
        [p(f'작성일: {datetime.date.today().strftime("%Y년 %m월 %d일")}  |  ver 1.0', 'cover_meta')],
    ]
    cover = Table(cover_rows, colWidths=[CW])
    cover.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), C_NAVY),
        ('TOPPADDING', (0,0),(-1,-1), 20),
        ('BOTTOMPADDING', (0,0),(-1,-1), 20),
        ('LEFTPADDING', (0,0),(-1,-1), 18),
        ('RIGHTPADDING', (0,0),(-1,-1), 18),
    ]))
    story += [cover, sp(0.8)]

    story.append(p(
        '본 문서는 원자력 발전소 원자로냉각재펌프(RCP) 진동 감시 시스템(VMS)의 결함 분류 모델 학습을 위한 '
        '합성 진동 데이터 생성 방법론을 기술한다. 2-자유도 Jeffcott 로터 동역학 모델을 기반으로 '
        '불균형(Unbalance), 정렬 불량(Misalignment), 오일 휩(Oil Whip) 세 가지 결함 유형의 진동 신호를 '
        '물리적으로 타당하게 합성하며, 파라미터 다양성 조치를 통해 학습 데이터의 분포 편향을 최소화한다.',
        'body'))
    story.append(sp(0.5))

    spec_h = ['항목', '내용']
    spec_r = [
        ['대상 설비', 'RCP (원자로냉각재펌프) — 수직 중심, 저널 베어링'],
        ['운전 조건', '3600 RPM (1X=60 Hz) / 1200 RPM (1X=20 Hz)'],
        ['결함 유형', 'Unbalance / Misalignment / Oil Whip'],
        ['샘플링 주파수', '40,000 Hz  |  채널: 24채널 float32 non-interleaved'],
        ['파일 길이', '400,000 samples / file (10 초)'],
        ['생성 규모', '합성 900 파일 + 정상 110 파일 = 총 1,010 파일'],
    ]
    story.append(ptable(spec_r, headers=spec_h,
                         col_widths=[CW*0.32, CW*0.68]))
    story.append(PageBreak())

    # =========================================================================
    # 1. 개요
    # =========================================================================
    story += [sec_header('1', '개요'), sp(0.7)]
    story.append(p(
        'RCP VMS는 원전 안전 계통 펌프의 진동 상태를 실시간으로 감시하는 시스템이다. '
        '결함 분류 AI 모델 학습에는 다양한 결함 유형과 심각도의 레이블링된 데이터가 필요하나, '
        '운전 중 RCP에서 결함 데이터를 수집하는 것은 안전·운용 제약으로 현실적으로 불가능하다. '
        '이를 해결하기 위해 실측 정상 신호를 베이스라인으로 사용하고 '
        'Jeffcott 로터 모델에서 유도된 결함 특성 신호를 주입하는 합성 방법론을 채택한다.',
        'body'))
    story.append(sp(0.4))
    story.append(h2_bar('1.1  물리적 타당성 근거'))
    story.append(sp(0.3))
    for b in [
        '• 실측 1X 주파수를 FFT로 추출하여 실제 운전 속도에 기반한 물리 모델 적용',
        '• 주파수 응답함수 H(r)로 진폭 보정 — 물리적 한계를 초과하는 과주입 방지',
        '• 각 결함 유형의 지배 주파수(1X, 2X, 0.43~0.48X)가 이론 메커니즘과 일치',
        '• 베어링 강성 이방성(κ)을 반영하여 타원화/기울기 orbit 형상 재현',
        '• 오일 휩의 자려진동 성장 및 주파수 고착 현상을 시간 영역에서 동적 모델링',
    ]:
        story.append(p(b, 'bullet'))
    story.append(PageBreak())

    # =========================================================================
    # 2. 생성 파이프라인
    # =========================================================================
    story += [sec_header('2', '합성 데이터 생성 파이프라인'), sp(0.7)]
    story.append(p(
        '합성 데이터 생성은 5단계 순차 파이프라인으로 구성된다. '
        '각 단계는 물리적 타당성을 보장하며, 원본 BIN 헤더를 보존하여 '
        '실계측 시스템 호환성을 유지한다.', 'body'))
    story.append(sp(0.4))
    pipe_h = ['단계', '명칭', '주요 처리', '출력']
    pipe_r = [
        ['1 — Parse',    'BIN 파싱',     '헤더 + 24채널 float32 데이터 추출',       '채널별 신호 배열'],
        ['2 — Profile',  '신호 프로파일', 'FFT 1X 주파수 추출, RMS·노이즈 측정',   '운전속도(Hz), 기준 진폭'],
        ['3 — Generate', '결함 신호 생성', 'Jeffcott 모델, per-file 파라미터 랜덤화', 'X/Y 결함 신호 쌍'],
        ['4 — Inject',   '결함 주입',    '정상 채널에 결함 신호 가산, 클리핑 처리', '합성 24채널 신호'],
        ['5 — Save',     'BIN 저장',     '원본 헤더 보존 + 합성 데이터 재패키징',   '합성 .bin 파일'],
    ]
    story.append(ptable_l(pipe_r, headers=pipe_h,
                           col_widths=[CW*0.17, CW*0.13, CW*0.42, CW*0.28]))
    story.append(sp(0.7))

    story.append(h2_bar('2.1  가산 주입 (Additive Injection) 방식'))
    story.append(sp(0.3))
    story.append(p(
        '결함 주입은 가산(additive) 방식을 사용한다. 정상 신호에 결함 신호를 선형 합산함으로써 '
        '초기 결함(incipient)부터 심각 결함(critical)까지 연속적인 심각도 스펙트럼을 표현한다.',
        'body'))
    story.append(fbox([
        'x_syn(t) = x_normal(t)  +  x_fault(t)',
        'y_syn(t) = y_normal(t)  +  y_fault(t)',
    ], '채널별 합성 신호 (가산 주입)'))
    story.append(sp(0.5))

    story.append(h2_bar('2.2  Jeffcott 강제력 보정 (Amplitude Calibration)'))
    story.append(sp(0.3))
    story.append(p(
        '결함 신호 진폭은 실측 정상 신호의 1X 피크 진폭을 기준으로 Jeffcott 전달함수에 의해 역산출된다. '
        'severity=1.0일 때 결함 주파수 성분의 응답 진폭이 정상 1X 진폭과 동등해지도록 보정한다.',
        'body'))
    story.append(fbox([
        'A_1X  : 실측 정상 신호의 1X FFT 피크 진폭',
        '',
        'F_0 = A_1X / H(omega_fault)          [Jeffcott 모델 역산]',
        'F_scaled = severity x F_0             [심각도 스케일]',
        '',
        '=> 결함 응답 진폭 = severity x F_0 x H(omega_fault) = severity x A_1X',
    ], '결함 주파수별 강제력 보정 — 과주입 방지'))
    story.append(sp(0.4))
    cal_h = ['결함 유형', '보정 기준 주파수 omega_fault', '물리적 의미']
    cal_r = [
        ['Unbalance',    'omega_run  (1X)',     '불균형 가진력은 1X에서 발생'],
        ['Misalignment', '2 x omega_run  (2X)', '결함 응답이 2X에서 지배적'],
        ['Oil Whip',     '0.45 x omega_run',   '아유동 주파수에서 응답 보정'],
    ]
    story.append(ptable(cal_r, headers=cal_h,
                         col_widths=[CW*0.22, CW*0.33, CW*0.45]))
    story.append(PageBreak())

    # =========================================================================
    # 3. Jeffcott 로터 물리 모델
    # =========================================================================
    story += [sec_header('3', 'Jeffcott 로터 물리 모델'), sp(0.7)]
    story.append(p(
        'Jeffcott 로터 모델(Jeffcott 1919)은 회전 기계의 임계 속도 및 진동 응답을 설명하는 '
        '가장 기본적인 2-자유도 연속체 모델이다. 집중 질량(m), 탄성 축(강성 k_x, k_y), '
        '점성 감쇠(c)로 구성되며, RCP와 같은 수직 중심 펌프의 저차 굽힘 모드를 효과적으로 재현한다.',
        'body'))
    story.append(sp(0.5))

    # 3.1 운동 방정식
    story.append(h2_bar('3.1  2-자유도 운동 방정식'))
    story.append(sp(0.3))
    story.append(p(
        '회전 좌표계에서 디스크 중심의 X방향 및 Y방향 변위를 독립 자유도로 취급한다. '
        '베어링 강성이 방향에 따라 다를 수 있으므로(이방성), X축과 Y축의 고유 진동수를 분리하여 모델링한다.',
        'body'))
    story.append(fbox([
        'm x_ddot  +  2 zeta omega_nx m x_dot  +  omega_nx^2 m x  =  F_x(t)',
        'm y_ddot  +  2 zeta omega_ny m y_dot  +  omega_ny^2 m y  =  F_y(t)',
    ], '2-DOF Jeffcott 로터 운동 방정식'))
    story.append(sp(0.4))
    eq_h = ['기호', '의미', '단위', '본 시스템 범위']
    eq_r = [
        ['zeta (ζ)',       '점성 감쇠비 (damping ratio)',         '무차원', '0.03 – 0.08'],
        ['omega_nx',       'X축 임계 각속도 = omega_run / r0',    'rad/s',  '운전속도 기반 산출'],
        ['omega_ny',       'Y축 임계 각속도 = kappa x omega_nx',  'rad/s',  'kappa x omega_nx'],
        ['r0 (freq_ratio)','운전/임계속도 비율',                   '무차원', '0.55 – 0.80'],
        ['kappa (κ)',       'Y/X 임계 각속도 비율',                '무차원', '0.70 – 1.00'],
        ['F_x, F_y',       '결함 가진력 (결함 유형별 정의)',       '[a.u.]', '결함 모델 참조'],
    ]
    story.append(ptable(eq_r, headers=eq_h,
                         col_widths=[CW*0.18, CW*0.33, CW*0.15, CW*0.34]))
    story.append(sp(0.7))

    # 3.2 전달함수
    story.append(h2_bar('3.2  정상상태 주파수 응답함수 H(r)'))
    story.append(sp(0.3))
    story.append(p(
        '조화 가진에 대한 정상상태 응답은 무차원 주파수비 r = omega_exc / omega_n의 함수로 '
        '표현된다. 공진점(r=1)에서 H(r)는 1/(2ζ)로 최대화되며, 감쇠가 작을수록 급격한 진폭 증폭이 발생한다.',
        'body'))
    story.append(fbox([
        'H(r) = 1 / sqrt[ (1 - r^2)^2  +  (2 zeta r)^2 ]',
        '',
        'X축:  r_x = omega_exc / omega_nx',
        'Y축:  r_y = omega_exc / omega_ny    (kappa ≠ 1이면  r_x ≠ r_y)',
    ], '무차원 진폭 배율 (Magnification Factor)'))
    story.append(sp(0.4))
    story.append(p(
        '위상 지연 phi(r)는 가진력과 변위 응답 사이의 위상차이다. '
        '임계 속도 이하에서 0에 가깝고, 임계 속도 초과 시 180°에 수렴하며 '
        '임계 속도(r=1)에서는 감쇠비와 무관하게 정확히 90°이다.',
        'body'))
    story.append(fbox([
        'phi(r) = arctan( 2 zeta r  /  (1 - r^2) )',
    ], '위상 지연 (Phase Lag) — 감쇠비 zeta에 의해 완전히 결정됨'))
    story.append(sp(0.7))

    # 3.3 이방성
    story.append(h2_bar('3.3  베어링 강성 이방성 (kappa, κ)'))
    story.append(sp(0.3))
    story.append(p(
        '실제 저널 베어링은 유막의 하중 분포로 인해 X방향과 Y방향의 유효 강성이 다르다(이방성). '
        'kappa = omega_ny / omega_nx ≠ 1이면 동일 가진 주파수에서 두 축의 H(r)와 phi(r)가 달라지고, '
        '이 차이가 orbit의 타원화(elliptical orbit)와 기울기(tilt angle)를 형성한다.',
        'body'))
    story.append(fbox([
        'kappa = omega_ny / omega_nx = sqrt(k_y / k_x)',
        '',
        'kappa = 1.0  =>  등방성 베어링  =>  원형 orbit (unbalance 시)',
        'kappa < 1.0  =>  이방성 베어링  =>  타원 / 기울어진 banana orbit',
    ], '방향성 강성 이방성 — orbit 형상의 핵심 결정 인자'))
    story.append(sp(0.4))
    story.append(p(
        '본 시스템에서는 kappa를 0.70~1.00 범위에서 파일마다 독립 샘플링한다. '
        'RCP 저널 베어링의 실측 이방성 범위는 일반적으로 0.70~0.90이다. '
        'kappa가 1에 가까울수록 orbit은 원에 가까워지고, 0.70에 가까울수록 편평해진다.',
        'body'))
    story.append(PageBreak())

    # =========================================================================
    # 4. 결함 유형별 물리 모델
    # =========================================================================
    story += [sec_header('4', '결함 유형별 물리 모델'), sp(0.7)]

    # 4.1 Unbalance
    story.append(h2_bar('4.1  불균형 (Unbalance)'))
    story.append(sp(0.3))
    story.append(p(
        '불균형 결함은 회전 디스크의 질량 중심이 기하학적 회전축에서 편심 거리 e만큼 벗어날 때 발생한다. '
        '편심 질량(m_u)이 회전하면서 1X 동기 주파수의 원심력을 생성하고, '
        '이 힘이 베어링과 축을 통해 진동을 유발한다.',
        'body'))
    story.append(p('<b>가진력 모델</b>', 'h3'))
    story.append(fbox([
        'F_x(t) = F_0 * cos(omega t + phi_0)',
        'F_y(t) = F_0 * sin(omega t + phi_0)',
        '',
        'F_0 = m_u * e * omega^2    [원심력, 운전 속도 제곱에 비례]',
    ], '불균형 가진력 — 1X 동기 회전 원심력'))
    story.append(sp(0.3))
    story.append(p('<b>정상상태 변위 응답</b>', 'h3'))
    story.append(fbox([
        'X(t) = A * cos(omega t + phi_0 - phi)',
        'Y(t) = A * sin(omega t + phi_0 - phi)',
        '',
        'A = F_0 * H(omega)    [orbit 반경, Jeffcott 전달함수로 결정]',
        'phi = arctan(2 zeta r / (1-r^2))    [감쇠에 의한 위상 지연]',
    ], 'Jeffcott 정상상태 응답 — 1X 원형/타원 orbit'))
    story.append(sp(0.3))
    story.append(p(
        '<b>Orbit 특성:</b> 등방성(kappa=1)에서 완전한 원형, 이방성(kappa≠1)에서 타원. '
        'FFT 스펙트럼에서 <b>1X 성분이 지배적</b>으로 나타난다. '
        '운전 속도가 임계 속도에 근접할수록(r→1) orbit 반경이 급격히 증가한다.',
        'body'))
    story.append(sp(0.7))

    # 4.2 Misalignment
    story.append(h2_bar('4.2  정렬 불량 (Misalignment)'))
    story.append(sp(0.3))
    story.append(p(
        '정렬 불량은 커플링으로 연결된 두 축의 중심선이 일치하지 않을 때 발생한다. '
        '각 오정렬(angular misalignment)은 회전당 2회 주기적 강성 변동을 유발하여 '
        '<b>2X 주파수 가진</b>을 지배적으로 생성하며, 동시에 잔류 1X 성분(15~35%)이 공존한다.',
        'body'))
    story.append(p('<b>지배 2X 성분 — 방향성 강성 반영</b>', 'h3'))
    story.append(fbox([
        'X(t) = A_x2 * cos(2 omega t + phi_0 - phi_x2)    [omega_nx 기준 H, phi 적용]',
        'Y(t) = A_y2 * sin(2 omega t + phi_0 - phi_y2)    [omega_ny 기준 H, phi 적용]',
        '',
        'A_x2 = F_amp * H_x(2 omega),    A_y2 = F_amp * H_y(2 omega)',
        'H_x ≠ H_y  when  kappa ≠ 1   =>  비대칭 figure-8 orbit',
    ], '지배 2X 성분 — 이방성 시 X/Y 진폭 및 위상이 비대칭'))
    story.append(sp(0.3))
    story.append(p('<b>잔류 1X 성분</b>', 'h3'))
    story.append(fbox([
        'X(t) += A_x1 * cos(omega t + phi_0 - phi_x1)',
        'Y(t) += A_y1 * sin(omega t + phi_0 - phi_y1)',
        '',
        'A_x1 = A_x2 * r_1x,    A_y1 = A_y2 * r_1x',
        'r_1x ~ U(0.15, 0.35)    [파일마다 독립 샘플링]',
    ], '잔류 1X 결합 성분 — 출력 진폭 비율로 정의 (물리적 타당성 보장)'))
    story.append(sp(0.3))
    story.append(p(
        '<b>설계 원칙:</b> 잔류 1X 비율을 <b>출력 진폭 비율</b>로 정의하는 것이 중요하다. '
        '만약 강제력 비율(forcing ratio)로 정의하면, kappa<1에서 Y축 1X 공진 근방에 있을 때 '
        '1X 성분이 2X보다 커지는 물리적 오류가 발생한다. '
        '출력 비율 정의는 어떤 kappa에서도 "2X 지배, 1X 잔류" 특성을 항상 보장한다.',
        'body'))
    story.append(sp(0.3))
    story.append(p(
        '<b>Orbit 특성:</b> 비대칭 figure-8 / banana 형상. '
        'kappa=1 등방성에서 대칭 figure-8, kappa<1에서 기울어진 banana로 변형. '
        'FFT에서 <b>2X 지배적, 1X 잔류 성분 공존</b>.',
        'body'))
    story.append(PageBreak())

    # 4.3 Oil Whip
    story.append(h2_bar('4.3  오일 휩 (Oil Whip)'))
    story.append(sp(0.3))
    story.append(p(
        '오일 휩은 유체막 베어링(저널 베어링)에서 발생하는 자려 불안정(self-excited instability) 현상이다. '
        '유막 내 쐐기 작용(wedge effect)이 로터에 순 양(+)의 에너지를 공급하여, '
        '특정 임계 조건(운전 속도 ≥ 2 x 임계 속도) 초과 시 아유동 주파수에서 급격한 진폭 성장이 발생한다. '
        '본 모델은 오일 휩의 두 가지 핵심 비선형 현상을 시간 영역에서 명시적으로 구현한다.',
        'body'))
    story.append(sp(0.3))

    story.append(p('<b>(1) 주파수 고착 (Frequency Lock-in)</b>', 'h3'))
    story.append(p(
        '초기에는 아유동 주파수가 운전 속도를 비율적으로 추종하며 변화한다(오일 선회, oil whirl). '
        '임계 조건 초과 후 아유동 주파수가 특정 값에 고착되어 운전 속도 변화와 무관하게 유지된다. '
        '이 천이를 선형 처프(chirp)로 모델링한다.',
        'body'))
    story.append(fbox([
        'omega_locked = f_r x Omega    [고착 주파수, f_r ~ U(0.43, 0.48)]',
        '',
        '-- 고착 천이 구간 (T_lockin 초 동안 처프) --',
        'omega_inst(t) = linspace(0.88 x omega_locked, omega_locked, N_lockin cycles)',
        '',
        '-- 고착 이후 --',
        'omega_inst(t) = omega_locked  (상수)',
        '',
        'phi(t) = cumsum(omega_inst) / fs  +  phi_0    [위상 누적]',
    ], '주파수 고착 모델 — 선형 처프로 천이 재현'))
    story.append(sp(0.3))

    story.append(p('<b>(2) 자려진동 진폭 성장 (Self-excitation Growth)</b>', 'h3'))
    story.append(p(
        '유막이 로터에 순 양의 에너지를 공급함에 따라 진동 진폭이 지수적으로 성장한다. '
        '성장 시상수 tau는 유막의 에너지 공급 속도를 결정하는 핵심 파라미터이다.',
        'body'))
    story.append(fbox([
        'A(t) = A_max x (1 - exp(-t / tau))',
        '',
        'tau ~ U(1.0, 5.0) s    [자려 성장 시상수, 파일마다 랜덤 샘플링]',
        'tau -> 0  :  즉각 정상상태 (단순 레거시 모델)',
        'tau -> inf  :  극히 느린 성장 (발산 임계 근방)',
    ], '자려진동 진폭 성장 엔벨로프 — 지수 증가'))
    story.append(sp(0.3))

    story.append(p('<b>전체 시간 영역 모델</b>', 'h3'))
    story.append(fbox([
        'H_w  = H(omega_locked)    [고착 주파수에서의 Jeffcott 전달함수]',
        'phi_w = arctan(2 zeta r_w / (1 - r_w^2))',
        '',
        'X(t) = A_max x (1 - exp(-t/tau)) x H_w x cos(phi(t) - phi_w)',
        'Y(t) = A_max x (1 - exp(-t/tau)) x H_w x sin(phi(t) - phi_w)',
    ], '오일 휩 완전 시간 영역 모델 (처프 + 자려 성장 + Jeffcott 응답 합성)'))
    story.append(sp(0.3))
    story.append(p(
        '<b>Orbit 특성:</b> 전방 세차(forward whirl) 타원 orbit으로, '
        '운전 속도에 무관한 고정 아유동 주파수(0.43~0.48X)에서 궤적이 형성된다. '
        'FFT에서 <b>0.43~0.48X 성분이 지배적</b>이며 진폭이 시간에 따라 성장하는 비정상 패턴을 보인다.',
        'body'))
    story.append(PageBreak())

    # =========================================================================
    # 5. 과도(Transient) 결함 모드
    # =========================================================================
    story += [sec_header('5', '과도(Transient) 결함 모드'), sp(0.7)]
    story.append(p(
        '결함 초기 단계에서는 결함이 지속적으로 나타나는 것이 아니라 주기적으로 나타났다 사라지는 '
        '간헐적(intermittent) 패턴을 보인다. 이를 재현하기 위해 Hanning 엔벨로프 기반 과도 버스트 모드를 구현한다.',
        'body'))
    story.append(sp(0.4))
    story.append(h2_bar('5.1  과도 버스트 엔벨로프 모델'))
    story.append(sp(0.3))
    story.append(fbox([
        '주기 구조:',
        '  [ active_cycles 동안 Hanning 버스트 ]  +  [ silent_cycles 동안 무음(0) ]',
        '',
        'T_burst  = active_cycles / f_run      [s]',
        'T_silent = silent_cycles / f_run      [s]',
        'T_period = T_burst + T_silent',
        '',
        'envelope(t) = Hanning window  (버스트 구간, 부드러운 on/off)',
        'envelope(t) = 0               (무음 구간)',
        '',
        'x_fault_transient(t) = x_fault(t) * envelope(t)',
    ], '과도 결함 엔벨로프 — Hanning 윈도 주기 버스트'))
    story.append(sp(0.4))
    story.append(p(
        'active_cycles와 silent_cycles를 파일마다 독립 샘플링하여 '
        '다양한 duty ratio와 버스트 주기를 가진 과도 패턴을 생성한다.',
        'body'))
    story.append(sp(0.4))
    tr_h = ['파라미터', '설명', '연속 모드', '과도 모드 (생성 범위)']
    tr_r = [
        ['active_cycles', '결함 활성 구간 (회전 수)', 'N/A', 'U(1.0, 6.0) cycles'],
        ['silent_cycles', '무음 구간 (회전 수)',       'N/A', 'U(5.0, 20.0) cycles'],
        ['severity',      '심각도 스케일',             'U(0.5, 3.0)', 'U(0.5, 1.5)'],
    ]
    story.append(ptable(tr_r, headers=tr_h,
                         col_widths=[CW*0.22, CW*0.30, CW*0.20, CW*0.28]))
    story.append(PageBreak())

    # =========================================================================
    # 6. 데이터 다양성 조치
    # =========================================================================
    story += [sec_header('6', '데이터 다양성 조치'), sp(0.7)]
    story.append(p(
        '학습 데이터의 다양성 부족은 CNN 모델이 Jeffcott 신호의 고정 파라미터 패턴을 암기하게 만들어 '
        '실제 데이터에 대한 일반화를 저해한다. '
        '이를 방지하기 위해 세 가지 수준의 다양성 조치를 적용한다.',
        'body'))
    story.append(sp(0.5))

    story.append(h2_bar('6.1  Stage 1 — Jeffcott 물리 파라미터 per-file 랜덤화'))
    story.append(sp(0.3))
    story.append(p(
        '900개 합성 파일 각각에 대해 Jeffcott 모델의 모든 물리 파라미터를 독립적으로 샘플링한다. '
        '동일 결함 유형이라도 파일마다 고유한 orbit 형상, 스펙트럼 분포, 위상 특성을 갖게 된다.',
        'body'))
    story.append(sp(0.3))
    div_h = ['파라미터', '물리적 의미', '이전 (고정값)', '현재 (랜덤 범위)']
    div_r = [
        ['zeta (ζ)',       '점성 감쇠비',               '0.05 (고정)',      'U(0.03, 0.08)'],
        ['freq_ratio (r0)','운전/임계속도 비율',          '0.70 (고정)',      'U(0.55, 0.80)'],
        ['kappa (κ)',       '베어링 강성 이방성비',        '0.75 (전역 고정)', 'U(0.70, 1.00)'],
        ['oil_whip_tau',   '자려 성장 시상수 [s]',        '2.0 (고정)',       'U(1.0, 5.0)'],
        ['oil_whip_lockin','주파수 고착 획득 사이클 수',   '10 (고정)',        'U(5, 20)'],
        ['r_1x',           '잔류 1X/2X 진폭 비율 (misalignment)', '0.25 (고정)', 'U(0.15, 0.35)'],
        ['f_r',            '아유동/운전속도 주파수비 (oil whip)',  '0.45 (고정)', 'U(0.43, 0.48)'],
    ]
    story.append(ptable(div_r, headers=div_h,
                         col_widths=[CW*0.20, CW*0.30, CW*0.22, CW*0.28]))
    story.append(sp(0.7))

    story.append(h2_bar('6.2  Stage 2 — Transient 파라미터 per-file 랜덤화'))
    story.append(sp(0.3))
    story.append(p(
        '과도 모드의 버스트 주기 파라미터도 파일마다 독립 샘플링한다. '
        'active_cycles와 silent_cycles의 변동은 다양한 duty ratio와 버스트 패턴을 생성하여 '
        '모델이 특정 버스트 주기에 과적합하는 것을 방지한다.',
        'body'))
    story.append(sp(0.7))

    story.append(h2_bar('6.3  Stage 3 — 베어링 쌍별 독립 진폭 변조'))
    story.append(sp(0.3))
    story.append(p(
        '실제 결함은 결함 위치와 각 베어링의 거리·전달 경로에 따라 측정 진폭이 다르게 나타난다. '
        '이를 재현하기 위해 24채널 = 12 베어링 쌍 각각에 독립 스케일을 적용한다.',
        'body'))
    story.append(fbox([
        's_i ~ U(0.50, 1.50)  for i = 0, 1, ..., 11  [12 베어링 쌍 각각 독립 샘플링]',
        '',
        'x_fault_ch(i)(t) = s_i x x_fault(t)    [X 채널]',
        'y_fault_ch(i)(t) = s_i x y_fault(t)    [Y 채널, 동일 쌍 내 동일 스케일]',
    ], '베어링별 독립 진폭 변조 — 결함 공간 전파 다양화'))
    story.append(sp(0.3))
    story.append(p(
        '동일 베어링 쌍(X/Y)에는 동일 스케일을 적용하므로 orbit의 종횡비(aspect ratio)는 유지되며, '
        '베어링 간 상대 진폭만 다양화된다. 이 정보는 adaptive axis scaling을 통해 '
        '2D CNN 입력 이미지에도 상대 크기 차이로 보존된다.',
        'body'))
    story.append(sp(0.5))

    other_h = ['다양성 요인', '범위 / 방법', '적용 위치']
    other_r = [
        ['Severity (심각도)',   'U(0.5, 3.0) per file',        'main.py — batch mode'],
        ['초기 위상 phi_0',    'U(0, 2pi) per file',          'main.py — generate_*()'],
        ['RPM 지터',           '± 0.05 Hz per file',          'JeffcottGenerator.__init__'],
        ['베이스 정상 파일',   'pool에서 무작위 선택',         'main.py — random.choice'],
        ['Train augmentation', 'RandomRotation(±180°) + HFlip', 'train_orbit_cnn.py'],
    ]
    story.append(ptable_l(other_r, headers=other_h,
                           col_widths=[CW*0.25, CW*0.35, CW*0.40]))
    story.append(PageBreak())

    # =========================================================================
    # 7. Orbit 이미지 생성
    # =========================================================================
    story += [sec_header('7', 'Orbit 이미지 생성 (2D CNN 입력)'), sp(0.7)]
    story.append(p(
        'Orbit 이미지는 X-Y 변위 신호 쌍을 2차원 밀도 맵으로 변환한 것이다. '
        '회전체 축 궤적(shaft centerline orbit)을 시각화하며 '
        '결함 유형에 따라 특징적인 형상(원, figure-8, 타원)을 나타낸다. '
        '4개 베어링 쌍의 orbit을 채널로 쌓아 (4, 256, 256) 텐서를 CNN에 입력한다.',
        'body'))
    story.append(sp(0.4))

    story.append(h2_bar('7.1  채널 선택 및 물리 단위 변환'))
    story.append(sp(0.3))
    ch_h = ['채널 인덱스', 'X/Y 쌍', '위치']
    ch_r = [
        ['0', '(ch 0, ch 1)',   '베어링 A — X / Y'],
        ['1', '(ch 4, ch 5)',   '베어링 B — X / Y'],
        ['2', '(ch 10, ch 11)', '베어링 C — X / Y'],
        ['3', '(ch 16, ch 17)', '베어링 D — X / Y'],
    ]
    story.append(ptable(ch_r, headers=ch_h,
                         col_widths=[CW*0.20, CW*0.30, CW*0.50]))
    story.append(sp(0.3))
    story.append(fbox([
        'x_mil = (x_raw - mean(x_raw)) x mils_per_v    [DC 제거 + 단위 변환]',
        'y_mil = (y_raw - mean(y_raw)) x mils_per_v',
    ], 'DC 제거 및 mils 변환 (volt_to_mil)'))
    story.append(sp(0.5))

    story.append(h2_bar('7.2  2D 밀도 이미지 생성'))
    story.append(sp(0.3))
    story.append(fbox([
        'scale = (img_size - 1) / (2 x axis_lim)',
        'p_x[i] = int( (x_mil[i] + axis_lim) x scale )',
        'p_y[i] = int( (y_mil[i] + axis_lim) x scale )',
        '',
        'img[p_y, p_x] += 1.0    [방문 횟수(visit count) 누적]',
        'img /= max(img)          [정규화 -> [0, 1]]',
    ], '256 x 256 orbit 밀도 이미지 생성'))
    story.append(sp(0.5))

    story.append(h2_bar('7.3  Per-file Adaptive Axis Scaling (Option A)'))
    story.append(sp(0.3))
    story.append(p(
        '고정 axis_lim(3.0 mils)은 파일 간 진폭 편차가 큰 데이터에서 '
        '고진폭 클리핑 또는 저진폭 중앙 집중 문제를 유발한다. '
        '이를 해결하기 위해 파일마다 모든 4개 채널의 orbit 반경 p99.5를 기준으로 '
        'axis_lim을 적응적으로 산출한다.',
        'body'))
    story.append(fbox([
        'r_i = sqrt(x_mil_i^2 + y_mil_i^2)    for all samples in all 4 bearing pairs',
        '',
        'axis_lim = max( percentile(all_r, 99.5),  0.1 mils )    [최소값 clamp]',
        '',
        '=> 모든 4채널에 동일 axis_lim 적용  (채널 간 상대 진폭 보존)',
    ], 'Per-file Global Adaptive Axis Scaling — Option A'))
    story.append(sp(0.3))
    story.append(p(
        '<b>Option A 선택 이유:</b> 동일 파일 내 4개 채널에 같은 axis_lim을 적용함으로써 '
        'Stage 3에서 부여한 베어링별 진폭 차이가 이미지 내 상대 크기 차이로 그대로 보존된다. '
        'Per-channel 방식(Option B)에서는 각 채널이 독립적으로 정규화되어 '
        'Stage 3의 다양성 정보가 완전히 소멸한다.',
        'body'))
    story.append(PageBreak())

    # =========================================================================
    # 8. 데이터셋 구성 및 검증
    # =========================================================================
    story += [sec_header('8', '데이터셋 구성 및 검증'), sp(0.7)]

    story.append(h2_bar('8.1  전체 데이터셋 구성'))
    story.append(sp(0.3))
    ds_h = ['RPM', '레이블', '유형', '파일 수', '심각도 범위']
    ds_r = [
        ['3600 rpm', 'Unbalance',    '연속', '100', 'U(0.5, 3.0)'],
        ['3600 rpm', 'Unbalance',    '과도',  '50', 'U(0.5, 1.5)'],
        ['3600 rpm', 'Misalignment', '연속', '100', 'U(0.5, 3.0)'],
        ['3600 rpm', 'Misalignment', '과도',  '50', 'U(0.5, 1.5)'],
        ['3600 rpm', 'Oil Whip',     '연속', '100', 'U(0.5, 3.0)'],
        ['3600 rpm', 'Oil Whip',     '과도',  '50', 'U(0.5, 1.5)'],
        ['1200 rpm', 'Unbalance',    '연속', '100', 'U(0.5, 3.0)'],
        ['1200 rpm', 'Unbalance',    '과도',  '50', 'U(0.5, 1.5)'],
        ['1200 rpm', 'Misalignment', '연속', '100', 'U(0.5, 3.0)'],
        ['1200 rpm', 'Misalignment', '과도',  '50', 'U(0.5, 1.5)'],
        ['1200 rpm', 'Oil Whip',     '연속', '100', 'U(0.5, 3.0)'],
        ['1200 rpm', 'Oil Whip',     '과도',  '50', 'U(0.5, 1.5)'],
        ['합계',     '—',            '—',    '900', '—'],
    ]
    story.append(ptable(ds_r, headers=ds_h,
                         col_widths=[CW*0.15, CW*0.20, CW*0.12, CW*0.15, CW*0.38]))
    story.append(sp(0.3))
    story.append(p('정상 데이터: 3600 rpm 28파일 + 1200 rpm 82파일 = 총 110파일 (실측 원본)', 'body'))
    story.append(sp(0.6))

    story.append(h2_bar('8.2  FFT 기반 검증 기준'))
    story.append(sp(0.3))
    val_h = ['결함 유형', '지배 주파수', '합격 조건']
    val_r = [
        ['Unbalance',    '1X = f_run ± 2 Hz',       '1X 피크 > 노이즈 바닥 (유의미)'],
        ['Misalignment', '2X = 2 x f_run ± 4 Hz',   '2X 피크 > 1X 피크 (severity >= 1.0)'],
        ['Oil Whip',     '0.43~0.48 x f_run',        '아유동 피크 >= 40% of 1X'],
    ]
    story.append(ptable_l(val_r, headers=val_h,
                           col_widths=[CW*0.20, CW*0.30, CW*0.50]))
    story.append(sp(0.6))

    story.append(h2_bar('8.3  코드 모듈 참조'))
    story.append(sp(0.3))
    code_h = ['모듈', '파일', '역할']
    code_r = [
        ['JeffcottGenerator',  'src/core/generator.py',    '3종 결함 신호 생성 (물리 모델)'],
        ['RCPVMSSynthesizer',  'src/core/synthesizer.py',  '결함 주입 + BIN 저장'],
        ['Pipeline 진입점',    'main.py',                  '단일 파일 생성 + 강제력 보정'],
        ['배치 생성',          'generate_all.py',          '900파일 일괄 생성 스크립트'],
        ['Orbit 생성',         'src/utils/orbit.py',       '2D 밀도 이미지 + adaptive scaling'],
        ['Dataset',            'src/datasets/orbit_dataset.py', 'BIN 직독 orbit 데이터셋'],
        ['검증',               'validate_synthetic.py',    'FFT 기반 합성 데이터 타당성 검증'],
    ]
    story.append(ptable_l(code_r, headers=code_h,
                           col_widths=[CW*0.22, CW*0.28, CW*0.50]))

    # Footer
    story += [sp(1.5), hr_line(),
              p(f'본 문서는 rcpvms-gen 프로젝트 기술 문서입니다.  생성일: {TODAY}', 'note')]

    doc.build(story,
              onFirstPage=on_first_page,
              onLaterPages=on_page)
    print(f'PDF saved: {OUT}')


if __name__ == '__main__':
    build()
