"""
eCRF Chart Generator v1.6 — Streamlit Edition
===============================================
Usage:
    streamlit run ecrf_chart_generator_streamlit.py
"""

import io
import re
from collections import OrderedDict, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
from difflib import SequenceMatcher
from streamlit_sortables import sort_items
from scipy import stats

# ═══════════════════════════════════════════════════════════════
# 1. CONFIGURATION & STYLE
# ═══════════════════════════════════════════════════════════════

CENTER_FILTER = "VCS"   

METADATA_COLS = {
    "STUDY REFERENCE", "SUBJECT ID", "STATUS", "STUDY CENTER ABBREV",
    "STUDY CENTER NAME", "RANDOMISATION ID", "RANDOMIZATION ID",
    "RANDOMISATION DATE", "RANDOMIZATION DATE",
    "RANDOMISATION GROUP", "RANDOMIZATION GROUP",
    "RANDOMISATION KIT", "RANDOMIZATION KIT",
    "START DATE", "LAST UPDATE DATE", "OCCURRENCE NO",
}

COL_SUBJECT = "SUBJECT ID"
COL_STATUS  = "STATUS"
COL_CENTER  = "STUDY CENTER ABBREV"
COL_STUDY   = "STUDY REFERENCE"

INCLUDED_STATUSES = {"COMPLETED", "VERIFICATION", "IN_PROGRESS", "QUERIES"}

EXCLUDED_PARAM_NAMES = {
    "ACC", "ADDPRD", "ADDPRD_1", "ADDPRNU", "ADDPRNU_1", "ADDPRO", "ADDPROD",
    "ADDPRON", "ADDPRNUM", "AGE", "AGSKIN", "ALLER", "ALLERGY", "APP", "APPLY",
    "APRCOLL", "ASSESS", "ASSBEG", "ASSBEGIN", "_ASSBEGIN",
    "BEGIN", "CANCER", "CELL", "CHAIRSTYLE", "CHEMSUN", "CHRO", "CLEAN",
    "COMMUN", "COOP", "CORNEO", "CUTO",
    "DERM", "DISC", "DISCON", "DLINST", "DLPROD", "EMPLOY", "FACETAT", "FAGRAD",
    "HORMON", "HORMONES", "GROOM", "IMMUNO", "INC",
    "LAST UPDATE DATE", "LOG", "LOGREV", "MEDCOND", "MEDHX", "MOBILE", "NCHTAT",
    "OCONTRA", "ORCONT", "OTHSTU", "PCOLL", "PHOTO", "PHOTO_1", "_PHOTOSELEC",
    "_PHOTSEL", "PHOTSEL", "PMAKEUP", "PNOTIN", "POUT", "PRCOLL", "PRDISP",
    "PREG", "PRINS", "PRLOINSP", "PRNUM", "PRET", "PROCEDURE", "PROD",
    "PROINS", "PRONUM",
    "RETTIM", "RPNUM", "SCAR", "SIGNIC", "SIGNPMR", "SKINDIS", "SHAVE",
    "START DATE", "STATUS", "STUDY CENTER ABBREV", "STUDY CENTER NAME",
    "STUDY REFERENCE", "SUBJECT ID", "SUPPRO",
    "TEWAHEX", "TOPMED", "TRET", "TUB", "TATSCAR", "UNABLE", "UNCMED",
    "UNREL", "VAPOEIN", "UNRELI", "WOPRO",
    "COLLPROD", "EXOUTPR", "LOTNUM", "MANUF", "OUTPROD", "PRESFORM",
    "PRODDIS", "PRODNAME", "PRODNUM", "RXCOLL", "TESTPROD", "TONPR", "TPRET",
    "DAILYLOG", "PREAPP", "POSTAPP", "BAREPHOTO", "PHOTOCAT", "PHOTOSEL",
    "EMPLOYEE", "OTHSTU", "UNCMED", "UNABLE", "WEAKLASH",
    "LASHEXT", "SIGNPMR", "SIGNIC",
}

METADATA_SUFFIXES  = {"EIN", "CAL", "SET", "CFG", "MODE", "SETTING", "CONFIG"}
WHITELIST_KEYWORDS = {"OVERCOV", "COVERAGE", "WEAR", "OVERALL", "UEDISC"}

EXPERT_GRADING_KEYWORDS = [
    "SEVER", "SEVERITY", "SIZE", "LSIZE", "RESIDUE", "APPEAR", "LESION",
    "FLW", "FLWIN", "WRINK", "LINES", "FINE", "CROW", "NASOL", "CREASE",
    "SKTEXT", "VSKTEXT", "SMOOTH", "SOFT", "SUPPLE", "CREP",
    "FIRM", "FIRMN", "ELAST", "TELAST", "LIFT", "LIFSAG", "SAG",
    "CONT", "CONTOUR", "PLUMP", "VOLUME", "VOL",
    "NECKLI", "NECKLW", "NECKWR", "NKWR",
    "UEBAGS", "EYEBAG", "PUFF", "DARK", "UEDISC",
    "PORE", "POREAPP", "PORES",
    "EVEN", "SKTEVE", "SKTEVEN", "SKCLAR", "CLAR", "TONE",
    "RADLUM", "BRIGHT", "LUM", "GLOW", "SHINE",
    "PIH", "PIGMENT", "DISCOL", "AGESPOT", "DARKSP", "CHYPER",
    "OVERALL", "COMPLEXH", "SCHEALTH", "SKQUAL", "HEALTH",
    "BHEAD", "LASH", "SALLOW", "HTHICK", "HASHINE", "HAMOIST",
    "HASOFT", "HASILK", "HARES", "COMB", "MANAGE", "FRIZZ",
    "CURLBOUN", "CURLDEF", "SPLENDS", "SPRBOUN",
    "STIFNESS", "HAIRHEAL",
    "COVER", "UNIFORM", "OVEROV",
    "MASCFLAK", "MASCSMUD", "FEATHER", "LIPBLE",
    "LTHICK", "LLENGTH", "LVOL", "LSHINE", "LCURL",
]

TOLERANCE_KEYWORDS = [
    "DRYNESS", "ERYTHEMA", "EDEMA", "BURNING", "STINGING",
    "ITCHING", "TINGLING", "PEELING", "FLAKING",
    "DRY", "ERYTH", "BURN", "STING", "ITCH", "TINGLE", "PEEL", "FLAKE",
]

COMEDOGENICITY_KEYWORDS = [
    "COMED", "ACNE", "BLACKHEAD", "WHITEHEAD", "PAPULE", "PUSTULE",
    "MICROC", "OPEN_C", "CLOSED_C", "NONINF", "INFLAM",
]

INSTRUMENT_KEYWORDS = [
    "VM", "TEWL", "MEP", "ITA", "CUTO", "CHROMA", "SEBO",
    "MELAN", "ERIT", "HYDRA", "FIRM", "ELAST",
    "VAPOMETER", "TEWAMETER", "CORNEOMETER", "CUTOMETER",
    "MEXAMETER", "CHROMAMETER", "SEBUMETER", "MOISTUREMETER",
    "ANTERA", "VISIOFACE", "PRIMOS", "VISIA",
]

MISSING_MARKERS = {"", ".", ".UNK", ".HIDDEN", "ND", "N/A", "#N/A", ".ND"}

KNOWN_TP_ORDER = [
    "SC", "BL", "PRE",
    "15MIN", "30MIN", "45MIN",
    "1H", "2H", "4H", "6H", "8H", "12H",
    "IMM", "IM", "POST",
    "W1", "W2", "W4", "W6", "W8", "W10", "W12",
    "W16", "W20", "W24", "W26", "W36", "W48", "W52",
    "M1", "M2", "M3", "M6", "M9", "M12",
    "D1", "D2", "D3", "D7", "D14", "D28",
]

TP_DISPLAY = {
    "SC": "Screening", "BL": "Baseline", "PRE": "Pre-Application",
    "15MIN": "15 Minutes", "30MIN": "30 Minutes", "45MIN": "45 Minutes",
    "1H": "1 Hour", "2H": "2 Hours", "4H": "4 Hours",
    "6H": "6 Hours", "8H": "8 Hours", "12H": "12 Hours",
    "IMM": "Immediate", "IM": "Immediate", "POST": "Post-Application",
    "W1": "Week 1", "W2": "Week 2", "W4": "Week 4",
    "W6": "Week 6", "W8": "Week 8", "W10": "Week 10",
    "W12": "Week 12", "W16": "Week 16", "W20": "Week 20",
    "W24": "Week 24", "W26": "Week 26", "W36": "Week 36",
    "W48": "Week 48", "W52": "Week 52",
    "M1": "Month 1", "M2": "Month 2", "M3": "Month 3",
    "M6": "Month 6", "M9": "Month 9", "M12": "Month 12",
    "D1": "Day 1", "D2": "Day 2", "D3": "Day 3",
    "D7": "Day 7", "D14": "Day 14", "D28": "Day 28",
}

COLORS = {
    'baseline':  '#949494', 'timepoint':  '#0173B2',
    'improved':  '#029E73', 'worsened':   '#D55E00',
    'no_change': '#949494', 'text_main':  '#333333',
    'text_sub':  '#666666', 'pct_color':  '#333333',
    'pill_bg':   '#EEEEEE',
}

ANALYSIS_MODES = [
    "All Parameters",
    "Expert Grading",
    "Tolerance Grading",
    "Comedogenicity Grading",
    "Instrument",
    "pH Level Strips",
    "Modified Schirmer Test",
]

# ASFS threshold bands: (label, low_inclusive, high_inclusive)
ASFS_BANDS = [
    ("Normal, Very Slight",  0, 15),
    ("Mild",                16, 24),
    ("Moderate",            25, 34),
    ("Severe",              35, 80),
]
ASFS_SCORE_KEYWORDS = {"ASFS", "ASFSSCORE", "ASFS_SCORE", "ASFSTOTAL"}

plt.rcParams['font.family']     = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size']       = 10
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.left']  = False
plt.rcParams['axes.grid']         = True
plt.rcParams['grid.alpha']        = 0.3


# ═══════════════════════════════════════════════════════════════
# 2. PARAMETER CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

def is_excluded_parameter(param_name):
    full = param_name.upper().strip()
    if not full:
        return True
    for wk in WHITELIST_KEYWORDS:
        if wk in full:
            return False
    core = full.split("_")[-1] if "_" in full else full
    if full in EXCLUDED_PARAM_NAMES or core in EXCLUDED_PARAM_NAMES:
        return True
    for excl in EXCLUDED_PARAM_NAMES:
        pos = full.find(excl)
        if pos < 0:
            continue
        before_ok = (pos == 0) or full[pos - 1] in ("_", " ")
        end_pos   = pos + len(excl)
        after_ok  = (end_pos >= len(full)) or full[end_pos] in ("_", " ") or full[end_pos].isdigit()
        if (before_ok and after_ok) or len(excl) > 5:
            return True
    for sfx in METADATA_SUFFIXES:
        if len(core) > len(sfx) and core.endswith(sfx):
            return True
    return False


def is_expert_or_tolerance_param(param_name: str) -> bool:
    upper = param_name.upper().strip()
    for kw in EXPERT_GRADING_KEYWORDS + TOLERANCE_KEYWORDS:
        if kw in upper:
            return True
    return False


def is_expert_grading_param(p: str) -> bool:
    u = p.upper().strip()
    return any(kw in u for kw in EXPERT_GRADING_KEYWORDS)

def is_tolerance_grading_param(p: str) -> bool:
    u = p.upper().strip()
    return any(kw in u for kw in TOLERANCE_KEYWORDS)

def is_comedogenicity_param(p: str) -> bool:
    u = p.upper().strip()
    return any(kw in u for kw in COMEDOGENICITY_KEYWORDS)

def is_instrument_param(p: str) -> bool:
    u = p.upper().strip()
    if len(u) <= 4 and u.isalpha() and not is_expert_grading_param(u) \
            and not is_tolerance_grading_param(u):
        return True
    return any(kw in u for kw in INSTRUMENT_KEYWORDS)

def is_ph_param(p: str) -> bool:
    return p.upper().strip() == "PH"

def is_schirmer_param(p: str) -> bool:
    u = p.upper().strip()
    return "MODSCH" in u or "SCHIRMER" in u

def is_asfs_score_param(p: str) -> bool:
    u = re.sub(r'[^A-Z0-9]', '', p.upper().strip())
    return u in ASFS_SCORE_KEYWORDS


def classify_parameter(param_name: str) -> str:
    if is_ph_param(param_name):       return "pH Level Strips"
    if is_schirmer_param(param_name): return "Modified Schirmer Test"
    if is_comedogenicity_param(param_name) and not is_expert_grading_param(param_name) \
            and not is_tolerance_grading_param(param_name):
        return "Comedogenicity Grading"
    if is_tolerance_grading_param(param_name) and not is_expert_grading_param(param_name):
        return "Tolerance Grading"
    if is_expert_grading_param(param_name): return "Expert Grading"
    if is_instrument_param(param_name):     return "Instrument"
    return "Unknown"


def filter_parameters_by_mode(param_names: list, mode: str) -> list:
    if mode == "All Parameters":
        return param_names
    result = []
    for p in param_names:
        is_tol   = is_tolerance_grading_param(p)
        is_inst  = is_instrument_param(p)
        is_exp   = is_expert_grading_param(p)
        is_comed = is_comedogenicity_param(p)
        is_ph    = is_ph_param(p)
        is_sch   = is_schirmer_param(p)
        if mode == "pH Level Strips":
            if is_ph: result.append(p)
        elif mode == "Modified Schirmer Test":
            if is_sch: result.append(p)
        elif mode == "Comedogenicity Grading":
            if not is_exp and not is_tol and not is_inst: result.append(p)
        elif mode == "Tolerance Grading":
            if not is_exp and not is_inst and not is_comed: result.append(p)
        elif mode == "Expert Grading":
            if not is_tol and not is_inst and not is_comed: result.append(p)
        elif mode == "Instrument":
            if not is_exp and not is_tol and not is_comed and not is_ph and not is_sch:
                result.append(p)
    return result


def collapse_blank_lines(txt: str) -> str:
    result = txt
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    return result.strip()


# ═══════════════════════════════════════════════════════════════
# 3. TIMEPOINT PARSING
# ═══════════════════════════════════════════════════════════════

def parse_header_info(header_text):
    if "_" not in header_text:
        return {'timepoint': '', 'parameter': header_text, 'raw': header_text}
    parts = header_text.split("_")
    composite = {"BL","BASELINE","BASE","IMM","IMMEDIATE","IM","PRE","POST","SCR","SCREENING"}
    if len(parts) >= 3 and parts[1].upper() in composite:
        tp    = f"{parts[0]}_{parts[1]}"
        param = "_".join(parts[2:])
    else:
        tp    = parts[0]
        param = "_".join(parts[1:])
    return {'timepoint': tp.upper(), 'parameter': param.upper(), 'raw': header_text}


def tp_sort_key(tp):
    u = tp.upper()
    if u in KNOWN_TP_ORDER:
        return (KNOWN_TP_ORDER.index(u), 0)
    m = re.match(r'^([A-Z]+)(\d+)$', u)
    if m:
        ranks = {"MIN": 4, "H": 5, "D": 6, "W": 7, "M": 8}
        return (ranks.get(m.group(1), 99), int(m.group(2)))
    return (999, 0)


# ═══════════════════════════════════════════════════════════════
# 4. DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

class ParameterInfo:
    def __init__(self, base_name, display_name):
        self.base_name    = base_name
        self.display_name = display_name
        self.tp_columns   = OrderedDict()

    def __repr__(self):
        return f"Param({self.base_name}: {self.display_name}, {len(self.tp_columns)} tps)"


class ECRFData:
    def __init__(self):
        self.study_ref         = ""
        self.parameters        = OrderedDict()
        self.orphaned_params   = OrderedDict()
        self.timepoint_order   = []
        self.unmapped_tps      = []
        self.baseline_prefix   = "BL"
        self.df                = None
        self.included_subjects = []
        self.excluded_subjects = []
        self.all_subjects_info = []
        self.n_included        = 0
        self.col_map: dict     = {}
        self.orphan_col_map: dict = {}


# ═══════════════════════════════════════════════════════════════
# 5. COLUMN DATA CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

def classify_column_data(series, max_samples=50):
    cnt      = {'numeric': 0, 'yesno': 0, 'time': 0, 'text': 0, 'empty': 0, 'total': 0}
    time_pat = re.compile(r'^\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?$', re.IGNORECASE)
    for val in series:
        s = str(val).strip().upper() if pd.notna(val) else ""
        if s in MISSING_MARKERS or s == "NAN" or pd.isna(val):
            cnt['empty'] += 1
            continue
        cnt['total'] += 1
        if s in ("YES", "NO"):
            cnt['yesno'] += 1
        elif time_pat.match(s):
            cnt['time'] += 1
        else:
            try:
                float(val)
                cnt['numeric'] += 1
            except:
                cnt['text'] += 1
        if cnt['total'] >= max_samples:
            break
    total = cnt['total']
    if total == 0:
        return 'empty'
    thresh = 0.6
    pcts   = {k: cnt[k] / total for k in ('numeric', 'yesno', 'time', 'text')}
    if pcts['numeric'] >= thresh: return 'numeric'
    if pcts['yesno']   >= thresh: return 'yesno'
    if pcts['time']    >= thresh: return 'time'
    if pcts['text']    >= thresh: return 'text'
    return 'mixed'

def strip_trailing_digits(name: str) -> str:
    i = len(name) - 1
    while i >= 0 and name[i].isdigit():
        i -= 1
    return name[: i + 1] if i >= 0 else name

def is_likely_data_parameter(series, param_code=""):
    return classify_column_data(series) in ('numeric', 'mixed')


# ═══════════════════════════════════════════════════════════════
# 6. OPTION VALUES & SHEET DETECTION
# ═══════════════════════════════════════════════════════════════

def find_option_values_sheet(excel_file, ecrf_sheet_name):
    xls  = pd.ExcelFile(excel_file)
    cand = ecrf_sheet_name + "1"
    if cand in xls.sheet_names:
        return cand
    base_lower = ecrf_sheet_name.lower()
    for sn in xls.sheet_names:
        if sn.lower() == base_lower:
            continue
        if sn.lower().startswith(base_lower[:min(len(base_lower), 10)]):
            try:
                peek = pd.read_excel(excel_file, sheet_name=sn, nrows=2, header=None)
                hdrs = [str(v).strip().upper() for v in peek.iloc[0] if pd.notna(v)]
                if hdrs[:3] == ["VARIABLE NAME", "OPTION NAME", "OPTION VALUE"]:
                    return sn
            except Exception:
                pass
    for sn in xls.sheet_names:
        if "option" in sn.lower() and "value" in sn.lower():
            return sn
    for sn in xls.sheet_names:
        if sn == ecrf_sheet_name:
            continue
        try:
            peek = pd.read_excel(excel_file, sheet_name=sn, nrows=2, header=None)
            hdrs = [str(v).strip().upper() for v in peek.iloc[0] if pd.notna(v)]
            if hdrs[:3] == ["VARIABLE NAME", "OPTION NAME", "OPTION VALUE"]:
                return sn
        except Exception:
            pass
    return None


def load_ov_variable_basenames(excel_file, ov_sheet_name):
    df = pd.read_excel(excel_file, sheet_name=ov_sheet_name, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    basenames  = set()
    for var in df.iloc[:, 0].dropna().astype(str):
        var = var.strip()
        if not var:
            continue
        m = re.match(r'^(?:BL|SC|W\d+)_(.+)$', var, re.IGNORECASE)
        basenames.add(m.group(1).upper() if m else var.upper())
    return basenames


def find_ecrf_sheets(excel_file):
    xls   = pd.ExcelFile(excel_file)
    cands = [sn for sn in xls.sheet_names
             if "ecrf" in sn.lower() or "crf" in sn.lower()]
    return cands if cands else xls.sheet_names


# ═══════════════════════════════════════════════════════════════
# 7. SUBJECT ID NORMALIZATION
# ═══════════════════════════════════════════════════════════════

def normalize_subject_id(raw: str) -> str:
    raw = raw.strip()
    if "-" in raw:
        parts = raw.split("-")
        tail  = parts[-1].strip()
        if tail.isdigit():
            raw = tail
    m = re.match(r'^[A-Za-z]+(\d+)$', raw)
    if m:
        raw = m.group(1)
    try:
        return f"{int(raw):04d}"
    except ValueError:
        return raw


# ═══════════════════════════════════════════════════════════════
# 8. eCRF DATA PARSING
# ═══════════════════════════════════════════════════════════════

def parse_ecrf_data(excel_file, ecrf_sheet, ov_exclusion_basenames,
                    global_exclusions=None, center_filter=None):
    global CENTER_FILTER
    if center_filter is not None:
        CENTER_FILTER = center_filter

    ecrf    = ECRFData()
    df_raw  = pd.read_excel(excel_file, sheet_name=ecrf_sheet, header=None)
    headers = [str(v).strip() for v in df_raw.iloc[0]]
    q_texts = [str(v).strip() if pd.notna(v) else "" for v in df_raw.iloc[1]]

    df_data = df_raw.iloc[2:].copy()
    df_data.columns = headers
    df_data = df_data.reset_index(drop=True)

    c_subj = next((h for h in headers if h.upper() == COL_SUBJECT), None)
    c_stat = next((h for h in headers if h.upper() == COL_STATUS),  None)
    c_cent = next((h for h in headers if h.upper() == COL_CENTER),  None)
    c_ref  = next((h for h in headers if h.upper() == COL_STUDY),   None)

    if not c_subj or not c_stat:
        return None, "eCRF sheet is missing SUBJECT ID or STATUS column."

    if c_ref:
        refs           = df_data[c_ref].dropna().astype(str).unique()
        ecrf.study_ref = next((r for r in refs if r.strip()), "Study")

    def fmt_sid(v):
        try:    return normalize_subject_id(str(v))
        except: return str(v).strip()

    df_data['_SID']          = df_data[c_subj].apply(fmt_sid)
    df_data['_STATUS']       = df_data[c_stat].astype(str).str.strip()
    df_data['_STATUS_UPPER'] = df_data['_STATUS'].str.upper()

    if global_exclusions:
        df_data = df_data[~df_data['_SID'].isin(set(global_exclusions))].copy()

    for _, row in df_data.iterrows():
        ecrf.all_subjects_info.append({
            'sid': row['_SID'], 'status': row['_STATUS'],
            'status_upper': row['_STATUS_UPPER'],
        })

    excluded_mask          = df_data['_STATUS_UPPER'].str.contains('EXCLUDED', na=False)
    ecrf.excluded_subjects = df_data[excluded_mask]['_SID'].tolist()
    df_included            = df_data[~excluded_mask].copy()
    ecrf.included_subjects = df_included['_SID'].tolist()
    ecrf.n_included        = len(ecrf.included_subjects)
    ecrf.df                = df_included

    tp_set          = set()
    param_collector = defaultdict(lambda: {'display': '', 'tp_cols': OrderedDict()})
    orphan_collector= defaultdict(lambda: {'display': '', 'col': ''})
    metadata_upper  = {m.upper() for m in METADATA_COLS}
    col_map         = {}
    orphan_col_map  = {}
    q_texts_map     = {}
    tp_appearance   = {}

    for i, var in enumerate(headers):
        if not var or var.upper() in metadata_upper \
                or var.startswith('.') or var.startswith('_'):
            continue
        info      = parse_header_info(var)
        tp_prefix = info['timepoint']
        base_name = info['parameter']

        if base_name in ov_exclusion_basenames:
            continue

        col_data = df_included[var] if var in df_included.columns \
                   else pd.Series(dtype=object)

        if not tp_prefix and base_name:
            if not is_excluded_parameter(base_name):
                if len(col_data) == 0 or is_likely_data_parameter(col_data, base_name):
                    orphan_collector[base_name]['col'] = var
                    if not orphan_collector[base_name]['display'] \
                            and i < len(q_texts) and q_texts[i]:
                        orphan_collector[base_name]['display'] = q_texts[i]
                    orphan_col_map[base_name] = var
            continue

        if not tp_prefix or not base_name:
            continue
        if is_excluded_parameter(base_name):
            continue
        if len(col_data) > 0 and not is_likely_data_parameter(col_data, base_name):
            continue

        tp_set.add(tp_prefix)
        if tp_prefix not in tp_appearance:
            tp_appearance[tp_prefix] = i
        entry = param_collector[base_name]
        entry['tp_cols'][tp_prefix] = var
        if not entry['display'] and i < len(q_texts) and q_texts[i]:
            entry['display'] = q_texts[i]
            q_texts_map[base_name] = q_texts[i]
        col_map[(tp_prefix, base_name)] = var

    all_tps_sorted       = sort_timepoints_by_appearance(list(tp_set), tp_appearance)
    known_upper          = {t.upper() for t in KNOWN_TP_ORDER}
    ecrf.unmapped_tps    = [t for t in all_tps_sorted if t.upper() not in known_upper]
    ecrf.timepoint_order = all_tps_sorted
    ecrf.col_map         = col_map
    ecrf.orphan_col_map  = orphan_col_map

    for base_name, info in sorted(param_collector.items()):
        display = info['display'] or base_name
        pi      = ParameterInfo(base_name, display)
        for tp in all_tps_sorted:
            if tp in info['tp_cols']:
                pi.tp_columns[tp] = info['tp_cols'][tp]
        ecrf.parameters[base_name] = pi

    ecrf.parameters = group_duplicate_parameters(
        ecrf.parameters, df_included, q_texts_map)

    for base_name, info in sorted(orphan_collector.items()):
        display = info['display'] or base_name
        pi      = ParameterInfo(base_name, display)
        ecrf.orphaned_params[base_name] = pi

    if "BL" in all_tps_sorted:
        ecrf.baseline_prefix = "BL"
    elif all_tps_sorted:
        ecrf.baseline_prefix = all_tps_sorted[0]

    return ecrf, None


# ═══════════════════════════════════════════════════════════════
# 9. HELPERS
# ═══════════════════════════════════════════════════════════════

def group_duplicate_parameters(parameters: OrderedDict,
                                df_included: pd.DataFrame,
                                q_texts_map: dict) -> OrderedDict:
    desc_to_first: dict = {}
    merged: OrderedDict = OrderedDict()
    for base, pi in parameters.items():
        desc = (q_texts_map.get(base) or pi.display_name or base).strip().upper()
        if desc and desc in desc_to_first:
            canonical = desc_to_first[desc]
            for tp, col in pi.tp_columns.items():
                if tp not in merged[canonical].tp_columns:
                    merged[canonical].tp_columns[tp] = col
        else:
            desc_to_first[desc] = base
            merged[base]        = pi
    return merged


def sort_timepoints_by_appearance(timepoint_order: list,
                                   col_appearance: dict) -> list:
    def sort_key(tp):
        if tp in col_appearance:
            return (0, col_appearance[tp])
        return (1, tp_sort_key(tp)[0] * 1000 + tp_sort_key(tp)[1])
    return sorted(timepoint_order, key=sort_key)

def find_orphan_conflicts(
    ecrf: "ECRFData",
    assignments: dict,          # {orphan_base: tp_key}
    df_included: "pd.DataFrame",
    similarity_threshold: float = 0.75,
) -> list:
    """
    For each orphan assignment, check whether:
      (a) the stripped base name matches an existing parameter, AND
      (b) the assigned timepoint is already occupied by that parameter, AND
      (c) the two columns' Row-2 descriptions are similar enough.

    Returns a list of dicts, one per conflict:
      {
        'orphan_base':    str,   # original orphan key e.g. "FIRM2"
        'stripped_base':  str,   # e.g. "FIRM"
        'existing_base':  str,   # matched existing parameter key
        'tp':             str,   # timepoint key e.g. "W4"
        'orphan_col':     str,   # column name in df
        'existing_col':   str,   # column name in df
        'orphan_display': str,
        'existing_display': str,
        'orphan_mean':    float | None,
        'existing_mean':  float | None,
        'orphan_n':       int,
        'existing_n':     int,
        'sample_df':      pd.DataFrame,  # 5-row side-by-side sample
        'similarity':     float,
      }
    """
    from difflib import SequenceMatcher

    conflicts = []

    for orphan_base, tp in assignments.items():
        if tp == "" or orphan_base not in ecrf.orphaned_params:
            continue

        orphan_col = ecrf.orphan_col_map.get(orphan_base)
        if orphan_col is None or orphan_col not in df_included.columns:
            continue

        stripped = strip_trailing_digits(orphan_base)

        # Find a matching existing parameter by stripped name or description
        matched_existing = None
        best_sim = 0.0

        orphan_display = ecrf.orphaned_params[orphan_base].display_name or orphan_base

        for existing_base, existing_param in ecrf.parameters.items():
            # Must occupy the same timepoint
            if tp not in existing_param.tp_columns:
                continue

            # Check name similarity (stripped base vs existing base)
            name_sim = SequenceMatcher(
                None,
                stripped.upper(),
                existing_base.upper(),
            ).ratio()

            # Check description similarity
            existing_display = existing_param.display_name or existing_base
            desc_sim = SequenceMatcher(
                None,
                orphan_display.upper(),
                existing_display.upper(),
            ).ratio()

            sim = max(name_sim, desc_sim)

            if sim >= similarity_threshold and sim > best_sim:
                best_sim = sim
                matched_existing = existing_base

        if matched_existing is None:
            continue

        existing_param = ecrf.parameters[matched_existing]
        existing_col   = existing_param.tp_columns[tp]

        if existing_col not in df_included.columns:
            continue

        # Compute per-column stats
        orphan_vals   = pd.to_numeric(df_included[orphan_col],   errors="coerce").dropna()
        existing_vals = pd.to_numeric(df_included[existing_col], errors="coerce").dropna()

        # Build 5-row side-by-side sample aligned on subject ID
        sample = (
            df_included[["_SID", existing_col, orphan_col]]
            .rename(columns={
                existing_col: f"{matched_existing} (existing)",
                orphan_col:   f"{orphan_base} (orphan)",
            })
            .head(5)
            .reset_index(drop=True)
        )
        sample.columns = ["Subject ID",
                          f"{matched_existing} (existing)",
                          f"{orphan_base} (orphan)"]

        conflicts.append({
            "orphan_base":      orphan_base,
            "stripped_base":    stripped,
            "existing_base":    matched_existing,
            "tp":               tp,
            "orphan_col":       orphan_col,
            "existing_col":     existing_col,
            "orphan_display":   orphan_display,
            "existing_display": existing_param.display_name or matched_existing,
            "orphan_mean":      float(orphan_vals.mean())   if len(orphan_vals)   > 0 else None,
            "existing_mean":    float(existing_vals.mean()) if len(existing_vals) > 0 else None,
            "orphan_n":         len(orphan_vals),
            "existing_n":       len(existing_vals),
            "sample_df":        sample,
            "similarity":       round(best_sim, 2),
        })

    return conflicts

# ═══════════════════════════════════════════════════════════════
# 10. ORPHAN ASSIGNMENT
# ═══════════════════════════════════════════════════════════════

def apply_orphan_assignments(
    ecrf: "ECRFData",
    assignments: dict,
    merge_decisions: dict = None,
) -> "ECRFData":
    """
    Applies orphan timepoint assignments to ecrf.parameters.

    merge_decisions keys:
      orphan_base          -> True (merge) or False (keep separate)
      __target_orphan_base -> the exact existing parameter key to merge into,
                             stashed by the UI from find_orphan_conflicts results
    """
    if merge_decisions is None:
        merge_decisions = {}

    for orphan_base, tp in assignments.items():
        if tp == "" or orphan_base not in ecrf.orphaned_params:
            continue

        orphan_col = ecrf.orphan_col_map.get(orphan_base)
        if orphan_col is None:
            continue

        # Add timepoint to order if not already present
        if tp not in ecrf.timepoint_order:
            ecrf.timepoint_order = sorted(
                ecrf.timepoint_order + [tp], key=tp_sort_key
            )

        stripped     = strip_trailing_digits(orphan_base)
        should_merge = merge_decisions.get(orphan_base, False)

        target_base = None
        if should_merge:
            # First choice: use the exact key stashed by the UI during conflict review.
            # This survives group_duplicate_parameters renaming the canonical key.
            stashed = merge_decisions.get(f"__target_{orphan_base}")
            if stashed and stashed in ecrf.parameters and tp in ecrf.parameters[stashed].tp_columns:
                target_base = stashed
            else:
                # Fallback: scan for a parameter that owns this tp and whose key
                # matches either the stripped base or the original orphan base.
                for existing_base, existing_param in ecrf.parameters.items():
                    if tp in existing_param.tp_columns and (
                        existing_base.upper() == stripped.upper()
                        or existing_base.upper() == orphan_base.upper()
                    ):
                        target_base = existing_base
                        break

        if target_base and should_merge:
            # Merge: point the existing parameter's tp slot at the orphan column.
            # Explicit user decision so overwrite is intentional.
            ecrf.parameters[target_base].tp_columns[tp] = orphan_col
            ecrf.col_map[(tp, target_base)] = orphan_col
        else:
            # Keep separate: register as its own parameter entry.
            if orphan_base not in ecrf.parameters:
                ecrf.parameters[orphan_base] = ecrf.orphaned_params[orphan_base]
            ecrf.parameters[orphan_base].tp_columns[tp] = orphan_col
            ecrf.col_map[(tp, orphan_base)] = orphan_col

    return ecrf

# ═══════════════════════════════════════════════════════════════
# 11. STATISTICS
# ═══════════════════════════════════════════════════════════════

def compute_parameter_stats(ecrf, param, min_pairs=30):
    df        = ecrf.df
    bl_prefix = ecrf.baseline_prefix
    result    = OrderedDict()

    bl_col  = param.tp_columns.get(bl_prefix)
    bl_mean = None
    if bl_col and bl_col in df.columns:
        bl_vals = pd.to_numeric(df[bl_col], errors='coerce').dropna().values
        bl_mean = np.mean(bl_vals) if len(bl_vals) > 0 else None

    for tp in ecrf.timepoint_order:
        col = param.tp_columns.get(tp)
        if not col or col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors='coerce').dropna().values
        if len(vals) == 0:
            continue
        mean_val    = np.mean(vals)
        std_val     = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        pct_change  = None
        significant = False

        if tp != bl_prefix and bl_mean is not None and bl_mean != 0:
            pct_change = ((mean_val - bl_mean) / bl_mean) * 100

        if tp != bl_prefix and bl_col and bl_col in df.columns:
            bl_s = pd.to_numeric(df[bl_col], errors='coerce')
            tp_s = pd.to_numeric(df[col],    errors='coerce')
            mask = bl_s.notna() & tp_s.notna()
            if mask.sum() >= min_pairs:
                _, p_val    = stats.ttest_rel(bl_s[mask].values, tp_s[mask].values)
                significant = p_val < 0.05

        result[tp] = {'mean': mean_val, 'std': std_val, 'n': len(vals),
                      'pct_change': pct_change, 'values': vals,
                      'significant': significant}
    return result


def auto_detect_improvement_direction(ecrf, param):
    s  = compute_parameter_stats(ecrf, param)
    bl = ecrf.baseline_prefix
    if bl not in s or len(s) < 2:
        return "lower"
    return "lower" if s[list(s.keys())[-1]]['mean'] <= s[bl]['mean'] else "higher"


def round_half_up(value, decimals=2):
    d = Decimal(str(value))
    return float(d.quantize(Decimal(10) ** -decimals, rounding=ROUND_HALF_UP))

def fmt_value(value, decimals=2):
    return f"{round_half_up(value, decimals):.{decimals}f}"


# ═══════════════════════════════════════════════════════════════
# 12. STATISTICAL SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════

def fmt_pvalue(p_val, significant: bool) -> str:
    if p_val is None:
        return "—"
    if p_val < 0.001:
        return "<0.001*"
    return f"{p_val:.3f}" + ("*" if significant else "")


def fmt_pct_change(pct) -> str:
    if pct is None:
        return "N/A"
    return f"{round_half_up(pct, 2):.2f}%"


def compute_freq_distribution(values: np.ndarray, grades=(0, 1, 2, 3, 4)) -> dict:
    n_total = len(values)
    result  = {}
    for g in grades:
        count = int(np.sum(values == g))
        pct   = (count / n_total * 100) if n_total > 0 else 0.0
        result[g] = (count, pct)
    return result


def compute_pct_improved(df, bl_col: str, tp_col: str,
                          improvement_dir: str) -> float | None:
    bl   = pd.to_numeric(df[bl_col], errors='coerce')
    tp   = pd.to_numeric(df[tp_col], errors='coerce')
    mask = bl.notna() & tp.notna()
    if mask.sum() == 0:
        return None
    bl_v, tp_v = bl[mask].values, tp[mask].values
    improved = int(np.sum(tp_v < bl_v)) if improvement_dir == "lower" \
               else int(np.sum(tp_v > bl_v))
    return (improved / mask.sum()) * 100


def _compute_raw_pvalue(df, bl_col, tp_col) -> float | None:
    if bl_col is None or tp_col is None:
        return None
    if bl_col not in df.columns or tp_col not in df.columns:
        return None
    bl_s  = pd.to_numeric(df[bl_col], errors='coerce')
    tp_s  = pd.to_numeric(df[tp_col], errors='coerce')
    mask  = bl_s.notna() & tp_s.notna()
    if mask.sum() < 2:
        return None
    bl_v, tp_v = bl_s[mask].values, tp_s[mask].values
    if np.std(bl_v - tp_v) == 0:
        return 1.0
    try:
        _, p = stats.ttest_rel(bl_v, tp_v)
        return float(p)
    except Exception:
        return None


def build_stats_table(ecrf: ECRFData,
                      all_param_stats: dict,
                      improvement_dirs: dict,
                      keep: list,
                      active_tps: list,
                      analysis_mode: str) -> pd.DataFrame:
    is_expert    = "Expert"    in analysis_mode
    is_tolerance = "Tolerance" in analysis_mode
    bl_prefix    = ecrf.baseline_prefix
    df_data      = ecrf.df
    rows         = []

    for base in keep:
        if base not in ecrf.parameters:
            continue
        param   = ecrf.parameters[base]
        s_dict  = all_param_stats.get(base, {})
        bl_col  = param.tp_columns.get(bl_prefix)
        imp_dir = improvement_dirs.get(base, "lower")
        bl_mean = s_dict.get(bl_prefix, {}).get('mean')

        for tp in active_tps:
            if tp not in s_dict:
                continue
            s        = s_dict[tp]
            tp_col   = param.tp_columns.get(tp)
            is_bl    = (tp == bl_prefix)
            tp_label = TP_DISPLAY.get(tp, tp)

            mean_val = s['mean']
            std_val  = s['std']
            n        = s['n']

            if is_bl:
                pval_str = ""
            else:
                raw_pval = _compute_raw_pvalue(df_data, bl_col, tp_col)
                pval_str = "—" if raw_pval is None \
                           else fmt_pvalue(raw_pval, raw_pval < 0.05)

            if is_bl:
                pct_str = ""
            elif bl_mean is None or bl_mean == 0:
                pct_str = "N/A"
            else:
                pct_str = fmt_pct_change(((mean_val - bl_mean) / bl_mean) * 100)

            row = {
                "Assessment":                  param.display_name,
                "Time Point":                  tp_label,
                "n":                           n,
                "Mean ± SD":                   f"{fmt_value(mean_val)} ± {fmt_value(std_val)}",
                "p-value":                     pval_str,
                "Mean % Change From Baseline": pct_str,
            }

            if is_expert:
                if is_bl or bl_col is None or tp_col is None:
                    row["% Subjects Improved"] = ""
                else:
                    pct_imp = compute_pct_improved(df_data, bl_col, tp_col, imp_dir)
                    row["% Subjects Improved"] = (
                        f"{round_half_up(pct_imp, 2):.2f}%" if pct_imp is not None else "—")

            if is_tolerance and tp_col and tp_col in df_data.columns:
                tp_vals = pd.to_numeric(df_data[tp_col], errors='coerce').dropna().values
                freq    = compute_freq_distribution(tp_vals)
                for g in range(5):
                    cnt, pct_g = freq[g]
                    row[f"Grade {g} n (%)"] = f"{cnt} ({round_half_up(pct_g, 1):.1f}%)"
            elif is_tolerance:
                for g in range(5):
                    row[f"Grade {g} n (%)"] = "—"

            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# 13. ASFS THRESHOLD TABLE
# ═══════════════════════════════════════════════════════════════

def build_asfs_threshold_table(ecrf: ECRFData,
                                keep: list,
                                active_tps: list) -> pd.DataFrame:
    """
    Port of VBA ASFS threshold classification table.
    Bands: Normal/Very Slight 0-15, Mild 16-24, Moderate 25-34, Severe 35-80.
    Detects ASFS Score param by base name; returns empty DF if not found.
    """
    asfs_base = next(
        (b for b in keep if is_asfs_score_param(b) and b in ecrf.parameters),
        None
    )
    if asfs_base is None:
        return pd.DataFrame()

    param     = ecrf.parameters[asfs_base]
    df_data   = ecrf.df
    rows      = []

    band_labels = [f"{label} ({lo}-{hi})" for label, lo, hi in ASFS_BANDS]

    for tp in active_tps:
        col = param.tp_columns.get(tp)
        if col is None or col not in df_data.columns:
            continue
        vals = pd.to_numeric(df_data[col], errors='coerce').dropna().values
        n    = len(vals)
        if n == 0:
            continue

        tp_label    = TP_DISPLAY.get(tp, tp)
        band_counts = {}
        for label, lo, hi in ASFS_BANDS:
            cnt = int(np.sum((vals >= lo) & (vals <= hi)))
            pct = (cnt / n * 100) if n > 0 else 0.0
            band_counts[f"{label} ({lo}-{hi})"] = \
                f"{cnt} ({round_half_up(pct, 2):.2f}%)"

        row = {"Threshold based classification": "ASFS Score",
               "Time Point": tp_label,
               "n": n}
        row.update(band_counts)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    col_order = ["Threshold based classification", "Time Point", "n"] + band_labels
    return pd.DataFrame(rows)[col_order]


# ═══════════════════════════════════════════════════════════════
# 14. DATA QUALITY SCAN
# ═══════════════════════════════════════════════════════════════

MISSING_VALUE_MARKERS = {"", ".", ".nd", ".unk", ".hidden", "nd", "n/a", "#n/a"}


def scan_data_quality(ecrf: ECRFData, selected_params: list,
                      selected_tps: list) -> dict:
    df           = ecrf.df
    issues: dict = {}

    for tp in selected_tps:
        for base in selected_params:
            col = ecrf.col_map.get((tp, base))
            if col is None or col not in df.columns:
                continue

            disp = ecrf.parameters[base].display_name \
                   if base in ecrf.parameters else base

            for _, row in df.iterrows():
                sid   = row.get('_SID', 'UNKNOWN')
                val   = row[col]
                s_val = str(val).strip().lower() if pd.notna(val) else ""
                has_data = (pd.notna(val) and s_val not in MISSING_VALUE_MARKERS
                            and s_val != "nan")
                if has_data:
                    try:    float(val)
                    except: has_data = False
                if not has_data:
                    issues.setdefault(sid, {}).setdefault(tp, {})[f"{tp}_{base}"] = {
                        'param_display': disp,
                    }
    return issues


def summarise_data_quality(issues: dict) -> dict:
    total             = 0
    affected          = set()
    per_subject: dict = {}

    for sid, tp_dict in issues.items():
        affected.add(sid)
        for tp, params in tp_dict.items():
            for _, meta in params.items():
                per_subject.setdefault(sid, []).append(
                    f"{TP_DISPLAY.get(tp, tp)}: {meta['param_display']}")
                total += 1

    return {'total_issues': total, 'affected_subjects': len(affected),
            'per_subject': per_subject}

# ═══════════════════════════════════════════════════════════════
# 15. CHART GENERATION
# ═══════════════════════════════════════════════════════════════

def draw_solid_bar(ax, x, y, width, height, color):
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch
    r     = min(width * 0.2, height * 0.15, 0.08)
    verts = [(x, y), (x, y+height-r), (x, y+height), (x+r, y+height),
             (x+width-r, y+height), (x+width, y+height),
             (x+width, y+height-r), (x+width, y), (x, y)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), fc=color, ec="none", zorder=3))


def draw_pill_badge(ax, x_center, y_center, text, bg_color, text_color='white',
                    fontsize=10, pad_x=0.35, pad_y=0.12):
    txt  = ax.text(x_center, y_center, text, ha='center', va='center',
                   fontsize=fontsize, fontweight='bold', color=text_color, zorder=6)
    rend = ax.get_figure().canvas.get_renderer()
    bbox = ax.transData.inverted().transform(txt.get_window_extent(renderer=rend))
    w    = bbox[1][0] - bbox[0][0] + pad_x
    h    = bbox[1][1] - bbox[0][1] + pad_y
    ax.add_patch(FancyBboxPatch(
        (x_center - w/2, y_center - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={h/2}",
        ec="none", fc=bg_color, alpha=1.0, zorder=5))


def draw_pill_chip(fig, x_center, y_center, text, fontsize=9,
                   pad_x=0.012, pad_y=0.006):
    txt  = fig.text(x_center, y_center, text, ha='center', va='center',
                    fontsize=fontsize, fontweight='medium',
                    color=COLORS['text_main'], zorder=6)
    rend = fig.canvas.get_renderer()
    bbox = fig.transFigure.inverted().transform(
               txt.get_window_extent(renderer=rend))
    w    = bbox[1][0] - bbox[0][0] + pad_x * 2
    h    = bbox[1][1] - bbox[0][1] + pad_y * 2
    fig.patches.append(FancyBboxPatch(
        (x_center - w/2, y_center - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={h/2}",
        ec="none", fc=COLORS['pill_bg'], alpha=1.0, zorder=5,
        transform=fig.transFigure, clip_on=False))


def create_parameter_chart(ecrf, param, param_stats, improvement_dir,
                            active_tps, custom_title=None, show_center=True):
    if not param_stats:
        return None
    bl_prefix   = ecrf.baseline_prefix
    ordered_tps = [tp for tp in active_tps if tp in param_stats]
    if not ordered_tps:
        return None

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor('white')

    x_pos  = np.arange(len(ordered_tps))
    means  = [param_stats[tp]['mean'] for tp in ordered_tps]
    ns     = [param_stats[tp]['n']    for tp in ordered_tps]
    labels = [TP_DISPLAY.get(tp, tp)  for tp in ordered_tps]

    bar_colors = []
    for tp in ordered_tps:
        if tp == bl_prefix:
            bar_colors.append(COLORS['baseline'])
        else:
            pct = param_stats[tp].get('pct_change')
            if pct is None:
                bar_colors.append(COLORS['no_change'])
            elif improvement_dir == "lower":
                bar_colors.append(COLORS['improved'] if pct < -0.5 else
                                  COLORS['worsened'] if pct > 0.5 else COLORS['no_change'])
            else:
                bar_colors.append(COLORS['improved'] if pct > 0.5 else
                                  COLORS['worsened'] if pct < -0.5 else COLORS['no_change'])

    y_max = max(means) * 1.25 if means else 10
    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.6, len(ordered_tps) - 0.4)

    bar_width = 0.55
    for i, tp in enumerate(ordered_tps):
        draw_solid_bar(ax, x_pos[i] - bar_width/2, 0, bar_width, means[i], bar_colors[i])

    for i, tp in enumerate(ordered_tps):
        lbl = fmt_value(means[i])
        if tp != bl_prefix and param_stats[tp].get('significant', False):
            lbl += "*"
        ax.text(x_pos[i], means[i]/2, lbl, ha='center', va='center',
                fontsize=12, color='white', fontweight='bold', zorder=6)

    fig.canvas.draw()
    for i, tp in enumerate(ordered_tps):
        if tp == bl_prefix:
            continue
        pct = param_stats[tp].get('pct_change')
        if pct is not None:
            dpct  = -pct if improvement_dir == "lower" else pct
            sign  = "+" if dpct > 0 else ""
            label = f"{sign}{fmt_value(dpct)}%"
            draw_pill_badge(ax, x_pos[i], means[i] + y_max * 0.045, label,
                            bg_color=bar_colors[i], text_color='white',
                            fontsize=10, pad_x=0.3, pad_y=0.08)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean", fontsize=13, fontweight='bold')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#E4E4E4', linewidth=1)
    ax.xaxis.grid(False)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax.tick_params(axis='y', which='major', length=6, width=1.2,
                   color=COLORS['text_main'])
    ax.tick_params(axis='y', which='minor', length=3, width=0.6, color='#999999')
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('#BBBBBB')
    ax.spines['left'].set_linewidth(1.2)

    if len(set(ns)) > 1:
        for i in range(len(ordered_tps)):
            ax.text(x_pos[i], -y_max * 0.04, f"n={ns[i]}", ha='center', va='top',
                    fontsize=8, color=COLORS['text_sub'])

    ax.set_xlabel("Timepoint", fontsize=13, fontweight='bold')

    title_main = custom_title or f"{ecrf.study_ref} — {param.display_name}"
    fig.text(0.5, 0.97, title_main, ha='center', va='top',
             fontsize=17, fontweight='bold', color=COLORS['text_main'])
    fig.text(0.5, 0.92, "Mean by Timepoint", ha='center', va='top',
             fontsize=13, fontweight='medium', color=COLORS['text_sub'])

    # Subtitle chip row — direction chip is per-chart
    dir_label = ("Decrease = Improvement" if improvement_dir == "lower"
                 else "Increase = Improvement")
    if show_center and CENTER_FILTER:
        chip_texts = [f"Center: {CENTER_FILTER}",
                      f"n={ecrf.n_included} Included Subjects",
                      dir_label]
    else:
        chip_texts = [f"n={ecrf.n_included} Included Subjects", dir_label]

    chip_gap     = 0.22
    chip_start_x = 0.5 - (len(chip_texts) - 1) * chip_gap / 2
    fig.canvas.draw()
    for ci, ct in enumerate(chip_texts):
        draw_pill_chip(fig, chip_start_x + ci * chip_gap, 0.855, ct, fontsize=9.5)

    fig.text(0.5, 0.82,
             "Values above bars = % improvement from baseline · "
             "* indicates statistical significance (p < 0.05, paired t-test)",
             ha='center', va='top', fontsize=9.5,
             color=COLORS['text_sub'], style='italic')

    x_start = 0.125
    for color, label in [(COLORS['baseline'], "Baseline"),
                          (COLORS['improved'], "Improved from Baseline"),
                          (COLORS['worsened'], "Worsened from Baseline")]:
        fig.patches.append(Rectangle((x_start, 0.03), 0.015, 0.015,
            transform=fig.transFigure, color=color, zorder=2, clip_on=False))
        fig.text(x_start + 0.02, 0.033, label, fontsize=9, color=COLORS['text_main'])
        x_start += 0.02 + len(label) * 0.006 + 0.03

    plt.tight_layout(rect=[0, 0.06, 1, 0.78])
    return fig

# ═══════════════════════════════════════════════════════════════
# 16. PDF GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_pdf_bytes(ecrf, all_param_stats, improvement_dirs, chart_titles,
                       active_tps, show_center=True, progress_bar=None):
    buf    = io.BytesIO()
    params = [p for p in ecrf.parameters.values()
              if p.base_name in all_param_stats and all_param_stats[p.base_name]]
    total  = len(params)

    with PdfPages(buf) as pdf:
        for i, param in enumerate(params):
            s     = all_param_stats[param.base_name]
            imp   = improvement_dirs.get(param.base_name, "lower")
            title = chart_titles.get(param.base_name)
            fig   = create_parameter_chart(
                ecrf, param, s, imp,
                active_tps=active_tps,
                custom_title=title,
                show_center=show_center,
            )
            if fig:
                pdf.savefig(fig)
                plt.close(fig)
            if progress_bar is not None:
                progress_bar.progress((i + 1) / total,
                                      text=f"Rendering {param.display_name}…")

    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════
# MANUAL ENTRY HELPERS
# ═══════════════════════════════════════════════════════════════

def _safe_float(s) -> float | None:
    try:
        return float(s)
    except Exception:
        return None


def _normalise_tp(raw: str) -> str:
    """Convert display labels (e.g. 'Week 4') back to canonical keys (e.g. 'W4')."""
    raw   = raw.strip()
    upper = raw.upper()
    if upper in KNOWN_TP_ORDER:
        return upper
    for k, v in TP_DISPLAY.items():
        if v.upper() == upper:
            return k
    return upper


def _parse_tp_list(raw: str) -> list[str]:
    """Split, normalise, and deduplicate a comma-separated timepoint string."""
    return list(dict.fromkeys(
        _normalise_tp(t) for t in raw.split(",") if t.strip()
    ))


def _split_mean_sd(cell: str) -> tuple[float | None, float | None]:
    cell = cell.strip()
    if not cell or cell.upper() in ("", "N/A", "—", "ND"):
        return None, None
    for sep in ("±", "+-", "+/-", "±"):
        if sep in cell:
            parts = cell.split(sep, 1)
            mean  = _safe_float(parts[0].strip())
            sd_s  = parts[1].strip()
            sd    = None if sd_s.upper() in ("N/A", "—", "ND", "") \
                    else _safe_float(sd_s)
            return mean, sd
    return _safe_float(cell), None


def _parse_pct(s: str) -> float | None:
    if not s:
        return None
    return _safe_float(s.strip().rstrip("%"))


def _manual_df_to_ecrf_and_stats(
    df: pd.DataFrame,
    study_ref: str,
    baseline_tp: str,
) -> tuple["ECRFData", dict, dict]:
    ecrf = ECRFData()
    ecrf.study_ref       = study_ref
    ecrf.baseline_prefix = baseline_tp
    ecrf.df              = pd.DataFrame()
    ecrf.n_included      = int(df["n"].max()) if df["n"].notna().any() else 0

    tp_order = []
    seen: set = set()
    for tp in df["timepoint"]:
        if tp not in seen:
            tp_order.append(tp)
            seen.add(tp)
    ecrf.timepoint_order = sorted(tp_order, key=tp_sort_key)

    all_param_stats: dict = {}
    auto_dirs: dict       = {}

    for param_name, grp in df.groupby("parameter", sort=False):
        pi        = ParameterInfo(param_name, param_name)
        stat_dict = OrderedDict()
        bl_mean   = None

        for _, row in grp.iterrows():
            tp       = row["timepoint"]
            mean_val = row["mean"]
            sd_val   = 0.0
            n_val    = int(row["n"]) if pd.notna(row["n"]) else 0
            pct      = row["pct_change"] if pd.notna(row["pct_change"]) else None

            if tp == baseline_tp:
                bl_mean = mean_val
                pct     = None

            p_raw = str(row.get("p_value", "")).strip()
            sig   = "*" in p_raw or (
                p_raw not in ("", "—", "N/A")
                and _safe_float(p_raw) is not None
                and _safe_float(p_raw) < 0.05
            )

            stat_dict[tp] = {
                "mean":        mean_val,
                "std":         sd_val,
                "n":           n_val,
                "pct_change":  pct,
                "values":      np.array([mean_val]),
                "significant": sig,
            }
            pi.tp_columns[tp] = f"__manual__{tp}"

        if bl_mean is not None and bl_mean != 0:
            for tp, s in stat_dict.items():
                if tp != baseline_tp and s["pct_change"] is None:
                    s["pct_change"] = ((s["mean"] - bl_mean) / bl_mean) * 100

        non_bl = [(tp, s) for tp, s in stat_dict.items() if tp != baseline_tp]
        if non_bl and bl_mean is not None:
            auto_dirs[param_name] = "lower" if non_bl[-1][1]["mean"] <= bl_mean else "higher"
        else:
            auto_dirs[param_name] = "lower"

        ecrf.parameters[param_name] = pi
        all_param_stats[param_name] = stat_dict

    return ecrf, all_param_stats, auto_dirs


# ═══════════════════════════════════════════════════════════════
# MANUAL ENTRY FLOW
# ═══════════════════════════════════════════════════════════════

def run_manual_entry_flow():
    st.header("Manual Entry")
    st.caption(
        "Enter pre-computed summary statistics directly — no raw subject data needed."
    )

    col_a, col_b = st.columns(2)
    study_ref   = col_a.text_input("Study Number", placeholder="CSXXXXXX")
    show_center = col_b.checkbox("Show center on charts", value=False)

    # ══════════════════════════════════════════════
    # PHASE 1 — STRUCTURE SETUP
    # ══════════════════════════════════════════════
    st.divider()
    st.subheader("Phase 1 — Define Parameters & Timepoints")

    n_params = st.number_input(
        "How many parameters?", min_value=1, max_value=30, value=1, step=1,
        key="me_n_params",
    )
    n_params = int(n_params)

    # Parameter name inputs
    param_names = []
    pcols = st.columns(min(n_params, 4))
    for i in range(n_params):
        with pcols[i % len(pcols)]:
            name = st.text_input(
                f"Parameter {i + 1} name",
                value=st.session_state.get(f"me_pname_{i}", f"Parameter {i + 1}"),
                key=f"me_pname_{i}",
                placeholder=f"e.g. FIRM, HYDRA, WRINK",
            )
            param_names.append(name.strip())

    # Shared vs per-parameter timepoints
    shared_tps = st.checkbox(
        "All parameters share the same timepoints", value=True, key="me_shared_tps"
    )

    param_tp_map: dict[str, list[str]] = {}  # param_name -> [tp, ...]

    if shared_tps:
        tp_raw = st.text_input(
            "Timepoints (comma-separated)",
            value=st.session_state.get("me_tp_global", "BL, W4, W8, W12"),
            key="me_tp_global",
            placeholder="e.g. BL, W4, W8, W12",
        )
        tps_global = _parse_tp_list(tp_raw)
        for name in param_names:
            param_tp_map[name] = tps_global
    else:
        st.caption("Enter timepoints for each parameter individually.")
        tp_cols = st.columns(min(n_params, 3))
        for i, name in enumerate(param_names):
            with tp_cols[i % len(tp_cols)]:
                tp_raw = st.text_input(
                    f"Timepoints for **{name or f'Parameter {i+1}'}**",
                    value=st.session_state.get(f"me_tp_{i}", "BL, W4, W8, W12"),
                    key=f"me_tp_{i}",
                    placeholder="e.g. BL, W4, W8",
                )
                param_tp_map[name] = _parse_tp_list(tp_raw)

    # Validate before allowing confirm
    all_names_filled = all(n for n in param_names)
    all_tps_filled   = all(len(v) > 0 for v in param_tp_map.values())
    setup_ready      = all_names_filled and all_tps_filled

    if not all_names_filled:
        st.warning("Fill in all parameter names before confirming.")
    if not all_tps_filled:
        st.warning("Each parameter needs at least one timepoint.")

    confirm_setup = st.button(
        "✔ Confirm Setup & Build Entry Tables",
        type="primary",
        disabled=not setup_ready,
    )

    if confirm_setup:
        st.session_state["me_setup_confirmed"] = True
        st.session_state["me_param_names"]     = param_names
        st.session_state["me_param_tp_map"]    = param_tp_map
        st.session_state.pop("me_submitted_df", None)

    if not st.session_state.get("me_setup_confirmed"):
        st.stop()

    # Recover confirmed structure from session state
    confirmed_names  = st.session_state["me_param_names"]
    confirmed_tp_map = st.session_state["me_param_tp_map"]

    # ══════════════════════════════════════════════
    # PHASE 2 — DATA ENTRY (one form per parameter)
    # ══════════════════════════════════════════════
    st.divider()
    st.subheader("Phase 2 — Enter Statistics")
    st.caption(
        "Fill in the statistics for each parameter. "
        "p-value, % improvement, and % subjects improved are disabled for the baseline row."
    )

    # First timepoint in each parameter's list is the baseline
    baseline_map = {name: confirmed_tp_map[name][0] for name in confirmed_names}

    st.divider()

    all_rows: list[dict] = []

    with st.form("me_data_form"):
        for p_idx, p_name in enumerate(confirmed_names):
            tps   = confirmed_tp_map[p_name]
            bl_tp = baseline_map[p_name]

            st.markdown(f"#### {p_name}")
            hdr = st.columns([1.6, 0.9, 1.6, 1.3, 1.6])
            for h, lbl in zip(hdr, [
                "Time Point", "n", "Mean", "p-value", "Mean % Improvement",
            ]):
                h.markdown(f"**{lbl}**")

            for t_idx, tp in enumerate(tps):
                is_bl = (tp == bl_tp)
                c     = st.columns([1.6, 0.9, 1.6, 1.3, 1.6])

                c[0].markdown(
                    f"**{TP_DISPLAY.get(tp, tp)}**" if is_bl
                    else TP_DISPLAY.get(tp, tp)
                )

                n_raw   = c[1].text_input("n",    value="", key=f"me_{p_idx}_{t_idx}_n",
                                           label_visibility="collapsed", placeholder="0")
                mn_raw  = c[2].text_input("Mean", value="", key=f"me_{p_idx}_{t_idx}_mean",
                                           label_visibility="collapsed", placeholder="0.0000")
                pv_raw  = c[3].text_input("p",    value="", key=f"me_{p_idx}_{t_idx}_pv",
                                           label_visibility="collapsed", placeholder="—",
                                           disabled=is_bl)
                pct_raw = c[4].text_input("%imp", value="", key=f"me_{p_idx}_{t_idx}_pct",
                                           label_visibility="collapsed", placeholder="—",
                                           disabled=is_bl)

                all_rows.append({
                    "parameter":   p_name,
                    "timepoint":   tp,
                    "n":           int(_safe_float(n_raw) or 0),
                    "mean":        _safe_float(mn_raw) or 0.0,
                    "p_value":     "" if is_bl else pv_raw.strip(),
                    "pct_change":  None if is_bl else _safe_float(pct_raw),
                })

            if p_idx < len(confirmed_names) - 1:
                st.divider()

        submitted = st.form_submit_button("✔ Submit", type="primary")

    if submitted:
        valid = [r for r in all_rows
                 if r["parameter"].strip() and r["mean"] not in (None, 0.0)]
        if valid:
            df_submitted = pd.DataFrame(valid)
            df_submitted["timepoint"] = df_submitted["timepoint"].apply(_normalise_tp)
            st.session_state["me_submitted_df"] = df_submitted
        else:
            st.warning("No valid rows found — ensure at least one mean value is entered.")

    if "me_submitted_df" not in st.session_state:
        st.stop()

    # ══════════════════════════════════════════════
    # PHASE 3 — REVIEW & EDIT
    # ══════════════════════════════════════════════
    st.divider()
    st.subheader("Phase 3 — Review & Edit")
    st.caption("Edit any cell inline. Changes here feed directly into the charts.")

    manual_df = st.session_state["me_submitted_df"]

    col_cfg = {
        "parameter":  st.column_config.TextColumn("Parameter"),
        "timepoint":  st.column_config.TextColumn("Time Point"),
        "n":          st.column_config.NumberColumn("n", min_value=0, step=1),
        "mean":       st.column_config.NumberColumn("Mean", format="%.4f"),
        "p_value":    st.column_config.TextColumn("p-value"),
        "pct_change": st.column_config.NumberColumn(
                          "Mean % Improvement From Baseline", format="%.2f"),
    }
    display_cols = [c for c in col_cfg if c in manual_df.columns]
    manual_df = st.data_editor(
        manual_df[display_cols],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={k: v for k, v in col_cfg.items() if k in display_cols},
    )

    detected_tps = list(dict.fromkeys(manual_df["timepoint"].tolist()))
    review_bl    = detected_tps[0]

    ecrf_m, stats_m, dirs_m = _manual_df_to_ecrf_and_stats(
        manual_df, study_ref, review_bl
    )
    ecrf_m.n_included = int(manual_df["n"].max()) if manual_df["n"].notna().any() else 0
    keep_m = list(ecrf_m.parameters.keys())

    # ══════════════════════════════════════════════
    # PHASE 4 — IMPROVEMENT DIRECTION
    # ══════════════════════════════════════════════
    st.divider()
    st.subheader("Phase 4 — Improvement Direction")

    dir_mode_m = st.radio(
        "Direction setting",
        options=["Auto-detected", "Decrease= Improvement",
                 "Increase= Improvement", "Per parameter"],
        horizontal=True,
        key="me_dir_mode",
    )
    if dir_mode_m == "Auto-detected":
        imp_dirs_m = dirs_m
    elif dir_mode_m == "All Lower = Improvement":
        imp_dirs_m = {k: "lower" for k in keep_m}
    elif dir_mode_m == "All Higher = Improvement":
        imp_dirs_m = {k: "higher" for k in keep_m}
    else:
        imp_dirs_m = {}
        dcols = st.columns(min(len(keep_m), 3))
        for i, k in enumerate(keep_m):
            with dcols[i % len(dcols)]:
                ch = st.radio(
                    f"**{k}**",
                    ["Lower = Improvement", "Higher = Improvement"],
                    index=0 if dirs_m.get(k, "lower") == "lower" else 1,
                    key=f"me_dir_{k}",
                )
                imp_dirs_m[k] = "lower" if "Lower" in ch else "higher"

    active_tps_m   = sorted(detected_tps, key=tp_sort_key)
    chart_titles_m = {k: f"{study_ref} — {k}" for k in keep_m}

    # ══════════════════════════════════════════════
    # PHASE 5 — PREVIEW
    # ══════════════════════════════════════════════
    st.divider()
    st.subheader("Phase 5 — Preview")

    preview_m = st.selectbox("Preview parameter", options=keep_m, key="me_preview")
    if preview_m and stats_m.get(preview_m):
        fig_m = create_parameter_chart(
            ecrf_m,
            ecrf_m.parameters[preview_m],
            stats_m.get(preview_m, {}),
            imp_dirs_m.get(preview_m, "lower"),
            active_tps=active_tps_m,
            custom_title=chart_titles_m.get(preview_m),
            show_center=show_center,
        )
        if fig_m:
            st.pyplot(fig_m, use_container_width=True)
            plt.close(fig_m)
    elif preview_m:
        st.info("No chart data yet — fill in the statistics above and click Submit.")

    # ══════════════════════════════════════════════
    # PHASE 6 — GENERATE PDF
    # ══════════════════════════════════════════════
    st.divider()
    st.subheader("Phase 6 — Generate PDF")
    st.write(
        f"**{len(keep_m)}** parameter(s) · "
        f"**{len(active_tps_m)}** timepoint(s)"
    )

    if st.button("📄 Generate PDF", type="primary"):
        progress_m  = st.progress(0, text="Starting…")
        pdf_bytes_m = generate_pdf_bytes(
            ecrf_m, stats_m, imp_dirs_m, chart_titles_m,
            active_tps=active_tps_m,
            show_center=show_center,
            progress_bar=progress_m,
        )
        progress_m.empty()
        st.success(f"✅ PDF generated — {len(keep_m)} chart(s).")
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes_m,
            file_name=f"{study_ref.replace(' ', '_')}_manual_charts.pdf",
            mime="application/pdf",
        )

def run_excel_flow():
    # ══════════════════════════════════════════════
    # STEP 1 — Upload
    # ══════════════════════════════════════════════
    st.header("Step 1 — Upload Workbook")
    uploaded = st.file_uploader("Select eCRF Excel workbook (.xlsx / .xls)",
                                 type=["xlsx", "xls"])
    if uploaded is None:
        st.info("Upload an eCRF Excel workbook to begin.")
        st.stop()

    if uploaded.name != st.session_state.uploaded_file_name:
        for k in ("ecrf", "all_param_stats", "auto_dirs"):
            st.session_state[k] = None
        st.session_state.uploaded_file_name = uploaded.name

    file_bytes = uploaded.read()

    sheets     = find_ecrf_sheets(io.BytesIO(file_bytes))
    ecrf_sheet = st.selectbox("eCRF data sheet", options=sheets, index=0)

    show_center = st.checkbox(
            "Show center on charts", value=True,
            help="When unchecked, the center chip is removed from the chart subtitle row.")

    ov_sheet      = find_option_values_sheet(io.BytesIO(file_bytes), ecrf_sheet)
    ov_basenames: set = set()
    if ov_sheet:
        st.success(f"Option Values sheet detected: **{ov_sheet}** — "
                   "Excluded subjects in the status column excluded automatically.")
        ov_basenames = load_ov_variable_basenames(io.BytesIO(file_bytes), ov_sheet)
    else:
        st.warning("No Option Values sheet found. Relying on built-in exclusion list.")

    excl_raw = st.text_input("Manually exclude Subject IDs (comma-separated, optional)",
                              placeholder="e.g. 0012, 0034")
    global_exclusions = []
    if excl_raw:
        for ex in excl_raw.split(','):
            ex = ex.strip()
            if not ex:
                continue
            try:    global_exclusions.append(f"{int(ex):04d}")
            except: global_exclusions.append(ex)

    if st.button("🔍 Parse eCRF Data", type="primary"):
        with st.spinner("Parsing eCRF data…"):
            ecrf, err = parse_ecrf_data(
                io.BytesIO(file_bytes), ecrf_sheet, ov_basenames,
                global_exclusions)
        if err:
            st.error(f"Parse error: {err}")
            st.stop()
        all_param_stats = OrderedDict()
        for param in ecrf.parameters.values():
            all_param_stats[param.base_name] = compute_parameter_stats(ecrf, param)
        auto_dirs = {p.base_name: auto_detect_improvement_direction(ecrf, p)
                     for p in ecrf.parameters.values()}
        st.session_state.ecrf            = ecrf
        st.session_state.all_param_stats = all_param_stats
        st.session_state.auto_dirs       = auto_dirs
        st.rerun()

    if st.session_state.ecrf is None:
        st.stop()

    ecrf: ECRFData               = st.session_state.ecrf
    all_param_stats: OrderedDict = st.session_state.all_param_stats
    auto_dirs: dict              = st.session_state.auto_dirs

    # ══════════════════════════════════════════════
    # STEP 2 — Subject Summary
    # ══════════════════════════════════════════════
    st.divider()
    st.header("Step 2 — Subject Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Study",             ecrf.study_ref or "—")
    c2.metric("Included Subjects", ecrf.n_included)
    c3.metric("Excluded Subjects", len(ecrf.excluded_subjects))

    status_counts = defaultdict(int)
    for s in ecrf.all_subjects_info:
        key = "EXCLUDED" if "EXCLUDED" in s['status_upper'] else s['status_upper']
        status_counts[key] += 1
    st.dataframe(
        pd.DataFrame([{"Status": k, "Count": v,
                       "In Analysis": "✓" if k in INCLUDED_STATUSES else ""}
                      for k, v in sorted(status_counts.items())]),
        hide_index=True, use_container_width=False)

    # ══════════════════════════════════════════════
    # STEP 3 — Timepoint Selection
    # ══════════════════════════════════════════════
    st.divider()
    st.header("Step 3 — Timepoint Selection")

    all_tps    = ecrf.timepoint_order
    mapped_tps = [t for t in all_tps if t.upper() in {x.upper() for x in KNOWN_TP_ORDER}]
    unmapped   = ecrf.unmapped_tps

    active_tps: list = st.multiselect(
        f"Select timepoints to include ({len(all_tps)} detected)",
        options=all_tps,
        default=mapped_tps,
        format_func=lambda t: f"{TP_DISPLAY.get(t, t)} ({t})"
                              + (" ⚠ unmapped" if t in unmapped else ""),
        help="Unmapped timepoints (marked ⚠) were found in the data but are not "
             "in the standard timepoint list.",
    )

    if unmapped:
        with st.expander(
            f"ℹ️ {len(unmapped)} unmapped timepoint(s) found — click to review",
            expanded=False,
        ):
            st.caption(
                "These timepoints exist in the eCRF but are not in the standard list. "
                "Select them above to include them in the analysis."
            )
            st.dataframe(
                pd.DataFrame([{"Raw Key": t,
                               "Already Selected": "✓" if t in active_tps else ""}
                              for t in unmapped]),
                hide_index=True, use_container_width=False)

    if not active_tps:
        st.warning("No timepoints selected — select at least one to continue.")
        st.stop()

    bl_options = list(active_tps)
    bl_default = bl_options.index(ecrf.baseline_prefix) \
                 if ecrf.baseline_prefix in bl_options else 0
    baseline_tp = st.selectbox(
        "Baseline timepoint",
        options=bl_options,
        index=bl_default,
        format_func=lambda t: f"{TP_DISPLAY.get(t, t)} ({t})",
        help="All % change calculations are relative to this timepoint.",
    )
    ecrf.baseline_prefix = baseline_tp

    st.caption("Active timepoints: "
               + " → ".join(TP_DISPLAY.get(t, t) for t in active_tps))

    with st.expander("🔀 Reorder timepoints (drag to fix order)", expanded=False):
        display_labels = [f"{TP_DISPLAY.get(t, t)} ({t})" for t in active_tps]
        sorted_labels  = sort_items(display_labels, direction="vertical")
        sorted_tps     = [active_tps[display_labels.index(l)] for l in sorted_labels]
        if sorted_tps != active_tps:
            active_tps = sorted_tps
            st.caption("Custom order applied: "
                       + " → ".join(TP_DISPLAY.get(t, t) for t in active_tps))

    # ══════════════════════════════════════════════
    # STEP 4 — Orphaned Parameters (conditional)
    # ══════════════════════════════════════════════
    orphan_assignments: dict = {}
    merge_decisions: dict    = {}
    step_offset = 0

    if ecrf.orphaned_params:
        st.divider()
        st.header("Step 4 — Orphaned Parameters")
        st.caption(
            "These columns have no timepoint prefix. Assign each to a timepoint "
            "to include it in the analysis, or leave as (skip)."
        )

        tp_options_for_orphans = ["(skip)"] + active_tps
        cols = st.columns(3)
        for i, (base, pi) in enumerate(ecrf.orphaned_params.items()):
            col_name = ecrf.orphan_col_map.get(base, "?")
            with cols[i % 3]:
                choice = st.selectbox(
                    f"**{base}** — {pi.display_name[:35]}\n`col: {col_name}`",
                    options=tp_options_for_orphans,
                    index=0,
                    format_func=lambda t: "(skip)" if t == "(skip)"
                                          else f"{TP_DISPLAY.get(t, t)} ({t})",
                    key=f"orphan_{base}",
                )
                if choice != "(skip)":
                    orphan_assignments[base] = choice

        if orphan_assignments:
            conflicts = find_orphan_conflicts(ecrf, orphan_assignments, ecrf.df)

            if conflicts:
                st.warning(
                    f"⚠️ **{len(conflicts)} potential merge conflict(s) detected.** "
                    "Review below before continuing."
                )
                with st.expander("Review conflicts and choose merge behaviour",
                                 expanded=True):
                    for c in conflicts:
                        st.markdown(
                            f"**{c['orphan_base']}** (orphan) vs "
                            f"**{c['existing_base']}** (existing) "
                            f"at timepoint **{TP_DISPLAY.get(c['tp'], c['tp'])}** "
                            f"— similarity {c['similarity']:.0%}"
                        )
                        mc1, mc2 = st.columns(2)
                        with mc1:
                            st.markdown(f"*Existing — {c['existing_display']}*")
                            st.markdown(
                                f"n = **{c['existing_n']}** · "
                                f"mean = **{c['existing_mean']:.2f}**"
                                if c["existing_mean"] is not None
                                else f"n = **{c['existing_n']}** · mean = —"
                            )
                        with mc2:
                            st.markdown(f"*Orphan — {c['orphan_display']}*")
                            st.markdown(
                                f"n = **{c['orphan_n']}** · "
                                f"mean = **{c['orphan_mean']:.2f}**"
                                if c["orphan_mean"] is not None
                                else f"n = **{c['orphan_n']}** · mean = —"
                            )
                        st.dataframe(
                            c["sample_df"], hide_index=True,
                            use_container_width=False
                        )
                        decision = st.radio(
                            f"Decision for **{c['orphan_base']}**:",
                            options=["Keep separate", "Merge into existing"],
                            index=0,
                            horizontal=True,
                            key=f"merge_decision_{c['orphan_base']}",
                        )
                        merge_decisions[c["orphan_base"]] = (
                            decision == "Merge into existing"
                        )
                        merge_decisions[f"__target_{c['orphan_base']}"] = c["existing_base"]
                        st.divider()

                unresolved = [
                    c for c in conflicts
                    if c["orphan_base"] not in merge_decisions
                ]
                if unresolved:
                    st.info("Resolve all conflicts above to continue.")
                    st.stop()

            ecrf = apply_orphan_assignments(ecrf, orphan_assignments, merge_decisions)
            ecrf.parameters = group_duplicate_parameters(
                ecrf.parameters, ecrf.df, {}
            )

            for base in orphan_assignments:
                target = base
                if merge_decisions.get(base):
                    for existing_base in ecrf.parameters:
                        if existing_base != base and strip_trailing_digits(base).upper() \
                                == existing_base.upper():
                            target = existing_base
                            break
                if target in ecrf.parameters:
                    all_param_stats[target] = compute_parameter_stats(
                        ecrf, ecrf.parameters[target])
                    if target not in auto_dirs:
                        auto_dirs[target] = auto_detect_improvement_direction(
                            ecrf, ecrf.parameters[target])

            merged_count   = sum(1 for v in merge_decisions.values() if v)
            separate_count = len(orphan_assignments) - merged_count
            if merged_count:
                st.success(f"✅ {merged_count} orphan(s) merged into existing parameters.")
            if separate_count:
                st.success(f"✅ {separate_count} orphan(s) added as new parameters.")

        step_offset = 1

    # ══════════════════════════════════════════════
    # STEP 4/5 — Parameters
    # ══════════════════════════════════════════════
    st.divider()
    st.header(f"Step {4 + step_offset} — Parameters")

    all_param_names = list(ecrf.parameters.keys())
    mode_counts     = defaultdict(int)
    for p in all_param_names:
        mode_counts[classify_parameter(p)] += 1
    dominant_mode    = max(mode_counts, key=mode_counts.get) if mode_counts else "All Parameters"
    default_mode_idx = ANALYSIS_MODES.index(dominant_mode) \
                       if dominant_mode in ANALYSIS_MODES else 0

    analysis_mode = st.selectbox(
        "Analysis type filter", options=ANALYSIS_MODES, index=default_mode_idx)

    is_expert = "Expert" in analysis_mode

    filtered_names = filter_parameters_by_mode(all_param_names, analysis_mode)
    if analysis_mode != "All Parameters":
        n_hidden = len(all_param_names) - len(filtered_names)
        if n_hidden:
            st.caption(f"ℹ️ {n_hidden} parameter(s) hidden by **{analysis_mode}** filter.")

    param_rows = []
    for base in filtered_names:
        p         = ecrf.parameters[base]
        s         = all_param_stats.get(base, {})
        bl_mean   = s.get(ecrf.baseline_prefix, {}).get('mean')
        last_tps  = [tp for tp in active_tps if tp in s and tp != ecrf.baseline_prefix]
        last_mean = s[last_tps[-1]]['mean'] if last_tps else None
        param_rows.append({
            "Base Name":    base + (" ★" if base in orphan_assignments else ""),
            "Display Name": p.display_name[:50],
            "Type":         classify_parameter(base),
            "Timepoints":   ", ".join(p.tp_columns.keys()),
            "BL Mean":      f"{bl_mean:.2f}" if bl_mean is not None else "—",
            "Last Mean":    f"{last_mean:.2f}" if last_mean is not None else "—",
            "Auto Dir":     auto_dirs.get(base, "lower").upper(),
        })

    keep = st.multiselect(
        f"Select parameters to include ({len(filtered_names)} shown / "
        f"{len(all_param_names)} total)",
        options=filtered_names,
        default=filtered_names,
    )
    if param_rows:
        df_rows = pd.DataFrame(param_rows)
        st.dataframe(
            df_rows[df_rows["Base Name"].str.rstrip(" ★").isin(keep)],
            hide_index=True, use_container_width=True)

    # ══════════════════════════════════════════════
    # STEP 5/6 — Improvement Direction
    # ══════════════════════════════════════════════
    st.divider()
    st.header(f"Step {5 + step_offset} — Improvement Direction")

    dir_mode = st.radio(
        "How would you like to set improvement direction?",
        options=["Accept all auto-detected directions",
                 "Set ALL parameters to the same direction",
                 "Set each parameter individually"],
        horizontal=True,
    )
    improvement_dirs: dict = {}
    if dir_mode == "Accept all auto-detected directions":
        improvement_dirs = {k: v for k, v in auto_dirs.items() if k in keep}
        st.caption("Using auto-detected directions.")
    elif dir_mode == "Set ALL parameters to the same direction":
        unified   = st.radio("Direction:", ["Lower = Improvement", "Higher = Improvement"],
                             horizontal=True)
        direction = "lower" if "Lower" in unified else "higher"
        improvement_dirs = {k: direction for k in keep}
    else:
        cols = st.columns(3)
        for i, base in enumerate(keep):
            p = ecrf.parameters[base]
            with cols[i % 3]:
                choice = st.radio(
                    f"**{base}** — {p.display_name[:30]}",
                    options=["Lower = Improvement", "Higher = Improvement"],
                    index=0 if auto_dirs.get(base, "lower") == "lower" else 1,
                    key=f"dir_{base}",
                )
                improvement_dirs[base] = "lower" if "Lower" in choice else "higher"

    # ══════════════════════════════════════════════
    # STEP 6/7 — Chart Titles
    # ══════════════════════════════════════════════
    st.divider()
    st.header(f"Step {6 + step_offset} — Chart Titles")
    with st.expander("Edit chart titles (click to expand)", expanded=False):
        chart_titles: dict = {}
        for base in keep:
            p     = ecrf.parameters[base]
            deflt = f"{ecrf.study_ref}: {analysis_mode} — {p.display_name}"
            chart_titles[base] = st.text_input(f"{base}", value=deflt,
                                                key=f"title_{base}")
    if not chart_titles:
        chart_titles = {
            base: f"{ecrf.study_ref}: {analysis_mode} — "
                  f"{ecrf.parameters[base].display_name}"
            for base in keep
        }

    # ══════════════════════════════════════════════
    # STEP 7/8 — Preview
    # ══════════════════════════════════════════════
    st.divider()
    st.header(f"Step {7 + step_offset} — Preview")
    if keep:
        preview_param = st.selectbox(
            "Preview chart for parameter:", options=keep,
            format_func=lambda b: f"{b} — {ecrf.parameters[b].display_name[:50]}")
        if preview_param:
            fig = create_parameter_chart(
                ecrf, ecrf.parameters[preview_param],
                all_param_stats.get(preview_param, {}),
                improvement_dirs.get(preview_param, "lower"),
                active_tps=active_tps,
                custom_title=chart_titles.get(preview_param),
                show_center=show_center,
            )
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
    else:
        st.warning("No parameters selected.")

    # ══════════════════════════════════════════════
    # STEP 8/9 — Data Quality
    # ══════════════════════════════════════════════
    st.divider()
    st.header(f"Step {8 + step_offset} — Data Quality")

    if keep and active_tps:
        dq_issues   = scan_data_quality(ecrf, keep, active_tps)
        dq_summary  = summarise_data_quality(dq_issues)
        total       = dq_summary['total_issues']
        n_affected  = dq_summary['affected_subjects']
        per_subject = dq_summary['per_subject']

        if total == 0:
            st.success("✅ No data quality issues detected.")
        else:
            st.warning(
                f"⚠️ **{total} missing value(s)** across **{n_affected} subject(s)**. "
                "Charts will use available data only."
            )
            with st.expander(
                f"🟡 Per-subject missing data ({n_affected} subject(s))",
                expanded=True):
                ps_rows = [{"Subject ID": sid, "Missing": entry}
                           for sid in sorted(per_subject)
                           for entry in per_subject[sid]]
                st.dataframe(pd.DataFrame(ps_rows),
                             hide_index=True, use_container_width=True)
    else:
        st.info("Select parameters and timepoints above to run the data quality scan.")

    # ══════════════════════════════════════════════
    # STEP 9/10 — Statistical Summary
    # ══════════════════════════════════════════════
    st.divider()
    st.header(f"Step {9 + step_offset} — Statistical Summary")
    st.caption("Review stats before generating the PDF. "
               "Verify % change direction and p-values are as expected.")

    if keep and active_tps:
        stats_df = build_stats_table(
            ecrf, all_param_stats, improvement_dirs,
            keep, active_tps, analysis_mode)

        if not stats_df.empty:
            def highlight_sig(val):
                if isinstance(val, str) and "*" in val:
                    return "background-color: #e6f4ea; color: #1a6e35; font-weight: bold"
                return ""

            st.dataframe(
                stats_df.style.applymap(highlight_sig, subset=["p-value"])
                if "p-value" in stats_df.columns else stats_df,
                hide_index=True,
                use_container_width=True,
                height=min(600, 38 + 35 * len(stats_df)),
            )

            if is_expert:
                asfs_df = build_asfs_threshold_table(ecrf, keep, active_tps)
                if not asfs_df.empty:
                    st.subheader("ASFS Threshold Classification — Frequency n (%)")
                    st.caption(
                        "Normal / Very Slight (0–15) · Mild (16–24) · "
                        "Moderate (25–34) · Severe (35–80)"
                    )
                    st.dataframe(asfs_df, hide_index=True,
                                 use_container_width=True,
                                 height=min(400, 38 + 35 * len(asfs_df)))
                else:
                    st.info("No ASFS Score parameter detected in the current selection. "
                            "The ASFS threshold table requires a column whose base name "
                            "matches ASFS, ASFSSCORE, ASFS_SCORE, or ASFSTOTAL.")

            csv_bytes = stats_df.to_csv(index=False).encode("utf-8")
            stem      = Path(uploaded.name).stem
            st.download_button(
                label="⬇️ Download Stats Table (CSV)",
                data=csv_bytes,
                file_name=f"{stem}_stats_summary.csv",
                mime="text/csv",
            )
        else:
            st.info("No stats available for the current selection.")
    else:
        st.info("Select parameters and timepoints above to generate the stats table.")

    # ══════════════════════════════════════════════
    # STEP 10/11 — Generate PDF
    # ══════════════════════════════════════════════
    st.divider()
    st.header(f"Step {10 + step_offset} — Generate PDF")
    st.write(f"**{len(keep)}** parameter(s) · "
             f"**{len(active_tps)}** timepoint(s) · "
             f"**{ecrf.n_included}** included subject(s)")

    if st.button("📄 Generate PDF", type="primary", disabled=len(keep) == 0):
        ecrf_subset            = ECRFData()
        ecrf_subset.__dict__   = {k: v for k, v in ecrf.__dict__.items()}
        ecrf_subset.parameters = OrderedDict(
            (k, v) for k, v in ecrf.parameters.items() if k in keep)

        progress  = st.progress(0, text="Starting…")
        pdf_bytes = generate_pdf_bytes(
            ecrf_subset, all_param_stats, improvement_dirs, chart_titles,
            active_tps=active_tps,
            show_center=show_center,
            progress_bar=progress,
        )
        progress.empty()

        stem = Path(uploaded.name).stem
        st.success(f"✅ PDF generated — {len(keep)} chart(s).")
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"{stem}_eCRF_Charts.pdf",
            mime="application/pdf")


def main():
    st.set_page_config(page_title="eCRF Chart Generator",
                       page_icon="📊", layout="wide")
    st.title("📊 eCRF Chart Generator v1.6")
    st.caption("Generates mean-change-from-baseline charts for eCRF data.")

    for key in ("ecrf", "all_param_stats", "auto_dirs", "uploaded_file_name"):
        if key not in st.session_state:
            st.session_state[key] = None

    input_mode = st.radio(
        "Input mode",
        options=["Upload Excel", "Manual Entry"],
        horizontal=True,
    )

    if input_mode == "Upload Excel":
        run_excel_flow()
    else:
        run_manual_entry_flow()


if __name__ == "__main__":
    main()