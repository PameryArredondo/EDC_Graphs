"""
EDC Graphs Generator v 2.0 — Streamlit Edition
===============================================
Usage:
    streamlit run EDC_Graphs.py
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

def _detect_file_format(file_bytes: bytes) -> str:
    try:
        raw_sheet = _detect_monaderm_sheet(io.BytesIO(file_bytes))
        if raw_sheet is None:
            return "edc"
        peek = pd.read_excel(io.BytesIO(file_bytes), sheet_name=raw_sheet, nrows=1)
        cols = {str(c).strip().upper() for c in peek.columns}
        if MONADERM_REQUIRED_COLS.issubset(cols):
            return "monaderm"
    except Exception:
        pass
    return "edc"

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
# 8. Ecrf DATA PARSING
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
        return None, "EDC sheet is missing SUBJECT ID or STATUS column."

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
# SECTION 8b — MONADERM HELPERS & PARSER
# ═══════════════════════════════════════════════════════════════
MONADERM_REQUIRED_COLS = {"SUBJECT", "KINETIC", "PARAMETER", "VALUE", "REPETITION"}


def _detect_monaderm_sheet(excel_file) -> str | None:
    """Return the first sheet that looks like a Monaderm RAW DATA sheet."""
    xls = pd.ExcelFile(excel_file)
    for sn in xls.sheet_names:
        sn_l = sn.lower()
        if "raw" in sn_l and "data" in sn_l:
            return sn
        if sn_l == "rawdata":
            return sn
    for sn in xls.sheet_names:
        try:
            peek = pd.read_excel(excel_file, sheet_name=sn, nrows=1)
            cols = {str(c).strip().upper() for c in peek.columns}
            if MONADERM_REQUIRED_COLS.issubset(cols):
                return sn
        except Exception:
            pass
    return None


def _scan_monaderm_file(file_bytes: bytes, sheet_name: str) -> dict:
    """Quick dimension discovery — no averaging."""
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=0)
    except Exception as exc:
        return {"error": str(exc), "params": [], "tps": [], "zones": [], "probe_names": []}

    df.columns = [str(c).strip().upper() for c in df.columns]
    missing = MONADERM_REQUIRED_COLS - set(df.columns)
    if missing:
        return {"error": f"Missing column(s): {', '.join(sorted(missing))}",
                "params": [], "tps": [], "zones": [], "probe_names": []}

    params      = sorted(df["PARAMETER"].dropna().astype(str).str.strip().unique().tolist())
    tps_raw     = df["KINETIC"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    tps         = sorted(tps_raw, key=tp_sort_key)
    zones       = (sorted(df["ZONE"].dropna().astype(str).str.strip().unique().tolist())
                   if "ZONE" in df.columns else ["S1"])
    probe_names = (sorted(df["PROBE_NAME"].dropna().astype(str).str.strip().unique().tolist())
                   if "PROBE_NAME" in df.columns else [])
    return {"error": None, "params": params, "tps": tps,
            "zones": zones, "probe_names": probe_names}


def _load_rep_df(
    file_bytes: bytes,
    sheet_name: str,
    selected_params: list[str],
    selected_tps: list[str],
    global_exclusions: list[str] | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Load the RAW DATA sheet and return a clean rep-level DataFrame:
        SUBJECT | ZONE | KINETIC | PARAMETER | REPETITION | VALUE
    One row per raw measurement — no averaging yet.
    """
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=0)
    except Exception as exc:
        return None, str(exc)

    df.columns = [str(c).strip().upper() for c in df.columns]
    missing = MONADERM_REQUIRED_COLS - set(df.columns)
    if missing:
        return None, f"Missing column(s): {', '.join(sorted(missing))}"

    df["SUBJECT"]    = df["SUBJECT"].astype(str).str.strip().apply(normalize_subject_id)
    df["KINETIC"]    = df["KINETIC"].astype(str).str.strip().str.upper()
    df["PARAMETER"]  = df["PARAMETER"].astype(str).str.strip()
    df["REPETITION"] = pd.to_numeric(df["REPETITION"], errors="coerce").fillna(0).astype(int)
    df["VALUE"]      = pd.to_numeric(df["VALUE"], errors="coerce")
    df["ZONE"]       = df["ZONE"].astype(str).str.strip() if "ZONE" in df.columns else "S1"

    df = df.dropna(subset=["VALUE"])
    if global_exclusions:
        df = df[~df["SUBJECT"].isin(set(global_exclusions))]
    df = df[df["PARAMETER"].isin(selected_params) & df["KINETIC"].isin(selected_tps)]
    if df.empty:
        return None, "No data remains after filtering."

    return df[["SUBJECT", "ZONE", "KINETIC", "PARAMETER", "REPETITION", "VALUE"]], None

def _compute_stats_from_rep_df(
    rep_df: pd.DataFrame,
    included_reps: dict,          # {(subj, zone, tp, param): set[int]}
    baseline_tp: str,
    dropped_subjects: list[str] | None = None,
) -> pd.DataFrame:
    """
    Given the rep-level DataFrame and a dict of which reps to include,
    compute per-group means, then build a stats summary matching the
    format produced by build_stats_table():

        Assessment | Time Point | n | Mean ± SD | p-value | Mean % Change From Baseline

    p-values are paired t-tests vs baseline (same as compute_parameter_stats path).
    """
    from scipy import stats as scipy_stats

    if dropped_subjects:
        rep_df = rep_df[~rep_df["SUBJECT"].isin(set(dropped_subjects))]

    # ── Step 1: compute per-group means from selected reps ───────────────────
    records = []
    for (subj, zone, tp, param), grp in rep_df.groupby(
        ["SUBJECT", "ZONE", "KINETIC", "PARAMETER"]
    ):
        key     = (subj, zone, tp, param)
        allowed = included_reps.get(key)
        if allowed is not None:
            vals = grp[grp["REPETITION"].isin(allowed)]["VALUE"].values
        else:
            vals = grp["VALUE"].values
        if len(vals) == 0:
            continue
        records.append({
            "SUBJECT": subj, "ZONE": zone, "KINETIC": tp,
            "PARAMETER": param,
            "MEAN": float(np.mean(vals)),
            "N_REPS": len(vals),
        })

    if not records:
        return pd.DataFrame()

    df_means = pd.DataFrame(records)

    # ── Step 2: build summary rows ────────────────────────────────────────────
    unique_zones   = sorted(df_means["ZONE"].unique())
    unique_params  = sorted(df_means["PARAMETER"].unique())
    active_tps     = sorted(df_means["KINETIC"].unique(), key=tp_sort_key)

    summary_rows = []

    for zone in unique_zones:
        for param in unique_params:
            param_label = f"{param} — {zone}" if len(unique_zones) > 1 else param
            df_pp = df_means[(df_means["ZONE"] == zone) & (df_means["PARAMETER"] == param)]

            # baseline subject means
            bl_df   = df_pp[df_pp["KINETIC"] == baseline_tp]
            bl_map  = dict(zip(bl_df["SUBJECT"], bl_df["MEAN"]))
            bl_vals = np.array(list(bl_map.values()))
            bl_mean = float(np.mean(bl_vals)) if len(bl_vals) > 0 else None

            for tp in active_tps:
                tp_df   = df_pp[df_pp["KINETIC"] == tp]
                tp_map  = dict(zip(tp_df["SUBJECT"], tp_df["MEAN"]))
                tp_vals = np.array(list(tp_map.values()))

                if len(tp_vals) == 0:
                    continue

                mean_val = float(np.mean(tp_vals))
                n        = len(tp_vals)
                is_bl    = (tp == baseline_tp)

                # paired t-test vs baseline on matched subjects
                pval_str = ""
                sig_flag = ""
                if not is_bl and bl_mean is not None:
                    common   = sorted(set(bl_map) & set(tp_map))
                    if len(common) >= 2:
                        bl_paired = np.array([bl_map[s] for s in common])
                        tp_paired = np.array([tp_map[s] for s in common])
                        if np.std(bl_paired - tp_paired) > 0:
                            _, pv = scipy_stats.ttest_rel(bl_paired, tp_paired)
                            pval_str = f"{pv:.3f}" + ("*" if pv < 0.05 else "")
                            sig_flag = "Significant" if pv < 0.05 else "Not significant"
                        else:
                            pval_str = "1.000"
                            sig_flag = "Not significant"
                    else:
                        pval_str = "—"
                        sig_flag = "—"

                pct_str = ""
                if not is_bl and bl_mean is not None and bl_mean != 0:
                    pct = ((mean_val - bl_mean) / bl_mean) * 100
                    pct_str = f"{round_half_up(pct, 2):.2f}%"
                elif not is_bl:
                    pct_str = "N/A"

                summary_rows.append({
                    "Assessment":                  param_label,
                    "Time Point":                  TP_DISPLAY.get(tp, tp),
                    "_tp_code":                    tp,          # hidden sort key
                    "_is_baseline":                is_bl,
                    "n":                           n,
                    "Mean":        fmt_value(mean_val),
                    "_mean_float": mean_val,    # for chart injection
                    "p-value":                     pval_str,
                    "Mean % Change From Baseline": pct_str,
                    "Significant":                 sig_flag,
                })

    return pd.DataFrame(summary_rows)

def _stats_df_to_ecrf_and_stats(
    stats_df: pd.DataFrame,
    study_ref: str,
    baseline_tp_display: str,
) -> tuple["ECRFData", dict, dict]:
    """
    Convert an edited stats summary DataFrame back into ECRFData + param_stats
    so create_parameter_chart / generate_pdf_bytes work unchanged.

    Uses the same pattern as _manual_df_to_ecrf_and_stats() in the Manual
    Entry flow — charts are driven by the Mean values in the table.
    """
    ecrf                 = ECRFData()
    ecrf.study_ref       = study_ref
    ecrf.df              = pd.DataFrame()
    ecrf.baseline_prefix = baseline_tp_display   # display name used as key

    # Rebuild a timepoint order from the table
    tp_order = []
    seen: set = set()
    for tp in stats_df["Time Point"]:
        if tp not in seen:
            tp_order.append(tp)
            seen.add(tp)
    # Sort by known order where possible
    ecrf.timepoint_order = sorted(tp_order, key=lambda t: tp_sort_key(
        next((k for k, v in TP_DISPLAY.items() if v == t), t)
    ))

    all_param_stats: dict = {}
    auto_dirs: dict       = {}

    for param_name, grp in stats_df.groupby("Assessment", sort=False):
        pi       = ParameterInfo(param_name, param_name)
        stat_dict = OrderedDict()
        bl_mean  = None

        for _, row in grp.iterrows():
            tp       = row["Time Point"]
            mean_val = row["_mean_float"]
            n_val    = int(row["n"]) if str(row["n"]).isdigit() else 0
            is_bl    = row["_is_baseline"]

            if is_bl:
                bl_mean = mean_val

            pct_val  = None
            if not is_bl and bl_mean is not None and bl_mean != 0:
                pct_raw = row["Mean % Change From Baseline"].replace("%", "").strip()
                try:
                    pct_val = float(pct_raw)
                except (ValueError, AttributeError):
                    pct_val = ((mean_val - bl_mean) / bl_mean) * 100 if bl_mean else None

            sig_str  = str(row.get("Significant", ""))
            sig_flag = "significant" in sig_str.lower() and "*" in str(row.get("p-value", ""))

            stat_dict[tp] = {
                "mean":        mean_val,
                "std":         0.0,
                "n":           n_val,
                "pct_change":  pct_val,
                "values":      np.array([mean_val]),
                "significant": sig_flag,
            }
            pi.tp_columns[tp] = f"__mn__{tp}"

        ecrf.parameters[param_name] = pi
        all_param_stats[param_name] = stat_dict

        non_bl = [(tp, s) for tp, s in stat_dict.items() if not stats_df[
            (stats_df["Assessment"] == param_name) &
            (stats_df["Time Point"] == tp)]["_is_baseline"].any()]
        if non_bl and bl_mean is not None:
            auto_dirs[param_name] = (
                "lower" if non_bl[-1][1]["mean"] <= bl_mean else "higher"
            )
        else:
            auto_dirs[param_name] = "lower"

    ecrf.n_included = int(stats_df["n"].max()) if not stats_df.empty else 0
    return ecrf, all_param_stats, auto_dirs


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
    assignments: dict,
    df_included: "pd.DataFrame",
    similarity_threshold: float = 0.75,
) -> list:
    from difflib import SequenceMatcher

    conflicts = []

    for orphan_base, tp in assignments.items():
        if tp == "" or orphan_base not in ecrf.orphaned_params:
            continue

        orphan_col = ecrf.orphan_col_map.get(orphan_base)
        if orphan_col is None or orphan_col not in df_included.columns:
            continue

        stripped = strip_trailing_digits(orphan_base)

        matched_existing = None
        best_sim = 0.0

        orphan_display = ecrf.orphaned_params[orphan_base].display_name or orphan_base

        for existing_base, existing_param in ecrf.parameters.items():
            if tp not in existing_param.tp_columns:
                continue

            name_sim = SequenceMatcher(
                None,
                stripped.upper(),
                existing_base.upper(),
            ).ratio()

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

        orphan_vals   = pd.to_numeric(df_included[orphan_col],   errors="coerce").dropna()
        existing_vals = pd.to_numeric(df_included[existing_col], errors="coerce").dropna()

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
    if merge_decisions is None:
        merge_decisions = {}

    for orphan_base, tp in assignments.items():
        if tp == "" or orphan_base not in ecrf.orphaned_params:
            continue

        orphan_col = ecrf.orphan_col_map.get(orphan_base)
        if orphan_col is None:
            continue

        if tp not in ecrf.timepoint_order:
            ecrf.timepoint_order = sorted(
                ecrf.timepoint_order + [tp], key=tp_sort_key
            )

        stripped     = strip_trailing_digits(orphan_base)
        should_merge = merge_decisions.get(orphan_base, False)

        target_base = None
        if should_merge:
            stashed = merge_decisions.get(f"__target_{orphan_base}")
            if stashed and stashed in ecrf.parameters and tp in ecrf.parameters[stashed].tp_columns:
                target_base = stashed
            else:
                for existing_base, existing_param in ecrf.parameters.items():
                    if tp in existing_param.tp_columns and (
                        existing_base.upper() == stripped.upper()
                        or existing_base.upper() == orphan_base.upper()
                    ):
                        target_base = existing_base
                        break

        if target_base and should_merge:
            ecrf.parameters[target_base].tp_columns[tp] = orphan_col
            ecrf.col_map[(tp, target_base)] = orphan_col
        else:
            if orphan_base not in ecrf.parameters:
                ecrf.parameters[orphan_base] = ecrf.orphaned_params[orphan_base]
            ecrf.parameters[orphan_base].tp_columns[tp] = orphan_col
            ecrf.col_map[(tp, orphan_base)] = orphan_col

    return ecrf

# ═══════════════════════════════════════════════════════════════
# 11. STATISTICS
# ═══════════════════════════════════════════════════════════════

def compute_parameter_stats(ecrf, param, min_pairs=3):
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
    return f"{round_half_up(value, decimals):,.{decimals}f}"


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

    fig = ax.get_figure()
    fig.canvas.draw()

    disp_r = 12.0
    inv    = ax.transData.inverted()
    origin = inv.transform(ax.transData.transform((x, y)))
    corner = inv.transform(ax.transData.transform((x, y)) + np.array([disp_r, disp_r]))
    rx = abs(corner[0] - origin[0])
    ry = abs(corner[1] - origin[1])

    ry = min(ry, height * 0.4)
    rx = min(rx, width  * 0.4)

    verts = [
        (x,             y),
        (x,             y + height - ry),
        (x,             y + height),
        (x + rx,        y + height),
        (x + width - rx,y + height),
        (x + width,     y + height),
        (x + width,     y + height - ry),
        (x + width,     y),
        (x,             y),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), fc=color, ec="none", zorder=3))


def draw_pill_badge(ax, x_center, y_center, text, bg_color, text_color='white',
                    fontsize=10, pad_x=0.35, pad_y=0.12):
    fig      = ax.get_figure()
    bar_width = 0.55

    left_disp  = ax.transData.transform((x_center - bar_width/2, y_center))
    right_disp = ax.transData.transform((x_center + bar_width/2, y_center))
    top_disp   = ax.transData.transform((x_center, y_center))

    left_fig  = fig.transFigure.inverted().transform(left_disp)
    right_fig = fig.transFigure.inverted().transform(right_disp)
    top_fig   = fig.transFigure.inverted().transform(top_disp)

    fx = (left_fig[0] + right_fig[0]) / 2
    fy = top_fig[1] + 0.02
    w  = right_fig[0] - left_fig[0]
    h  = 0.035

    fig.patches.append(FancyBboxPatch(
        (fx - w/2, fy - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={h/2}",
        ec="none", fc=bg_color, alpha=1.0, zorder=5,
        transform=fig.transFigure, clip_on=False))
    fig.text(fx, fy, text, ha='center', va='center',
             fontsize=fontsize, fontweight='bold', color=text_color, zorder=6)
    
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

    # ═══════════════════════════════════════════════════════════════
    # 1. APPLY AXIS STYLING FIRST
    # ═══════════════════════════════════════════════════════════════
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean", fontsize=13, fontweight='bold')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#E4E4E4', linewidth=1)
    ax.xaxis.grid(False)

    raw_interval = max(y_max, 1.0) / 8
    magnitude    = 10 ** np.floor(np.log10(raw_interval))
    residual     = raw_interval / magnitude
    if residual <= 1.5:   step = 1
    elif residual <= 3.5: step = 2
    elif residual <= 7.5: step = 5
    else:                 step = 10
    tick_interval = max(0.5, step * magnitude)

    ax.yaxis.set_major_locator(plt.MultipleLocator(tick_interval))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(tick_interval / 2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.tick_params(axis='y', which='major', length=6, width=1.2,
                   color=COLORS['text_main'])
    ax.tick_params(axis='y', which='minor', length=3, width=0.6, color='#999999')
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('#BBBBBB')
    ax.spines['left'].set_linewidth(1.2)
    ax.set_xlabel("Timepoint", fontsize=13, fontweight='bold')

    if len(set(ns)) > 1:
        for i in range(len(ordered_tps)):
            n_val = ns[i]
            if n_val and n_val > 0:
                ax.text(x_pos[i], -y_max * 0.08, f"n={n_val}", ha='center', va='top',
                        fontsize=8, color=COLORS['text_sub'])

    # ═══════════════════════════════════════════════════════════════
    # 2. ADD MAIN TITLES AND FOOTNOTE
    # ═══════════════════════════════════════════════════════════════
    title_main = custom_title or f"{ecrf.study_ref} — {param.display_name}"
    fig.text(0.5, 0.97, title_main, ha='center', va='top',
             fontsize=17, fontweight='bold', color=COLORS['text_main'])
    fig.text(0.5, 0.92, "Mean by Timepoint", ha='center', va='top',
             fontsize=13, fontweight='medium', color=COLORS['text_sub'])
    fig.text(0.5, 0.82,
             "Values above bars = % improvement from baseline · "
             "* indicates statistical significance (p < 0.05, paired t-test)",
             ha='center', va='top', fontsize=9.5,
             color=COLORS['text_sub'], style='italic')

    # ═══════════════════════════════════════════════════════════════
    # 3. LOCK LAYOUT BEFORE CALCULATING ABSOLUTE FIGURE POSITIONS
    # ═══════════════════════════════════════════════════════════════
    plt.tight_layout(rect=[0, 0.09, 1, 0.78])
    fig.canvas.draw()

    # ═══════════════════════════════════════════════════════════════
    # 4. DRAW PILL BADGES (position calculated after layout is locked)
    # ═══════════════════════════════════════════════════════════════
    for i, tp in enumerate(ordered_tps):
        if tp == bl_prefix:
            continue
        pct = param_stats[tp].get('pct_change')
        if pct is not None:
            dpct  = -pct if improvement_dir == "lower" else pct
            sign  = "+" if dpct > 0 else ""
            label = f"{sign}{fmt_value(dpct)}%"
            draw_pill_badge(ax, x_pos[i], means[i] + y_max * 0.015, label,
                            bg_color=bar_colors[i], text_color='white',
                            fontsize=10, pad_x=0.3, pad_y=0.08)

    # ═══════════════════════════════════════════════════════════════
    # 5. DRAW SUBTITLE CHIPS
    # ═══════════════════════════════════════════════════════════════
    dir_label = ("Decrease = Improvement" if improvement_dir == "lower"
                 else "Increase = Improvement")
    n_chip = (f"n={max(ns)} Included Subjects" if len(set(ns)) == 1
              else f"n={min(ns)}–{max(ns)} Included Subjects")
    if show_center and CENTER_FILTER:
        chip_texts = [f"Center: {CENTER_FILTER}", n_chip, dir_label]
    else:
        chip_texts = [n_chip, dir_label]

    chip_gap     = 0.22
    chip_start_x = 0.5 - (len(chip_texts) - 1) * chip_gap / 2
    fig.canvas.draw()
    for ci, ct in enumerate(chip_texts):
        draw_pill_chip(fig, chip_start_x + ci * chip_gap, 0.855, ct, fontsize=9.5)

    # ═══════════════════════════════════════════════════════════════
    # 6. DRAW LEGEND
    # ═══════════════════════════════════════════════════════════════
    x_start = 0.125
    for color, label in [(COLORS['baseline'], "Baseline"),
                          (COLORS['improved'], "Improved from Baseline"),
                          (COLORS['worsened'], "Worsened from Baseline")]:
        fig.patches.append(Rectangle((x_start, 0.03), 0.015, 0.015,
            transform=fig.transFigure, color=color, zorder=2, clip_on=False))
        fig.text(x_start + 0.02, 0.033, label, fontsize=9, color=COLORS['text_main'])
        x_start += 0.02 + len(label) * 0.006 + 0.03

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
        return float(str(s).strip().rstrip("%"))
    except Exception:
        return None

def _normalise_tp(raw: str) -> str:
    raw   = raw.strip()
    upper = raw.upper()
    if upper in KNOWN_TP_ORDER:
        return upper
    for k, v in TP_DISPLAY.items():
        if v.upper() == upper:
            return k
    return upper


def _parse_tp_list(raw: str) -> list[str]:
    return list(dict.fromkeys(
        _normalise_tp(t) for t in raw.split(",") if t.strip()
    ))


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
            n_val    = int(row["n"]) if pd.notna(row["n"]) else 0
            is_bl    = (tp == baseline_tp)

            if is_bl:
                bl_mean = mean_val

            sig = bool(row.get("significant", False)) if not is_bl else False

            stat_dict[tp] = {
                "mean":        mean_val,
                "std":         0.0,
                "n":           n_val,
                "pct_change":  None,
                "values":      np.array([mean_val]),
                "significant": sig,
            }
            pi.tp_columns[tp] = f"__manual__{tp}"

        if bl_mean is not None and bl_mean != 0:
            for tp, s in stat_dict.items():
                if tp != baseline_tp:
                    s["pct_change"] = ((s["mean"] - bl_mean) / bl_mean) * 100

        non_bl = [(tp, s) for tp, s in stat_dict.items() if tp != baseline_tp]
        if non_bl and bl_mean is not None:
            auto_dirs[param_name] = (
                "lower" if non_bl[-1][1]["mean"] <= bl_mean else "higher"
            )
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

    col_a, col_b, col_c = st.columns(3)
    study_num    = col_a.text_input("Study Number",    placeholder="e.g. CS251037")
    analysis_lbl = col_b.text_input("Analysis Type",   placeholder="e.g. Expert Grading")
    show_center  = col_c.checkbox("Show center on charts", value=False)
    study_ref    = (
        f"{study_num}: {analysis_lbl}".strip(": ")
        if study_num or analysis_lbl else ""
    )

    st.divider()
    st.subheader("Phase 1 — Define Parameters & Timepoints")

    n_params = int(st.number_input(
        "How many parameters?", min_value=1, max_value=30, value=1, step=1,
        key="me_n_params",
    ))

    param_names = []
    pcols = st.columns(min(n_params, 4))
    for i in range(n_params):
        with pcols[i % len(pcols)]:
            name = st.text_input(
                f"Parameter {i + 1} name",
                value=st.session_state.get(f"me_pname_{i}", f"Parameter {i + 1}"),
                key=f"me_pname_{i}",
                placeholder="e.g. FIRM, HYDRA, WRINK",
            )
            param_names.append(name.strip())

    shared_tps = st.checkbox(
        "All parameters share the same timepoints", value=True, key="me_shared_tps"
    )

    param_tp_map: dict[str, list[str]] = {}

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

    confirmed_names  = st.session_state["me_param_names"]
    confirmed_tp_map = st.session_state["me_param_tp_map"]

    st.divider()
    st.subheader("Phase 2 — Enter Statistics")
    st.caption(
        "Enter **n** and **Mean** for each timepoint. "
        "For non-baseline rows, select whether the result is statistically significant "
        "(p < 0.05). % change from baseline is calculated automatically."
    )

    baseline_map = {name: confirmed_tp_map[name][0] for name in confirmed_names}

    all_rows: list[dict] = []

    with st.form("me_data_form"):
        for p_idx, p_name in enumerate(confirmed_names):
            tps   = confirmed_tp_map[p_name]
            bl_tp = baseline_map[p_name]

            st.markdown(f"#### {p_name}")

            hdr = st.columns([1.6, 0.9, 1.8, 1.6])
            for h, lbl in zip(hdr, ["Time Point", "n", "Mean", "Significant?"]):
                h.markdown(f"**{lbl}**")

            for t_idx, tp in enumerate(tps):
                is_bl = (tp == bl_tp)
                c     = st.columns([1.6, 0.9, 1.8, 1.6])

                c[0].markdown(
                    f"**{TP_DISPLAY.get(tp, tp)}** *(baseline)*"
                    if is_bl else TP_DISPLAY.get(tp, tp)
                )

                n_raw  = c[1].text_input(
                    "n", value="", key=f"me_{p_idx}_{t_idx}_n",
                    label_visibility="collapsed", placeholder="0",
                )
                mn_raw = c[2].text_input(
                    "Mean", value="", key=f"me_{p_idx}_{t_idx}_mean",
                    label_visibility="collapsed", placeholder="0.0000",
                )

                if is_bl:
                    c[3].markdown("*n/a — baseline*")
                    sig_val = False
                else:
                    sig_choice = c[3].selectbox(
                        "sig",
                        options=["Not significant", "Significant (p < 0.05)"],
                        index=0,
                        key=f"me_{p_idx}_{t_idx}_sig",
                        label_visibility="collapsed",
                    )
                    sig_val = (sig_choice == "Significant (p < 0.05)")

                all_rows.append({
                    "parameter":   p_name,
                    "timepoint":   tp,
                    "n":           int(_safe_float(n_raw) or 0),
                    "mean":        _safe_float(mn_raw) or 0.0,
                    "significant": sig_val,
                })

            if p_idx < len(confirmed_names) - 1:
                st.divider()

        submitted = st.form_submit_button("✔ Submit", type="primary")

    if submitted:
        valid = [r for r in all_rows if r["parameter"].strip() and r["mean"] is not None]
        if valid:
            df_submitted = pd.DataFrame(valid)
            df_submitted["timepoint"] = df_submitted["timepoint"].apply(_normalise_tp)
            st.session_state["me_submitted_df"] = df_submitted
        else:
            st.warning("No valid rows found — ensure at least one mean value is entered.")

    if "me_submitted_df" not in st.session_state:
        st.stop()

    manual_df = st.session_state["me_submitted_df"]

    first_tp_per_param = (
        manual_df.groupby("parameter", sort=False)["timepoint"]
        .first()
        .to_dict()
    )
    baseline_tp_global = (
        next(iter(first_tp_per_param.values())) if first_tp_per_param else "BL"
    )

    ecrf_m, stats_m, dirs_m = _manual_df_to_ecrf_and_stats(
        manual_df, study_ref, baseline_tp_global,
    )
    ecrf_m.n_included = int(manual_df["n"].max()) if manual_df["n"].notna().any() else 0
    keep_m            = list(ecrf_m.parameters.keys())

    active_tps_m = sorted(
        set(tp for s in stats_m.values() for tp in s.keys()),
        key=tp_sort_key,
    )

    st.divider()
    st.subheader("Phase 3 — Confirm % Change & Improvement Direction")
    st.caption(
        "% change from baseline is calculated from your entered means. "
        "Review the table below — if a value looks wrong, go back and correct "
        "the mean in Phase 2. Then set the improvement direction."
    )

    preview_rows = []
    for p_name in keep_m:
        s_dict  = stats_m[p_name]
        bl_mean = s_dict.get(baseline_tp_global, {}).get("mean")
        for tp in active_tps_m:
            if tp not in s_dict:
                continue
            s        = s_dict[tp]
            is_bl    = (tp == baseline_tp_global)
            pct      = s.get("pct_change")
            pct_str  = "—" if is_bl or pct is None else f"{pct:+.2f}%"
            sig_str  = "—" if is_bl else ("✱ Yes" if s["significant"] else "No")
            preview_rows.append({
                "Parameter":           p_name,
                "Timepoint":           TP_DISPLAY.get(tp, tp),
                "n":                   s["n"],
                "Mean":                f"{s['mean']:.4f}",
                "% Change from BL":    pct_str,
                "Significant?":        sig_str,
            })

    if preview_rows:
        st.dataframe(
            pd.DataFrame(preview_rows),
            hide_index=True,
            use_container_width=True,
            height=min(500, 38 + 35 * len(preview_rows)),
        )

    st.markdown("**Improvement direction**")
    dir_mode_m = st.radio(
        "Direction setting",
        options=[
            "Auto-detected",
            "Decrease = Improvement",
            "Increase = Improvement",
            "Per parameter",
        ],
        horizontal=True,
        key="me_dir_mode",
    )

    if dir_mode_m == "Auto-detected":
        imp_dirs_m = dirs_m
        st.caption(
            "Auto-detected: "
            + ", ".join(
                f"**{k}** → {'↓' if v == 'lower' else '↑'}"
                for k, v in dirs_m.items()
            )
        )
    elif dir_mode_m == "Decrease = Improvement":
        imp_dirs_m = {k: "lower" for k in keep_m}
    elif dir_mode_m == "Increase = Improvement":
        imp_dirs_m = {k: "higher" for k in keep_m}
    else:
        imp_dirs_m = {}
        dcols = st.columns(min(len(keep_m), 3))
        for i, k in enumerate(keep_m):
            with dcols[i % len(dcols)]:
                ch = st.radio(
                    f"**{k}**",
                    ["Decrease = Improvement", "Increase = Improvement"],
                    index=0 if dirs_m.get(k, "lower") == "lower" else 1,
                    key=f"me_dir_{k}",
                )
                imp_dirs_m[k] = "lower" if "Decrease" in ch else "higher"

    chart_titles_m = {k: f"{study_ref} — {k}" for k in keep_m}

    st.divider()
    st.subheader("Phase 4 — Preview")

    preview_m = st.selectbox("Preview parameter", options=keep_m, key="me_preview")
    if preview_m:
        param_stats_m = stats_m.get(preview_m, {})
        if param_stats_m:
            fig_m = create_parameter_chart(
                ecrf_m,
                ecrf_m.parameters[preview_m],
                param_stats_m,
                imp_dirs_m.get(preview_m, "lower"),
                active_tps=active_tps_m,
                custom_title=chart_titles_m.get(preview_m),
                show_center=show_center,
            )
            if fig_m:
                st.pyplot(fig_m, use_container_width=True)
                plt.close(fig_m)
            else:
                st.warning(
                    f"Chart returned None for **{preview_m}**. "
                    "Check that at least one non-baseline timepoint has a mean value."
                )
        else:
            st.info("No stats found for this parameter.")

    st.divider()
    st.subheader("Phase 5 — Generate PDF")
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
# ═══════════════════════════════════════════════════════════════
# MONADERM STREAMLIT FLOW
# ═══════════════════════════════════════════════════════════════

def run_monaderm_flow(file_bytes: bytes, file_name: str):
    # ── Persistent state keys ────────────────────────────────────────────────
    for k in ("mn_file_name", "mn_scan", "mn_rep_df",
              "mn_included_reps", "mn_dropped_subjects",
              "mn_stats_df", "mn_stats_edited",
              "mn_ecrf", "mn_param_stats", "mn_auto_dirs",
              "mn_param_renames", "mn_preview_idx"):
        if k not in st.session_state:
            st.session_state[k] = None
    if st.session_state["mn_dropped_subjects"] is None:
        st.session_state["mn_dropped_subjects"] = []
    if st.session_state["mn_preview_idx"] is None:
        st.session_state["mn_preview_idx"] = 0

    # Reset state on new file
    if file_name != st.session_state["mn_file_name"]:
        for k in ("mn_scan", "mn_rep_df", "mn_included_reps",
                  "mn_stats_df", "mn_stats_edited",
                  "mn_ecrf", "mn_param_stats", "mn_auto_dirs",
                  "mn_param_renames"):
            st.session_state[k] = None
        st.session_state["mn_dropped_subjects"] = []
        st.session_state["mn_preview_idx"]      = 0
        st.session_state["mn_file_name"]        = file_name

    # ── Reset button ─────────────────────────────────────────────────────────
    if st.button("↺ Reset", key="mn_reset", type="secondary"):
        for k in ("mn_scan", "mn_rep_df", "mn_included_reps",
                  "mn_stats_df", "mn_stats_edited",
                  "mn_ecrf", "mn_param_stats", "mn_auto_dirs",
                  "mn_param_renames", "mn_file_name"):
            st.session_state[k] = None
        st.session_state["mn_dropped_subjects"] = []
        st.session_state["mn_preview_idx"]      = 0
        st.rerun()

    # ── Sheet detection ───────────────────────────────────────────────────────
    xls        = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet_opts = xls.sheet_names
    auto_sheet = _detect_monaderm_sheet(io.BytesIO(file_bytes))
    if auto_sheet:
        raw_sheet = auto_sheet
    else:
        raw_sheet = st.selectbox("RAW DATA sheet", sheet_opts, key="mn_sheet")

    # ── Study config ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    study_ref   = col_a.text_input(
        "Study reference",
        placeholder="e.g. CS251008 (defaults to filename if blank)",
        key="mn_ref",
    )
    show_center = col_b.checkbox("Show center on charts", value=False, key="mn_center")

    # ── Subject Management ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Subject Management")

    excl_raw = st.text_input(
        "Manually exclude Subject IDs (comma-separated)",
        placeholder="e.g. 0012, 0034",
        key="mn_excl",
    )
    global_excl = [
        (f"{int(e):04d}" if e.isdigit() else e)
        for e in (x.strip() for x in excl_raw.split(",")) if e
    ]

    # Subject completeness expander — populates after data is loaded
    rep_df_for_completeness: pd.DataFrame | None = st.session_state["mn_rep_df"]
    if rep_df_for_completeness is not None:
        included_reps_state = st.session_state["mn_included_reps"] or {}
        all_tps_in_data  = sorted(rep_df_for_completeness["KINETIC"].unique().tolist(), key=tp_sort_key)
        all_subjects_c   = sorted(rep_df_for_completeness["SUBJECT"].unique().tolist())
        all_zones_c      = sorted(rep_df_for_completeness["ZONE"].unique().tolist())
        all_params_c     = sorted(rep_df_for_completeness["PARAMETER"].unique().tolist())

        def subject_has_data(subj, zone, tp, param) -> bool:
            key4    = (subj, zone, tp, param)
            allowed = included_reps_state.get(key4)
            grp = rep_df_for_completeness[
                (rep_df_for_completeness["SUBJECT"]   == subj) &
                (rep_df_for_completeness["ZONE"]      == zone) &
                (rep_df_for_completeness["KINETIC"]   == tp)   &
                (rep_df_for_completeness["PARAMETER"] == param)
            ]
            if grp.empty:
                return False
            if allowed is not None:
                return len(grp[grp["REPETITION"].isin(allowed)]) > 0
            return True

        flagged_info: list[dict] = []
        for subj in all_subjects_c:
            tps_with_any_data = {
                tp for tp in all_tps_in_data
                if any(subject_has_data(subj, z, tp, p)
                       for z in all_zones_c for p in all_params_c)
            }
            tps_fully_absent = set(all_tps_in_data) - tps_with_any_data
            if tps_with_any_data and tps_fully_absent:
                flagged_info.append({
                    "Subject":     subj,
                    "Missing at":  ", ".join(TP_DISPLAY.get(t, t) for t in
                                   sorted(tps_fully_absent, key=tp_sort_key)),
                    "Has data at": ", ".join(TP_DISPLAY.get(t, t) for t in
                                   sorted(tps_with_any_data, key=tp_sort_key)),
                })

        badge = f"⚠️ {len(flagged_info)} subject(s) with missing timepoints" \
                if flagged_info else "✅ All subjects complete"
        with st.expander(f"Incomplete Subjects — {badge}", expanded=False):
            if not flagged_info:
                st.success("All subjects have data at every selected timepoint.")
            else:
                st.dataframe(pd.DataFrame(flagged_info), hide_index=True,
                             use_container_width=True)
                flagged_ids = [r["Subject"] for r in flagged_info]
                valid_prev  = [s for s in st.session_state["mn_dropped_subjects"]
                               if s in flagged_ids]
                dropped = st.multiselect(
                    "Exclude from analysis",
                    options=flagged_ids, default=valid_prev,
                    key="mn_drop_subjects_select",
                )
                if dropped != st.session_state["mn_dropped_subjects"]:
                    st.session_state["mn_dropped_subjects"] = dropped
                    for k in ("mn_stats_df", "mn_stats_edited", "mn_ecrf"):
                        st.session_state[k] = None
    else:
        st.caption("Incomplete subject detection will appear here after data is loaded.")

    # ════════════════════════════════════════════════════════════
    # STEP 1 — Scan & select parameters / timepoints
    # ════════════════════════════════════════════════════════════
    st.divider()
    st.header("Step 1 — Select Parameters & Timepoints")

    # Auto-scan
    if st.session_state["mn_scan"] is None:
        with st.spinner("Scanning file…"):
            st.session_state["mn_scan"] = _scan_monaderm_file(file_bytes, raw_sheet)
            for k in ("mn_rep_df", "mn_included_reps", "mn_stats_df",
                      "mn_stats_edited", "mn_ecrf"):
                st.session_state[k] = None

    scan: dict | None = st.session_state["mn_scan"]

    if scan:
        if scan["error"]:
            st.error(scan["error"])
            st.stop()
        if scan["probe_names"]:
            st.caption("Probe(s): " + ", ".join(scan["probe_names"]))
        if scan["zones"] not in ([], ["S1"]):
            st.caption("Zone(s): " + ", ".join(scan["zones"]))

        sel_params = st.multiselect(
            f"Parameters ({len(scan['params'])} found)",
            scan["params"], default=scan["params"], key="mn_sel_params",
        )
        mapped_tps = [t for t in scan["tps"]
                      if t.upper() in {x.upper() for x in KNOWN_TP_ORDER}]
        sel_tps = st.multiselect(
            f"Timepoints ({len(scan['tps'])} found)",
            scan["tps"],
            default=mapped_tps or scan["tps"],
            format_func=lambda t: f"{TP_DISPLAY.get(t, t)} ({t})",
            key="mn_sel_tps",
        )
    else:
        sel_params, sel_tps = [], []

    load_disabled = not (scan and not scan["error"] and sel_params and sel_tps)
    if st.button("📥 Load Data", type="primary", disabled=load_disabled, key="mn_load"):
        with st.spinner("Loading rep-level data…"):
            rep_df, err = _load_rep_df(
                file_bytes, raw_sheet, sel_params, sel_tps,
                global_excl or None,
            )
        if err:
            st.error(err)
        else:
            st.session_state["mn_rep_df"]       = rep_df
            st.session_state["mn_included_reps"] = {}
            st.session_state["mn_preview_idx"]   = 0
            for k in ("mn_stats_df", "mn_stats_edited", "mn_ecrf"):
                st.session_state[k] = None
            st.rerun()

    rep_df: pd.DataFrame | None = st.session_state["mn_rep_df"]
    if rep_df is None:
        st.stop()

    st.success(
        f"Loaded **{len(rep_df)}** raw measurements · "
        f"**{rep_df['SUBJECT'].nunique()}** subjects · "
        f"**{rep_df['PARAMETER'].nunique()}** parameter(s) · "
        f"**{rep_df['KINETIC'].nunique()}** timepoint(s)"
    )

    # ════════════════════════════════════════════════════════════
    # STEP 2 — Rep review (collapsed)
    # ════════════════════════════════════════════════════════════
    st.divider()
    all_params_loaded = sorted(rep_df["PARAMETER"].unique().tolist())
    all_tps_loaded    = sorted(rep_df["KINETIC"].unique().tolist(), key=tp_sort_key)

    n_excl_reps = sum(
        1 for v in (st.session_state["mn_included_reps"] or {}).values()
        if isinstance(v, set) and len(v) == 0
    )
    rep_badge = f" — {n_excl_reps} rep(s) excluded" if n_excl_reps else ""

    with st.expander(f"Step 2 — Review Repetitions{rep_badge}", expanded=False):
        st.caption(
            "Uncheck any rep to exclude it from the average for that "
            "subject × zone × timepoint × parameter combination."
        )
        sel_param_view = st.selectbox(
            "Parameter", all_params_loaded, key="mn_param_view"
        )
        sel_tp_view = st.selectbox(
            "Timepoint", all_tps_loaded,
            format_func=lambda t: f"{TP_DISPLAY.get(t, t)} ({t})",
            key="mn_tp_view",
        )
        view_df = rep_df[
            (rep_df["PARAMETER"] == sel_param_view) &
            (rep_df["KINETIC"]   == sel_tp_view)
        ].sort_values(["SUBJECT", "ZONE", "REPETITION"])

        if view_df.empty:
            st.info("No data for this combination.")
        else:
            included_reps: dict = st.session_state["mn_included_reps"] or {}
            subjects_in_view    = sorted(view_df["SUBJECT"].unique())
            zones_in_view       = sorted(view_df["ZONE"].unique())
            changed = False

            for zone in zones_in_view:
                if len(zones_in_view) > 1:
                    st.markdown(f"**Zone: {zone}**")
                max_rep     = int(view_df[view_df["ZONE"] == zone]["REPETITION"].max())
                cols_header = st.columns([1.5] + [1] * max_rep)
                cols_header[0].markdown("**Subject**")
                for r in range(1, max_rep + 1):
                    cols_header[r].markdown(f"**Rep {r}**")

                for subj in subjects_in_view:
                    subj_rows = view_df[
                        (view_df["SUBJECT"] == subj) & (view_df["ZONE"] == zone)
                    ]
                    if subj_rows.empty:
                        continue
                    key4             = (subj, zone, sel_tp_view, sel_param_view)
                    all_reps_for_key = sorted(subj_rows["REPETITION"].tolist())
                    current_set      = included_reps.get(key4, set(all_reps_for_key))
                    cols_row         = st.columns([1.5] + [1] * max_rep)
                    cols_row[0].write(subj)
                    new_set = set()
                    for r in all_reps_for_key:
                        val_rows = subj_rows[subj_rows["REPETITION"] == r]["VALUE"]
                        val_str  = f"{val_rows.values[0]:.4f}" if len(val_rows) else "—"
                        if r <= max_rep:
                            new_checked = cols_row[r].checkbox(
                                val_str, value=(r in current_set),
                                key=f"mn_rep_{subj}_{zone}_{sel_tp_view}_{sel_param_view}_{r}",
                            )
                            if new_checked:
                                new_set.add(r)
                            if new_checked != (r in current_set):
                                changed = True
                    if new_set != current_set:
                        included_reps[key4] = new_set
                        changed = True

            if changed:
                st.session_state["mn_included_reps"] = included_reps
                for k in ("mn_stats_df", "mn_stats_edited", "mn_ecrf"):
                    st.session_state[k] = None

    # ════════════════════════════════════════════════════════════
    # STEP 3 — Statistical Summary
    # ════════════════════════════════════════════════════════════
    st.divider()
    st.header("Step 3 — Statistical Summary")
    st.caption(
        "Review and edit any values below before generating charts. "
        "Mean values drive bar heights; % Change and significance drive pill badges and highlights."
    )

    # Baseline picker lives here — directly above recompute
    bl_options  = all_tps_loaded
    bl_def      = bl_options.index("BL") if "BL" in bl_options else 0
    baseline_tp = st.selectbox(
        "Baseline timepoint", bl_options, index=bl_def,
        format_func=lambda t: f"{TP_DISPLAY.get(t, t)} ({t})",
        key="mn_baseline",
    )

    # Auto-compute on first load
    if st.session_state["mn_stats_edited"] is None:
        with st.spinner("Computing statistics…"):
            raw_stats = _compute_stats_from_rep_df(
                rep_df,
                st.session_state["mn_included_reps"] or {},
                baseline_tp,
                dropped_subjects=st.session_state.get("mn_dropped_subjects") or [],
            )
        st.session_state["mn_stats_df"]     = raw_stats
        st.session_state["mn_stats_edited"] = raw_stats.copy()
        st.session_state["mn_ecrf"]         = None

    if st.button("⚙️ Recompute Statistics", type="secondary", key="mn_compute"):
        with st.spinner("Computing…"):
            raw_stats = _compute_stats_from_rep_df(
                rep_df,
                st.session_state["mn_included_reps"] or {},
                baseline_tp,
                dropped_subjects=st.session_state.get("mn_dropped_subjects") or [],
            )
        st.session_state["mn_stats_df"]     = raw_stats
        st.session_state["mn_stats_edited"] = raw_stats.copy()
        st.session_state["mn_ecrf"]         = None
        st.rerun()

    stats_edited: pd.DataFrame | None = st.session_state["mn_stats_edited"]
    if stats_edited is None:
        st.stop()

    DISPLAY_COLS = ["Assessment", "Time Point", "n", "Mean", "p-value",
                    "Mean % Change From Baseline", "Significant"]
    display_df   = stats_edited[DISPLAY_COLS].copy()

    edited_display = st.data_editor(
        display_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Assessment":                  st.column_config.TextColumn(disabled=True),
            "Time Point":                  st.column_config.TextColumn(disabled=True),
            "n":                           st.column_config.NumberColumn(min_value=0, step=1),
            "Mean":                        st.column_config.NumberColumn(format="%.4f"),
            "p-value":                     st.column_config.TextColumn(),
            "Mean % Change From Baseline": st.column_config.TextColumn(),
            "Significant": st.column_config.SelectboxColumn(
                options=["Significant", "Not significant", "—"], required=False
            ),
        },
        key="mn_table_editor",
    )

    merged = stats_edited.copy()
    for col in DISPLAY_COLS:
        if col in edited_display.columns:
            merged[col] = edited_display[col].values

    def _parse_mean(cell) -> float:
        try:    return float(str(cell).replace(",", "").strip())
        except: return float("nan")

    merged["_mean_float"] = merged["Mean"].apply(_parse_mean)
    if not merged.equals(st.session_state["mn_stats_edited"]):
        st.session_state["mn_stats_edited"] = merged
        st.session_state["mn_ecrf"]         = None

    # ════════════════════════════════════════════════════════════
    # STEP 4 — Charts
    # ════════════════════════════════════════════════════════════
    st.divider()
    st.header("Step 4 — Charts")

    # Build ECRFData lazily
    if st.session_state["mn_ecrf"] is None and st.session_state["mn_stats_edited"] is not None:
        bl_display = TP_DISPLAY.get(baseline_tp, baseline_tp)
        ecrf_mn, pstats_mn, adirs_mn = _stats_df_to_ecrf_and_stats(
            st.session_state["mn_stats_edited"],
            study_ref or Path(file_name).stem,
            bl_display,
        )
        st.session_state["mn_ecrf"]        = ecrf_mn
        st.session_state["mn_param_stats"] = pstats_mn
        st.session_state["mn_auto_dirs"]   = adirs_mn

    ecrf_mn   = st.session_state["mn_ecrf"]
    pstats_mn = st.session_state["mn_param_stats"]
    adirs_mn  = st.session_state["mn_auto_dirs"]

    if ecrf_mn is None:
        st.info("Compute statistics first.")
        st.stop()

    keep = list(ecrf_mn.parameters.keys())
    ref  = study_ref or Path(file_name).stem

    # Restore saved renames
    if st.session_state["mn_param_renames"] is None:
        st.session_state["mn_param_renames"] = {b: b for b in keep}
    param_renames: dict = st.session_state["mn_param_renames"]
    # Add any new keys that weren't there before
    for b in keep:
        if b not in param_renames:
            param_renames[b] = b

    # ── Chart Settings expander ───────────────────────────────────────────────
    with st.expander("⚙️ Chart Settings", expanded=False):
        n_cols   = min(len(keep), 3)
        r_cols   = st.columns(n_cols)
        dir_cols = st.columns(n_cols)

        st.markdown("**Rename parameters**")
        rename_cols = st.columns(n_cols)
        new_renames = {}
        for i, base in enumerate(keep):
            with rename_cols[i % n_cols]:
                val = st.text_input(
                    base,
                    value=param_renames.get(base, base),
                    key=f"mn_rename_{base}",
                    placeholder="e.g. Corneometer",
                )
                new_renames[base] = val.strip() or base

        if new_renames != param_renames:
            st.session_state["mn_param_renames"] = new_renames
            param_renames = new_renames
            st.session_state["mn_ecrf"] = None  # force title rebuild

        st.markdown("**Improvement direction**")
        dir_mode = st.radio(
            "Direction",
            ["Accept all auto-detected", "All same direction", "Per parameter"],
            horizontal=True, key="mn_dir_mode", label_visibility="collapsed",
        )
        imp_dirs: dict = {}
        if dir_mode == "Accept all auto-detected":
            imp_dirs = dict(adirs_mn)
        elif dir_mode == "All same direction":
            d = st.radio(
                "Dir:", ["Decrease = Improvement", "Increase = Improvement"],
                horizontal=True, key="mn_unified_dir",
            )
            imp_dirs = {k: ("lower" if "Decrease" in d else "higher") for k in keep}
        else:
            dcols = st.columns(n_cols)
            for i, base in enumerate(keep):
                with dcols[i % n_cols]:
                    ch = st.radio(
                        f"**{param_renames.get(base, base)}**",
                        ["Decrease = Improvement", "Increase = Improvement"],
                        index=0 if adirs_mn.get(base, "lower") == "lower" else 1,
                        key=f"mn_dir_{base}",
                    )
                    imp_dirs[base] = "lower" if "Decrease" in ch else "higher"

        st.markdown("**Chart titles**")
        chart_titles = {}
        title_cols   = st.columns(n_cols)
        for i, base in enumerate(keep):
            with title_cols[i % n_cols]:
                label = param_renames.get(base, base)
                chart_titles[base] = st.text_input(
                    param_renames.get(base, base),
                    value=f"{ref} — {label}",
                    key=f"mn_title_{base}",
                )

    if not chart_titles:
        chart_titles = {b: f"{ref} — {param_renames.get(b, b)}" for b in keep}

    # Apply renames to display names
    for base in keep:
        ecrf_mn.parameters[base].display_name = param_renames.get(base, base)

    # ── Preview with prev/next navigation ────────────────────────────────────
    idx     = st.session_state["mn_preview_idx"]
    idx     = max(0, min(idx, len(keep) - 1))
    preview_p = keep[idx]

    if len(keep) > 1:
        nav_l, nav_mid, nav_r = st.columns([1, 4, 1])
        if nav_l.button("◀", key="mn_prev"):
            st.session_state["mn_preview_idx"] = (idx - 1) % len(keep)
            st.rerun()
        nav_mid.markdown(
            f"<div style='text-align:center; padding-top:6px'>"
            f"<b>{param_renames.get(preview_p, preview_p)}</b> "
            f"<span style='color:gray'>({idx + 1} of {len(keep)})</span></div>",
            unsafe_allow_html=True,
        )
        if nav_r.button("▶", key="mn_next"):
            st.session_state["mn_preview_idx"] = (idx + 1) % len(keep)
            st.rerun()
    else:
        st.markdown(f"**{param_renames.get(preview_p, preview_p)}**")

    active_tps_mn = sorted(
        [r["Time Point"] for _, r in st.session_state["mn_stats_edited"].iterrows()
         if r["Assessment"] == preview_p],
        key=lambda t: tp_sort_key(next((k for k, v in TP_DISPLAY.items() if v == t), t)),
    )
    fig = create_parameter_chart(
        ecrf_mn, ecrf_mn.parameters[preview_p],
        pstats_mn.get(preview_p, {}),
        imp_dirs.get(preview_p, "lower"),
        active_tps=active_tps_mn,
        custom_title=chart_titles.get(preview_p),
        show_center=show_center,
    )
    if fig:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.warning("No chart data — check that at least one non-baseline timepoint has a mean.")

    # ════════════════════════════════════════════════════════════
    # STEP 5 — Generate PDF
    # ════════════════════════════════════════════════════════════
    st.divider()
    st.header("Step 5 — Generate PDF")
    st.write(f"**{len(keep)}** parameter(s) · **{ecrf_mn.n_included}** subjects")

    if st.button("📄 Generate PDF", type="primary", key="mn_pdf"):
            all_active_tps = sorted(
                st.session_state["mn_stats_edited"]["Time Point"].unique().tolist(),
                key=lambda t: tp_sort_key(
                    next((k for k, v in TP_DISPLAY.items() if v == t), t)
                ),
            )
            progress  = st.progress(0, text="Starting…")
            pdf_bytes = generate_pdf_bytes(
                ecrf_mn, pstats_mn, imp_dirs, chart_titles,
                active_tps=all_active_tps,
                show_center=show_center,
                progress_bar=progress,
            )
            progress.empty()
            st.success(f"✅ PDF generated — {len(keep)} chart(s).")
            st.download_button(
                "⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"{Path(file_name).stem}_Monaderm_Charts.pdf",
                mime="application/pdf",
                key="mn_pdf_dl",
            )
    
def run_excel_flow(file_bytes: bytes = None, file_name: str = None):
    st.header("Step 1 — Configure")

    if file_name != st.session_state.uploaded_file_name:
        for k in ("ecrf", "all_param_stats", "auto_dirs"):
            st.session_state[k] = None
        st.session_state.uploaded_file_name = file_name

    sheets     = find_ecrf_sheets(io.BytesIO(file_bytes))
    ecrf_sheet = st.selectbox("EDC data sheet", options=sheets, index=0)

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

    if st.button("🔍 Parse EDC Data", type="primary"):
        with st.spinner("Parsing EDC data…"):
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
                "These timepoints exist in the EDC but are not in the standard list. "
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
            "Auto Dir": "Decrease = Improvement" if auto_dirs.get(base, "lower") == "lower" else "Increase = Improvement",
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
                improvement_dirs[base] = "lower" if "Decrease" in choice else "higher"

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
            stem      = Path(file_name).stem
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

        stem = Path(file_name).stem
        st.success(f"✅ PDF generated — {len(keep)} chart(s).")
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"{stem}_EDC_Charts.pdf",
            mime="application/pdf")

def main():
    st.set_page_config(
        page_title="EDC Data Visualizations Generator",
        page_icon="📊", layout="wide",
    )
    st.title("📊 EDC Data Visualizations Generator v2.0")
    st.caption("Generates mean-change-from-baseline charts for eCRF and Monaderm datasets.")

    for key in (
        "ecrf", "all_param_stats", "auto_dirs", "uploaded_file_name",
        "mn_file_name", "mn_scan", "mn_rep_df", "mn_included_reps",
        "mn_stats_df", "mn_stats_edited", "mn_ecrf",
        "mn_param_stats", "mn_auto_dirs",
    ):
        if key not in st.session_state:
            st.session_state[key] = None

    mode = st.radio(
        "Mode",
        ["Upload File", "Manual Entry"],
        horizontal=True,
    )

    if mode == "Manual Entry":
        run_manual_entry_flow()
        return

    # ── Upload & auto-detect ─────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload workbook — eCRF (.xlsx/.xls) or Monaderm RAW DATA (.xlsx/.xls/.xlsm)",
        type=["xlsx", "xls", "xlsm"],
        key="main_uploader",
    )

    if uploaded is None:
        st.info("Upload a file to begin. The format will be detected automatically.")
        st.stop()

    file_bytes  = uploaded.read()
    file_format = _detect_file_format(file_bytes)

    if file_format == "monaderm":
        st.success("📡 Monaderm RAW DATA format detected.")
        run_monaderm_flow(file_bytes=file_bytes, file_name=uploaded.name)
    else:
        st.success("📋 eCRF format detected.")
        run_excel_flow(file_bytes=file_bytes, file_name=uploaded.name)


if __name__ == "__main__":
    main()