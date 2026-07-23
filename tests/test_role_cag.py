import pytest
from app.graph.pipeline import resolve_user_role, _filter_kb_by_role

def test_resolve_user_role():
    # HO Location always gets HO role (full access)
    assert resolve_user_role({"location": "HO", "grade": "Staff"}) == "HO"
    assert resolve_user_role({"location": "ho", "grade": "any"}) == "HO"
    assert resolve_user_role({"location": "HO", "grade": "BP - 2"}) == "HO"
    
    # FO Location with BP grades (Field Officer role mapping)
    assert resolve_user_role({"location": "FO", "grade": "BP - Junior"}) == "BP"
    assert resolve_user_role({"location": "FO", "grade": "BP 2"}) == "BP"
    assert resolve_user_role({"location": "FO", "grade": "BP_Senior"}) == "BP"
    assert resolve_user_role({"location": "FO", "grade": "BP/Level3"}) == "BP"
    assert resolve_user_role({"location": "FO", "grade": "BP"}) == "BP"
    
    # FO Location with BM grades
    assert resolve_user_role({"location": "FO", "grade": "BM - 2"}) == "BM"
    assert resolve_user_role({"location": "FO", "grade": "BM"}) == "BM"

    # FO Location with RM grades
    assert resolve_user_role({"location": "FO", "grade": "RM - 1A"}) == "RM"
    assert resolve_user_role({"location": "FO", "grade": "RM"}) == "RM"

    # FO Location with AM grades
    assert resolve_user_role({"location": "FO", "grade": "AM - 2B"}) == "AM"
    assert resolve_user_role({"location": "FO", "grade": "AM"}) == "AM"

    # FO Location with HMB grades
    assert resolve_user_role({"location": "FO", "grade": "HMB - 1A"}) == "HMB"
    assert resolve_user_role({"location": "FO", "grade": "HMB"}) == "HMB"
    
    # Fallbacks
    assert resolve_user_role({"location": "FO", "grade": "Security - 1A"}) == "FO"
    assert resolve_user_role({"location": "Unknown", "grade": "BP - Junior"}) == "BP"  # Grade match takes precedence over non-FO loc
    assert resolve_user_role(None) == "ALL"
    assert resolve_user_role({}) == "ALL"


def test_filter_kb_by_role():
    kb_text = (
        '<knowledge_base version="sha256:abc123xyz">\n'
        '<kb_index>\n'
        '- [DOC-001] Course: 3 Section: A File: a.md\n'
        '- [DOC-002] Course: 3 Section: B File: b.md\n'
        '- [DOC-003] Course: 3 Section: C File: c.md\n'
        '</kb_index>\n'
        '<doc id="DOC-001" course="3" section="A" file="a.md" roles="BP,BM">\n'
        'BP and BM content\n'
        '</doc>\n'
        '<doc id="DOC-002" course="3" section="B" file="b.md" roles="HO">\n'
        'HO content\n'
        '</doc>\n'
        '<doc id="DOC-003" course="3" section="C" file="c.md">\n'
        'General content with no roles attribute\n'
        '</doc>\n'
        '</knowledge_base>'
    )
    
    # Filter for BP
    filtered_bp = _filter_kb_by_role(kb_text, "BP")
    assert "BP and BM content" in filtered_bp
    assert "HO content" not in filtered_bp
    assert "General content with no roles attribute" in filtered_bp
    assert '<kb_index>' in filtered_bp
    assert 'version="sha256:abc123xyz"' in filtered_bp
    assert 'DOC-001' in filtered_bp.split("</kb_index>")[0]
    assert 'DOC-002' not in filtered_bp.split("</kb_index>")[0]
    assert 'DOC-003' in filtered_bp.split("</kb_index>")[0]
    
    # Filter for HO
    filtered_ho = _filter_kb_by_role(kb_text, "HO")
    assert "BP and BM content" not in filtered_ho
    assert "HO content" in filtered_ho
    assert "General content with no roles attribute" in filtered_ho
    assert 'DOC-001' not in filtered_ho.split("</kb_index>")[0]
    assert 'DOC-002' in filtered_ho.split("</kb_index>")[0]
    assert 'DOC-003' in filtered_ho.split("</kb_index>")[0]
    
    # Filter for RM (should only get the general/no-roles one)
    filtered_rm = _filter_kb_by_role(kb_text, "RM")
    assert "BP and BM content" not in filtered_rm
    assert "HO content" not in filtered_rm
    assert "General content with no roles attribute" in filtered_rm
    assert 'DOC-001' not in filtered_rm.split("</kb_index>")[0]
    assert 'DOC-002' not in filtered_rm.split("</kb_index>")[0]
    assert 'DOC-003' in filtered_rm.split("</kb_index>")[0]


def test_filter_inner_role_blocks():
    kb_text = (
        '<doc id="DOC-001" roles="ALL">\n'
        'General info for all.\n'
        '<role_block roles="BP">\n'
        'Khusus BP: Limit 10jt\n'
        '</role_block>\n'
        '<role_block roles="BM">\n'
        'Khusus BM: Limit 50jt\n'
        '</role_block>\n'
        '<!-- role: BP,BM -->\n'
        'Khusus BP dan BM info\n'
        '<!-- /role -->\n'
        '</doc>'
    )

    filtered_bp = _filter_kb_by_role(kb_text, "BP")
    assert "General info for all." in filtered_bp
    assert "Khusus BP: Limit 10jt" in filtered_bp
    assert "Khusus BM: Limit 50jt" not in filtered_bp
    assert "Khusus BP dan BM info" in filtered_bp

    filtered_bm = _filter_kb_by_role(kb_text, "BM")
    assert "General info for all." in filtered_bm
    assert "Khusus BP: Limit 10jt" not in filtered_bm
    assert "Khusus BM: Limit 50jt" in filtered_bm
    assert "Khusus BP dan BM info" in filtered_bm

    filtered_rm = _filter_kb_by_role(kb_text, "RM")
    assert "General info for all." in filtered_rm
    assert "Khusus BP: Limit 10jt" not in filtered_rm
    assert "Khusus BM: Limit 50jt" not in filtered_rm
    assert "Khusus BP dan BM info" not in filtered_rm

