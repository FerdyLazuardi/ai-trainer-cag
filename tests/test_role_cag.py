import pytest
from app.graph.pipeline import resolve_user_role, _filter_kb_by_role

def test_resolve_user_role():
    # HO Location
    assert resolve_user_role({"location": "HO", "position": "Operations Senior Analyst"}) == "HO"
    assert resolve_user_role({"location": "ho", "position": "any"}) == "HO"
    
    # FO Location with BP keywords
    assert resolve_user_role({"location": "FO", "position": "Business Partner"}) == "BP"
    assert resolve_user_role({"location": "FO", "position": "Business Partner OJT"}) == "BP"
    assert resolve_user_role({"location": "FO", "position": "BP - Middle"}) == "BP"
    assert resolve_user_role({"location": "FO", "position": "Business Partner Recovery"}) == "BP"
    
    # FO Location with BM keywords
    assert resolve_user_role({"location": "FO", "position": "Business Manager"}) == "BM"
    assert resolve_user_role({"location": "FO", "position": "Business Manager OJT"}) == "BM"
    assert resolve_user_role({"location": "FO", "position": "Business Manager PJS"}) == "BM"
    assert resolve_user_role({"location": "FO", "position": "BM"}) == "BM"

    # FO Location with RM keywords
    assert resolve_user_role({"location": "FO", "position": "Regional Manager"}) == "RM"
    assert resolve_user_role({"location": "FO", "position": "RM"}) == "RM"

    # FO Location with AM keywords
    assert resolve_user_role({"location": "FO", "position": "Area Manager"}) == "AM"
    assert resolve_user_role({"location": "FO", "position": "AM"}) == "AM"

    # FO Location with HMB keywords
    assert resolve_user_role({"location": "FO", "position": "Hub Manager Bisnis"}) == "HMB"
    assert resolve_user_role({"location": "FO", "position": "HMB"}) == "HMB"
    
    # Fallbacks
    assert resolve_user_role({"location": "FO", "position": "Security Guard"}) == "FO"
    assert resolve_user_role({"location": "Unknown", "position": "BP"}) == "BP"  # Position match takes precedence over non-FO loc
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
    
    # Filter for HO
    filtered_ho = _filter_kb_by_role(kb_text, "HO")
    assert "BP and BM content" not in filtered_ho
    assert "HO content" in filtered_ho
    assert "General content with no roles attribute" in filtered_ho
    
    # Filter for RM (should only get the general/no-roles one)
    filtered_rm = _filter_kb_by_role(kb_text, "RM")
    assert "BP and BM content" not in filtered_rm
    assert "HO content" not in filtered_rm
    assert "General content with no roles attribute" in filtered_rm
