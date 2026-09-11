import pytest
from app.graph.pipeline import _format_user_context_block


def test_format_user_context_block_includes_pulau():
    uctx = {
        "name": "Wina Nursyifa",
        "point": "Tutuyan",
        "area": "Minahasa",
        "regional": "Sulawesi Utara",
        "pulau": "Jawa 1",
        "dept": "",
    }
    block = _format_user_context_block(uctx)

    assert "- Name: Wina Nursyifa" in block
    assert "- Point: Tutuyan" in block
    assert "- Area: Minahasa" in block
    assert "- Regional: Sulawesi Utara" in block
    assert "- Pulau: Jawa 1" in block
    # Empty dept should be omitted
    assert "- Dept:" not in block
    # Pulau must be in Profile, not in other KPIs
    assert "[Metrik Performa & KPI Lainnya]" not in block


def test_format_user_context_block_case_deduplication():
    # If both lowercase and uppercase exist, standard_keys should deduplicate
    uctx = {
        "name": "Test User",
        "Pulau": "Sumatera",
        "pulau": "Sumatera",
        "Point": "Point A",
        "point": "Point A",
        "Total Skor KPI": "90%",
    }
    block = _format_user_context_block(uctx)

    assert block.count("- Pulau: Sumatera") == 1
    assert block.count("- Point: Point A") == 1
    assert "- Total Skor KPI: 90%" in block


def test_spreadsheet_overrides_moodle_user_context():
    # Simulate Moodle user profile
    moodle_user_context = {
        "name": "User A",
        "dept": "Operations",
        "location": "FO",
        "position": "Business Manager",
        "grade": "BM - 1",
        "point": "Moodle Point",
        "gender": "Male",
        "area": "Moodle Area",
        "regional": "Moodle Regional",
        "pulau": "",
    }

    # Simulate spreadsheet KPI data
    spreadsheet_data = {
        "Point": "Spreadsheet Point",
        "Area": "Spreadsheet Area",
        "Regional": "Spreadsheet Regional",
        "Pulau": "Jawa 1",
        "Total Skor KPI": "95%",
    }

    # Apply override logic
    user_context = dict(moodle_user_context)
    for k, v in spreadsheet_data.items():
        k_lower = str(k).lower().strip()
        val_str = str(v).strip() if v is not None else ""
        if val_str:
            if k_lower in ("point", "cabang"):
                user_context["point"] = val_str
            elif k_lower in ("area", "wilayah"):
                user_context["area"] = val_str
            elif k_lower in ("regional", "region"):
                user_context["regional"] = val_str
            elif k_lower in ("pulau", "island"):
                user_context["pulau"] = val_str

    assert user_context["point"] == "Spreadsheet Point"
    assert user_context["area"] == "Spreadsheet Area"
    assert user_context["regional"] == "Spreadsheet Regional"
    assert user_context["pulau"] == "Jawa 1"

    # Formatted block verification
    block = _format_user_context_block(user_context)
    assert "- Point: Spreadsheet Point" in block
    assert "- Area: Spreadsheet Area" in block
    assert "- Regional: Spreadsheet Regional" in block
    assert "- Pulau: Jawa 1" in block
    assert "Moodle Point" not in block

