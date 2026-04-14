from __future__ import annotations

import json
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import csv
from pathlib import Path


FIELD_ALIASES = {
    "name": "Name",
    "Name": "Name",
    "smiles": "SMILES",
    "SMILES": "SMILES",
    "degree of polymerization": "DP",
    "DP": "DP",
    "short length": "length_short",
    "length_short": "length_short",
    "num_chains": "Num_chain",
    "Num_chains": "Num_chain",
    "Num_chain": "Num_chain",
    "num_litfsi": "Num_salt",
    "Num_LiTFSI": "Num_salt",
    "Num_salt": "Num_salt",
    "left_cap": "left_cap",
    "right_cap": "right_cap",
}


def _canonical_field_name(name: str) -> str:
    clean = (name or "").strip()
    return FIELD_ALIASES.get(clean, FIELD_ALIASES.get(clean.lower(), clean))


def _to_number_or_str(value: object) -> object:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _col_letters_to_index(col_letters: str) -> int:
    idx = 0
    for ch in col_letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def read_sheet_rows_xlsx(path: Path, sheet_name: str = "PEMD") -> list[dict[int, object]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ns_rel = {"r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    rows: list[dict[int, object]] = []

    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sst.findall("a:si", ns):
                parts = [t.text or "" for t in si.findall(".//a:t", ns)]
                shared_strings.append("".join(parts))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {r.attrib["Id"]: r.attrib["Target"] for r in rels}

        target = None
        for sheet in workbook.findall("a:sheets/a:sheet", ns):
            if sheet.attrib.get("name") == sheet_name:
                rel_id = sheet.attrib.get(f"{{{ns_rel['r']}}}id")
                if rel_id:
                    target = rel_map.get(rel_id)
                break
        if target is None:
            first_sheet = workbook.find("a:sheets/a:sheet", ns)
            if first_sheet is None:
                raise ValueError(f"No sheet found in {path}")
            rel_id = first_sheet.attrib.get(f"{{{ns_rel['r']}}}id")
            if rel_id is None:
                raise ValueError(f"Cannot locate sheet relationship in {path}")
            target = rel_map.get(rel_id)
            if target is None:
                raise ValueError(f"Cannot resolve sheet relationship in {path}")

        if not target.startswith("xl/"):
            target = f"xl/{target}"

        worksheet = ET.fromstring(zf.read(target))
        cell_ref_re = re.compile(r"([A-Z]+)(\d+)")
        for row in worksheet.findall("a:sheetData/a:row", ns):
            row_data: dict[int, object] = {}
            for cell in row.findall("a:c", ns):
                ref = cell.attrib.get("r", "")
                match = cell_ref_re.match(ref)
                if match is None:
                    continue
                col_idx = _col_letters_to_index(match.group(1))
                ctype = cell.attrib.get("t")
                value = ""
                if ctype == "inlineStr":
                    tnode = cell.find("a:is/a:t", ns)
                    value = (tnode.text if tnode is not None else "") or ""
                else:
                    vnode = cell.find("a:v", ns)
                    if vnode is not None:
                        raw = vnode.text or ""
                        if ctype == "s":
                            try:
                                value = shared_strings[int(raw)]
                            except (ValueError, IndexError):
                                value = raw
                        else:
                            value = raw
                row_data[col_idx] = _to_number_or_str(value)
            rows.append(row_data)
    return rows


def _records_from_indexed_rows(rows: list[dict[int, object]]) -> list[dict[str, object]]:
    if not rows:
        raise ValueError("No rows found")

    header_row = None
    header_map: dict[int, str] = {}
    for row in rows:
        labels = {k: str(v).strip() for k, v in row.items() if str(v).strip()}
        if labels:
            header_row = row
            header_map = labels
            break
    if header_row is None:
        raise ValueError("No header row found")

    records: list[dict[str, object]] = []
    header_cols = sorted(header_map.keys())
    start_collect = False
    for row in rows:
        if row is header_row:
            start_collect = True
            continue
        if not start_collect:
            continue
        rec: dict[str, object] = {}
        non_empty = False
        for col in header_cols:
            key = header_map[col]
            val = row.get(col, "")
            if isinstance(val, str):
                val = val.strip()
            rec[key] = val
            if val != "":
                non_empty = True
        if non_empty:
            records.append(rec)
    return records


def read_sheet_rows_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        records: list[dict[str, object]] = []
        for row in reader:
            rec: dict[str, object] = {}
            non_empty = False
            for key, value in row.items():
                clean_key = _canonical_field_name(key or "")
                clean_val = _to_number_or_str(value)
                rec[clean_key] = clean_val
                if clean_val != "":
                    non_empty = True
            if non_empty:
                records.append(rec)
    return records


def load_table_records(path: Path, sheet_name: str = "PEMD") -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        records = read_sheet_rows_csv(path)
        if not records:
            raise ValueError(f"No rows found in {path}")
        return records

    rows = read_sheet_rows_xlsx(path, sheet_name=sheet_name)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return _records_from_indexed_rows(rows)


def sorted_polymer_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    parsed: list[tuple[int, dict[str, object]]] = []
    for rec in records:
        name = str(rec.get("Name", "")).strip()
        match = re.fullmatch(r"poly_(\d+)", name)
        if match:
            parsed.append((int(match.group(1)), rec))
    parsed.sort(key=lambda item: item[0])
    return [rec for _, rec in parsed]


def _coerce_int(value: object, default: int) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def _render_json(template: dict, record: dict[str, object], *, short_length_default: int = 4) -> dict:
    out = json.loads(json.dumps(template))
    poly = out["polymer"]
    poly["name"] = str(record["Name"]).strip()
    poly["repeating_unit"] = str(record["SMILES"]).strip()
    poly["left_cap"] = str(record.get("left_cap", "")).strip()
    poly["right_cap"] = str(record.get("right_cap", "")).strip()
    degree = _coerce_int(record.get("DP"), poly.get("length", [short_length_default, short_length_default])[1])
    short_length = _coerce_int(record.get("length_short"), short_length_default)
    poly["length"] = [short_length, degree]
    poly["numbers"] = _coerce_int(record.get("Num_chain"), poly.get("numbers", 1))

    litfsi = _coerce_int(record.get("Num_salt"), 0)
    if "Li_cation" in out:
        out["Li_cation"]["numbers"] = litfsi
    if "salt_anion" in out:
        out["salt_anion"]["numbers"] = litfsi
    return out


def _render_py(template_py: str, *, stage: str = "pack_cell") -> str:
    stage_flags = {
        "build_polymer": {
            "RUN_BUILD_POLYMER": True,
            "RUN_POLYMER_FORCEFIELD": False,
            "RUN_SMALL_MOLECULE_FF": False,
            "RUN_RELAX_CHAIN": False,
            "RUN_PACK_CELL": False,
            "RUN_BOX_MD": False,
        },
        "generate_forcefield": {
            "RUN_BUILD_POLYMER": True,
            "RUN_POLYMER_FORCEFIELD": True,
            "RUN_SMALL_MOLECULE_FF": True,
            "RUN_RELAX_CHAIN": False,
            "RUN_PACK_CELL": False,
            "RUN_BOX_MD": False,
        },
        "relax_chain": {
            "RUN_BUILD_POLYMER": True,
            "RUN_POLYMER_FORCEFIELD": True,
            "RUN_SMALL_MOLECULE_FF": True,
            "RUN_RELAX_CHAIN": True,
            "RUN_PACK_CELL": False,
            "RUN_BOX_MD": False,
        },
        "pack_cell": {
            "RUN_BUILD_POLYMER": True,
            "RUN_POLYMER_FORCEFIELD": True,
            "RUN_SMALL_MOLECULE_FF": True,
            "RUN_RELAX_CHAIN": False,
            "RUN_PACK_CELL": True,
            "RUN_BOX_MD": False,
        },
        "run_md": {
            "RUN_BUILD_POLYMER": True,
            "RUN_POLYMER_FORCEFIELD": True,
            "RUN_SMALL_MOLECULE_FF": True,
            "RUN_RELAX_CHAIN": False,
            "RUN_PACK_CELL": True,
            "RUN_BOX_MD": True,
        },
    }
    flags = stage_flags.get(stage)
    if flags is None:
        raise ValueError(f"Unsupported stage: {stage}")

    rendered = template_py
    for key, value in flags.items():
        rendered = re.sub(
            rf"^{key}\s*=\s*(True|False)(.*)$",
            rf"{key} = {'True' if value else 'False'}\2",
            rendered,
            flags=re.MULTILINE,
        )
    return rendered


@dataclass
class GeneratedProject:
    name: str
    root: Path
    md_json: Path
    md_py: Path


def generate_projects_from_table(
    *,
    xlsx_path: Path,
    out_base: Path,
    template_json: Path,
    template_py: Path,
    sheet_name: str = "PEMD",
    stage: str = "pack_cell",
    short_length_default: int = 4,
) -> list[GeneratedProject]:
    records = sorted_polymer_records(load_table_records(xlsx_path, sheet_name=sheet_name))
    template_data = json.loads(template_json.read_text(encoding="utf-8"))
    rendered_py = _render_py(template_py.read_text(encoding="utf-8"), stage=stage)

    out_base.mkdir(parents=True, exist_ok=True)
    generated: list[GeneratedProject] = []
    for record in records:
        name = str(record["Name"]).strip()
        project_dir = out_base / name
        project_dir.mkdir(parents=True, exist_ok=True)
        json_path = project_dir / "md.json"
        py_path = project_dir / "md.py"
        json_path.write_text(
            json.dumps(_render_json(template_data, record, short_length_default=short_length_default), ensure_ascii=True, indent=4) + "\n",
            encoding="utf-8",
        )
        py_path.write_text(rendered_py, encoding="utf-8")
        generated.append(GeneratedProject(name=name, root=project_dir, md_json=json_path, md_py=py_path))
    return generated
