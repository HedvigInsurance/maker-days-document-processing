"""Build the health-timeline page (Hope-like frame) from a record.json written by run_contract.py or the notebook.

Run:  .venv/bin/python build_page.py --record output/<contract_id>/record.json
Writes output/<contract_id>/health_timeline.html with the record embedded, ready to open in a browser or publish.

Records written before 20 Nov 2026 lack two blocks; files next to the record fill them in:
  - "claims"   (the contract's claims)                      <- claims.json
  - "contract" (master_inception, pet name/birth/species)   <- contract.json, whose claim_costs also fill each claim's "cost"
The page is self-contained: no server, no network; a "Load another record.json" button re-renders from a local file.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "page_template.html"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", required=True, help="path to record.json")
    p.add_argument("--out", help="output html path (default: health_timeline.html next to the record)")
    a = p.parse_args()
    record_path = Path(a.record)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if "claims" not in record:
        claims_path = record_path.with_name("claims.json")
        record["claims"] = json.loads(claims_path.read_text(encoding="utf-8")) if claims_path.exists() else []
        print(f"claims: {'from ' + str(claims_path) if claims_path.exists() else 'none (no claims list in the record, no claims.json)'}")
    if "contract" not in record:
        contract_path = record_path.with_name("contract.json")
        if contract_path.exists():
            side = json.loads(contract_path.read_text(encoding="utf-8"))
            record["contract"] = {"master_inception": side.get("master_inception"), "pet": side.get("pet", {})}
            for c in record["claims"]:
                if "cost" not in c and c["claim_id"] in side.get("claim_costs", {}):
                    c["cost"] = side["claim_costs"][c["claim_id"]]
            print(f"contract facts from {contract_path}")
        else:
            record["contract"] = {}
            print("no contract block in the record and no contract.json: the page shows no inception date or pet age")
    html = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(record, ensure_ascii=False).replace("</", "<\\/")   # keep the JSON from closing the script tag
    assert "__RECORD_JSON__" in html
    html = html.replace("__RECORD_JSON__", payload)
    out = Path(a.out) if a.out else record_path.with_name("health_timeline.html")
    out.write_text(html, encoding="utf-8")
    print(f"written {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
