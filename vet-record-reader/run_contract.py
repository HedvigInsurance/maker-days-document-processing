"""Read every file on every closed claim of one pet contract and build the pet's health timeline.

Run:  .venv/bin/python run_contract.py                 (uses CONTRACT_ID below)
      .venv/bin/python run_contract.py --contract <id>  (another contract)

Asks for your AWS MFA code in the terminal when the cached read-only session has expired.

Stages (same as contract_journal_reader.ipynb):
  1. BigQuery: list the contract's claims and files; fetch the files from S3; mark duplicates and unreadable types.
  2. Gemini sorts each readable file into DOCUMENT_TYPES (several labels per file). Labels are cached in
     output/label_cache.csv so a file is never labelled twice.
  3. Gemini reads the JOURNAL_TYPES files in one call with Desiree's prompt and the claims' details.
  4. Cost per stage.

Writes output/<contract_id>/: contract_files.csv, record.json, visits.csv, gemini_calls.csv, summary.txt
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import boto3
import pandas as pd
from google import genai
from google.genai import types
from google.cloud import bigquery
from pydantic import BaseModel

# %%
BASE_DIR = Path(__file__).resolve().parent   # paths below are relative to this file, not to where you run it from

# =============================== Config: every input in one place ===============================
CONTRACT_ID = "2bd1ee7b-7f56-4cc4-8163-d761027049fd"       # the test contract (Emma)

# --- Google Cloud / Gemini ---
GEMINI_PROJECT = "hedvig-coruscant"
GEMINI_LOCATION = "eu"
GEMINI_MODEL = "gemini-3.8-flash"
TEMPERATURE = 0
LABELS = {"project": "vet-record-reader", "team": "maker-days-2026"}   # shown against the cost in Cloud Billing
MAX_INLINE_MB = 20            # Gemini's limit for files attached to one request; above it, one call per journal

# Prices for gemini-3.8-flash on the Vertex AI EU endpoint, USD per million tokens. Third-party listings
# (requesty.ai, sim.ai), read 18 Nov 2026; promotional rates until 31 Dec 2026, listed to double from 1 Jan 2027.
PRICE_PER_M_INPUT_USD = 0.83
PRICE_PER_M_OUTPUT_USD = 4.13        # thinking tokens are billed as output

# --- AWS (files are on S3 in the prod account) ---
AWS_REGION = "eu-central-1"
SOURCE_PROFILE = "ops"               # your own user, in ~/.aws
ROLE_ARN = "arn:aws:iam::658549670687:role/CrossAccountReadOnlyRole"
CREDENTIALS_CACHE = Path("/tmp/.vet-record-reader-aws-credentials.json")   # shared with the notebooks
SESSION_HOURS = 1                    # the role allows 1 hour today; 12 once #platform raises its maximum

# --- Folders ---
DATA_ROOT = BASE_DIR / "data" / "records"      # downloaded files, one folder per contract (data/ is ignored by git)
OUTPUT_ROOT = BASE_DIR / "output"              # results, one folder per contract (output/ is ignored by git)
LABEL_CACHE = OUTPUT_ROOT / "label_cache.csv"  # every label Gemini has given, reused across runs and contracts

# --- Stage 2: the document types Gemini may choose from (derived from the 100-file sample, 19 Nov) ---
DOCUMENT_TYPES = {
    "medical_record": "veterinary journal or patient history: what the vet observed, diagnosed and did during a visit",
    "discharge_instructions": "aftercare advice to the owner after a visit: medication schedule, diet, rest, follow-up",
    "invoice": "a bill listing treatments and the amount owed, including a clinic's billing specification or draft invoice",
    "receipt": "proof that payment was made: kvitto, card payment slip, pharmacy receipt",
    "direct_settlement_request": "the clinic's request that Hedvig pay it directly (direktreglering), or a claim form",
    "lab_or_imaging_report": "laboratory results, blood analyser readout, X-ray or other diagnostic image",
    "veterinary_certificate": "health or inspection certificate (besiktningsintyg), consent form, referral",
    "prescription_or_medication": "a prescription slip, or a photo of a medicine package with its label",
    "insurance_policy": "an insurance policy document",
    "pet_passport": "pages from a pet passport",
    "photo_of_animal": "a photograph of the pet itself, showing the injury or symptom",
    "other": "none of the above, for example a clinic logo image",
}
JOURNAL_TYPES = {"medical_record", "discharge_instructions", "lab_or_imaging_report", "veterinary_certificate"}   # go to stage 3

# A cached label is reused only if it was made with this same list of document types
LABELS_VERSION = hashlib.sha256(json.dumps(DOCUMENT_TYPES, sort_keys=True).encode()).hexdigest()[:8]

# File types Gemini can read as attachments; anything else is labelled unreadable and skipped
READABLE_CONTENT_TYPES = {"application/pdf", "image/jpeg", "image/png", "image/webp", "image/heic", "image/heif", "text/plain"}

# --- Stage 3: claim-type names in Desiree's prompt → Hedvig subclaim.type names (from the 18 Nov value list) ---
CLAIM_TYPE_TO_HEDVIG = {
    "Arthritis and Joint issues": "PET_ARTHRITIS_AND_JOINT_ISSUES",
    "Asthma": "PET_ASTHMA",
    "Behavior Therapy": "PET_BEHAVIOUR_THERAPY",
    "Cancer": "PET_CANCER",
    "Castration": "PET_CASTRATION",
    "Chronic gastrointestinal Issues": "PET_CHRONIC_GASTROINTESTINAL_ISSUES",
    "Claw Injury": "PET_CLAW_INJURY",
    "Cruciate Ligament Injuries": "PET_CRUCIATE_LIGAMENT_INJURIES",
    "Default": "PET_DEFAULT",
    "Dental Issues": "PET_DENTAL_ISSUES",
    "Diabetes": "PET_DIABETES",
    "Dystocia": "PET_DYSTOCIA",
    "Ear infections": "PET_EAR_INFECTIONS",
    "Endocrine Disorders": "PET_ENDOCRINE_DISORDERS",
    "Eye Problems": "PET_EYE_PROBLEMS",
    "Foreign Body Ingestion": "PET_FOREIGN_BODY_INGESTION",
    "Fractures and Broken Bones": "PET_FRACTURES_AND_BROKEN_BONES",
    "Heart Conditions": "PET_HEART_CONDITIONS",
    "Infections": "PET_INFECTIONS",
    "Intestinal Parasites": "PET_INTESTINAL_PARASITES",
    "Kidney and Liver disease": "PET_KIDNEY_AND_LIVER_DISEASE",
    "Lacerations and Bite Wounds": "PET_LACERATIONS_AND_BITE_WOUNDS",
    "Last Rest": "PET_LAST_REST",
    "Non Chronic gastrointestinal Issues": "PET_NON_CHRONIC_GASTROINTESTINAL_ISSUES",
    "Pancreatitis": "PET_PANCREATITIS",
    "Poisoning": "PET_POISONING",
    "Seizures and Neurological Disorders": "PET_SEIZURES_AND_NEUROLOGICAL_DISORDERS",
    "Skin Conditions and Allergies": "PET_SKIN_CONDITIONS_AND_ALLERGIES",
    "Tooth Fracture and Baby Teeth": "PET_TOOTH_FRACTURE_AND_BABY_TEETH",
    "Urinary Tract Infections": "PET_URINARY_TRACT_INFECTIONS",
    "Plastic Surgery Vet": None,   # CHECK: no PET_* value matches; SCARS_AND_PHYSICAL_ALTERATION exists in subclaim.type
}
# subclaim.type values with no name in the prompt: PET_LIFE

# %%
# ====================================== Helpers ======================================
bq_client = bigquery.Client(project="hedvig-coruscant")


def run_query(sql, max_gb=10):
    dry = bq_client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    gb = dry.total_bytes_processed / 1e9
    print(f"Dry run: this query scans {gb:.2f} GB")
    if gb > max_gb:
        raise ValueError(f"{gb:.2f} GB is above max_gb={max_gb}. Approve by raising max_gb.")
    return bq_client.query(sql).result().to_dataframe()


def assume_read_only_role():
    """Temporary keys as CrossAccountReadOnlyRole, valid SESSION_HOURS. Asks for your MFA code once."""
    ops = boto3.Session(profile_name=SOURCE_PROFILE)
    mfa_serial = ops._session.full_config["profiles"].get("prod", {}).get("mfa_serial")
    kwargs = {"RoleArn": ROLE_ARN, "RoleSessionName": "vet-record-reader", "DurationSeconds": SESSION_HOURS * 3600}
    if mfa_serial:
        kwargs |= {"SerialNumber": mfa_serial, "TokenCode": input("AWS MFA code: ")}
    c = ops.client("sts").assume_role(**kwargs)["Credentials"]
    CREDENTIALS_CACHE.write_text(json.dumps({
        "AccessKeyId": c["AccessKeyId"], "SecretAccessKey": c["SecretAccessKey"],
        "SessionToken": c["SessionToken"], "Expiration": c["Expiration"].timestamp(),
    }))
    print(f"AWS credentials valid until {c['Expiration']:%Y-%m-%d %H:%M %Z}")


def prod_session():
    if not CREDENTIALS_CACHE.exists() or json.loads(CREDENTIALS_CACHE.read_text())["Expiration"] < time.time() + 60:
        assume_read_only_role()
    c = json.loads(CREDENTIALS_CACHE.read_text())
    return boto3.Session(aws_access_key_id=c["AccessKeyId"], aws_secret_access_key=c["SecretAccessKey"],
                         aws_session_token=c["SessionToken"], region_name=AWS_REGION)


def fetch_file(bucket, key):
    """The file's bytes. Read-only: nothing in S3 is written, moved or deleted."""
    return prod_session().client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()


gemini = genai.Client(vertexai=True, project=GEMINI_PROJECT, location=GEMINI_LOCATION)
CALLS = []   # one row per Gemini call (or cache hit): stage, label, tokens, cost


def ask_gemini(files, prompt, stage, label, output_format=None, model=GEMINI_MODEL):
    """Send one or more files and a prompt to Gemini in one call.

    files: list of (file_label, bytes, content_type). Each file is preceded by a text part naming it,
           so the prompt can refer to files as file_1, file_2, ...
    output_format: a pydantic class the answer must fit, or None for free text.
    Returns (answer, usage). Falls back to JSON-in-the-prompt if Gemini rejects the forced shape.
    """
    parts = []
    for file_label, data, content_type in files:
        parts += [f"[{file_label}]", types.Part.from_bytes(data=data, mime_type=content_type)]
    config = types.GenerateContentConfig(temperature=TEMPERATURE, labels=LABELS)
    forced = output_format is not None
    if forced:
        config.response_mime_type = "application/json"
        config.response_schema = output_format
    try:
        response = gemini.models.generate_content(model=model, contents=parts + [prompt], config=config)
    except Exception as e:
        if not forced or "schema" not in str(e).lower():
            raise
        print(f"Gemini rejected the forced output shape ({str(e)[:120]}...). Asking for JSON in the prompt instead.")
        config.response_schema = None
        schema_text = json.dumps(output_format.model_json_schema())
        response = gemini.models.generate_content(
            model=model, contents=parts + [prompt + "\n\nAnswer with JSON only, matching this JSON schema:\n" + schema_text],
            config=config)
    u = response.usage_metadata
    usage = {"stage": stage, "label": label, "model": model, "cached": False,
             "input_tokens": u.prompt_token_count, "output_tokens": u.candidates_token_count,
             "thinking_tokens": u.thoughts_token_count or 0, "total_tokens": u.total_token_count}
    usage["cost_usd"] = round((usage["input_tokens"] * PRICE_PER_M_INPUT_USD
                               + (usage["output_tokens"] + usage["thinking_tokens"]) * PRICE_PER_M_OUTPUT_USD) / 1e6, 4)
    CALLS.append(usage)
    if forced:
        answer = response.parsed if response.parsed is not None else output_format.model_validate_json(response.text)
    else:
        answer = response.text
    return answer, usage


# ---------------------------------- Label cache (stage 2) ----------------------------------
LABEL_CACHE_COLUMNS = ["file_id", "claim_id", "sha256", "file_name", "document_types", "primary_type", "confidence",
                       "one_line_summary", "language", "model", "labels_version", "labelled_at"]


def load_label_cache():
    if LABEL_CACHE.exists():
        return pd.read_csv(LABEL_CACHE, dtype=str).fillna("")
    return pd.DataFrame(columns=LABEL_CACHE_COLUMNS)


def cached_label(cache, file_id, sha256):
    """The stored label for this file (by file id, else by content fingerprint), if made with the current DOCUMENT_TYPES."""
    current = cache[cache["labels_version"] == LABELS_VERSION]
    hit = current[current["file_id"] == str(file_id)]
    if hit.empty and sha256:
        hit = current[current["sha256"] == sha256]
    if hit.empty:
        return None
    row = hit.iloc[-1]
    return {"document_types": [t for t in row["document_types"].split(";") if t], "primary_type": row["primary_type"],
            "confidence": row["confidence"], "one_line_summary": row["one_line_summary"], "language": row["language"]}


def append_label_cache(row):
    LABEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=LABEL_CACHE_COLUMNS).to_csv(LABEL_CACHE, mode="a", header=not LABEL_CACHE.exists(), index=False)


# ===================================== Stage 1: gather =====================================
def stage1(contract_id):
    sql = f"""
WITH pet_claims AS (
SELECT
claim_id,
ANY_VALUE(contract_id) AS contract_id,
ANY_VALUE(promise_subcategory) AS promise_subcategory,
MIN(notification_date) AS notification_date,
MIN(date_of_loss) AS date_of_loss,
min(outcome) as claim_outcome,
ARRAY_AGG(description_of_loss IGNORE NULLS ORDER BY total_claim_cost DESC) AS claim_types_by_cost,

FROM hedvig-coruscant.dbt_prod.claims_reporting
WHERE promise_category = 'Pet'
AND acquisition_source != 'manypets'
AND claim_status = 'CLOSED'
AND outcome IN ('COMPENSATION_BELOW_DEDUCTIBLE', 'NOT_COVERED_BY_TERMS', 'PAID_OUT')
GROUP BY claim_id
)
SELECT
p.contract_id,
ANY_VALUE(p.promise_subcategory) AS pet_type,
COUNT(DISTINCT p.claim_id) AS nr_claims,
COUNT(*) AS nr_files,
ARRAY_AGG(
STRUCT(
f.id,
f.content_type,
f.uploaded_at,
p.claim_id,
p.notification_date AS claim_notification_date,
p.date_of_loss AS claim_date_of_loss,
p.claim_outcome,
p.claim_types_by_cost,
f.file_name,
f.category,
f.upload_source,
f.bucket,
f.key
)
ORDER BY f.uploaded_at, f.id
) AS files
FROM pet_claims p
JOIN hedvig-coruscant.dbt_prod.stg_claims_service__claim_file f
ON f.claims_id = p.claim_id
WHERE  f.deleted_at IS NULL   -- the populated deleted column (markd_as_deleted_at is never set; counts 19 Nov)
and contract_id = '{contract_id}'
GROUP BY p.contract_id
ORDER BY FARM_FINGERPRINT(p.contract_id)
"""
    contract = run_query(sql)
    assert len(contract) == 1, f"expected one contract row, got {len(contract)}"
    contract_row = contract.iloc[0]
    print(f"contract {contract_row['contract_id']}: pet_type={contract_row['pet_type']}, "
          f"{contract_row['nr_claims']} claims, {contract_row['nr_files']} files")
    contract_files = pd.DataFrame(list(contract_row["files"]))
    contract_files["claim_types_by_cost"] = contract_files["claim_types_by_cost"].apply(list)

    # Read every file from S3 once; duplicates (same bytes) are kept in the table but point at the first copy
    data_dir = DATA_ROOT / contract_id
    data_dir.mkdir(parents=True, exist_ok=True)
    file_bytes = {}          # file id -> bytes, only for the first copy of each distinct content
    hashes, status, duplicate_of = [], [], []
    seen = {}                # sha256 -> file id of the first copy
    for _, r in contract_files.iterrows():
        if r["content_type"] not in READABLE_CONTENT_TYPES:
            hashes.append(None); status.append("unreadable"); duplicate_of.append(None)
            continue
        data = fetch_file(r["bucket"], r["key"])
        h = hashlib.sha256(data).hexdigest()
        hashes.append(h)
        if h in seen:
            status.append("duplicate"); duplicate_of.append(seen[h])
        else:
            seen[h] = r["id"]; file_bytes[r["id"]] = data
            (data_dir / f"{r['id']}-{r['file_name']}").write_bytes(data)
            status.append("readable"); duplicate_of.append(None)
    contract_files["sha256"] = hashes
    contract_files["status"] = status
    contract_files["duplicate_of"] = duplicate_of
    contract_files["size_mb"] = contract_files["id"].map(lambda i: round(len(file_bytes[i]) / 1e6, 2) if i in file_bytes else None)
    print(contract_files["status"].value_counts().to_dict(), "| total readable MB:", round(contract_files["size_mb"].sum(), 1))
    return contract_row, contract_files, file_bytes


# ====================================== Stage 2: sort ======================================
DocumentType = Literal[tuple(DOCUMENT_TYPES)]


class FileType(BaseModel):
    document_types: list[DocumentType]   # every kind of content in the file
    primary_type: DocumentType           # the one that best describes the file as a whole
    confidence: Literal["high", "medium", "low"]
    one_line_summary: str
    language: str


SORT_PROMPT = (
    "You are sorting a file uploaded to a pet insurance claim. A file often holds several kinds of content "
    "(for example a clinic's direct settlement bundle has a journal, an invoice draft and discharge instructions). "
    "List every type present in the file, choosing only from these types:\n"
    + "\n".join(f"- {k}: {v}" for k, v in DOCUMENT_TYPES.items())
    + "\n\nThen give the one primary type that best describes the file as a whole, your confidence, a one-line "
    "summary of what the file contains (no owner personal data), and the language of the document. "
    "Use only what is in the file."
)


def stage2(contract_files, file_bytes):
    cache = load_label_cache()
    results = {}
    for _, r in contract_files[contract_files["status"] == "readable"].iterrows():
        hit = cached_label(cache, r["id"], r["sha256"])
        if hit:
            results[r["id"]] = {**hit, "cached": True}
            CALLS.append({"stage": "2 sort", "label": r["file_name"], "model": GEMINI_MODEL, "cached": True,
                          "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0, "cost_usd": 0.0})
            print(f"{r['file_name']}: {', '.join(hit['document_types'])} (primary {hit['primary_type']}) | cached")
            continue
        answer, usage = ask_gemini([("file", file_bytes[r["id"]], r["content_type"])], SORT_PROMPT,
                                   stage="2 sort", label=r["file_name"], output_format=FileType)
        results[r["id"]] = {**answer.model_dump(), "cached": False}
        append_label_cache({"file_id": r["id"], "claim_id": r["claim_id"], "sha256": r["sha256"], "file_name": r["file_name"],
                            "document_types": ";".join(answer.document_types), "primary_type": answer.primary_type,
                            "confidence": answer.confidence, "one_line_summary": answer.one_line_summary,
                            "language": answer.language, "model": GEMINI_MODEL, "labels_version": LABELS_VERSION,
                            "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        print(f"{r['file_name']}: {', '.join(answer.document_types)} (primary {answer.primary_type}, {answer.confidence}) | ${usage['cost_usd']}")

    sorted_files = contract_files.merge(pd.DataFrame.from_dict(results, orient="index"), left_on="id", right_index=True, how="left")
    # duplicates take the types of their first copy
    dup = sorted_files["status"] == "duplicate"
    for col in ["document_types", "primary_type", "confidence", "one_line_summary", "language", "cached"]:
        sorted_files.loc[dup, col] = sorted_files.loc[dup, "duplicate_of"].map(lambda i: results.get(i, {}).get(col))
    sorted_files.loc[sorted_files["status"] == "unreadable", "primary_type"] = "unreadable"
    sorted_files["document_types"] = sorted_files["document_types"].apply(lambda t: t if isinstance(t, list) else [])

    is_journal = sorted_files["document_types"].apply(lambda t: bool(JOURNAL_TYPES & set(t)))
    journals = sorted_files[is_journal & (sorted_files["status"] == "readable")]
    print(f"\n{len(journals)} journal(s) to read, {round(journals['size_mb'].sum(), 1)} MB")
    return sorted_files, journals


# ====================================== Stage 3: read ======================================
# ---- The fixed output shape: one class per thing Desiree's prompt names. DRAFT until confirmed with her. ----
ClaimType = Literal[tuple(CLAIM_TYPE_TO_HEDVIG)]


class ClaimRelevance(BaseModel):
    primary: ClaimType
    also_relevant: list[ClaimType] = []
    confidence: Literal["high", "medium", "low"]
    note_sv: Optional[str] = None
    terms_flags: list[Literal["dental_TR_FORL", "tooth_fracture", "baby_teeth", "hereditary_or_congenital_stated",
                              "accident_stated", "disease_stated", "routine_care", "foreign_body_or_poisoning"]] = []


class Symptom(BaseModel):
    text: str
    text_sv: Optional[str] = None
    onset_as_written: Optional[str] = None
    status: Literal["confirmed", "suspected", "ruled_out"]
    source_quote: str
    claim_relevance: ClaimRelevance


class Finding(BaseModel):
    text: str
    text_sv: Optional[str] = None
    is_negative: bool = False
    source_location: Optional[str] = None


class Diagnosis(BaseModel):
    text: str
    text_sv: Optional[str] = None
    status: Literal["confirmed", "suspected", "ruled_out"]
    source_quote: str
    claim_relevance: ClaimRelevance


class Procedure(BaseModel):
    text: str
    text_sv: Optional[str] = None
    result: Optional[str] = None
    source_location: Optional[str] = None
    claim_relevance: ClaimRelevance


class Medication(BaseModel):
    name: str
    dose: Optional[str] = None
    duration: Optional[str] = None
    reason: Optional[str] = None
    source_location: Optional[str] = None


class VetComment(BaseModel):
    type: Literal["assessment", "cause_opinion", "prognosis", "concern", "differential", "history_comment", "owner_compliance", "other"]
    text: str
    text_sv: Optional[str] = None
    source_quote: str


class Instruction(BaseModel):
    id: str                      # I1, I2, ...
    what: str
    what_sv: Optional[str] = None
    by_when: Optional[str] = None
    source_quote: str


class Visit(BaseModel):
    id: str                      # V1, V2, ... oldest first
    source_file: str             # file_1, file_2, ... (which attachment)
    page: Optional[int] = None
    date: Optional[str] = None   # ISO 8601
    date_as_written: Optional[str] = None
    date_precision: Literal["day", "month", "year", "unknown"] = "unknown"
    clinic: Optional[str] = None
    veterinarian: Optional[str] = None
    visit_type: Optional[str] = None
    channel: Optional[str] = None
    referral: Optional[str] = None
    weight: Optional[str] = None
    cost: Optional[str] = None
    reason_for_visit: Optional[str] = None
    symptoms: list[Symptom] = []
    findings: list[Finding] = []
    diagnoses: list[Diagnosis] = []
    procedures: list[Procedure] = []
    medications: list[Medication] = []
    vet_comments: list[VetComment] = []
    instructions: list[Instruction] = []
    review_flags: list[str] = []


class FollowThrough(BaseModel):
    instruction_id: str
    status: Literal["followed", "partially_followed", "not_followed", "declined_by_owner", "delayed", "no_evidence_in_record"]
    evidence_quote: Optional[str] = None
    visit_id: Optional[str] = None
    date: Optional[str] = None
    reason: Optional[str] = None


class Certificate(BaseModel):
    type: Optional[str] = None
    date: Optional[str] = None
    clinic: Optional[str] = None
    remarks_verbatim: Optional[str] = None
    source_location: Optional[str] = None


class RecordMeta(BaseModel):
    source_file: str
    animal_name: Optional[str] = None
    species: Optional[str] = None
    chip_number: Optional[str] = None
    clinics: list[str] = []
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    completeness_note: Optional[str] = None
    review_flags: list[str] = []


class RecordExtraction(BaseModel):
    record_meta: list[RecordMeta]
    visits: list[Visit]
    follow_through: list[FollowThrough] = []
    certificates: list[Certificate] = []


DESIREE_PROMPT = """# ROLE

You support claims handlers (skadereglerare) at Hedvig, a Swedish pet insurer. You turn a veterinary record (journalkopia, "JK") for a dog or cat into structured, source-traceable data for a health-history timeline.

You only record what the record says, when, and where. You never decide or imply coverage, pre-existing conditions, waiting period (karens), or that the owner breached the terms. The handler decides.

# INPUT

Text from ONE record (PDF/OCR/export): possibly several clinics and years, Swedish/English/other languages, OCR errors, abbreviations, template text. Optional animal facts from our system: use ONLY to check it is the right animal, never to fill gaps. If the record looks cut off or pages seem missing, say so in record_meta.completeness_note.

# DEFINITIONS

Visit = one encounter on one date at one clinic (phone and digital contacts count). Lab results, X-ray answers, referral letters and certificates belong to the visit they relate to; if they have no visit, make an entry with visit_type "dokument".

# TASK

1. Inventory: list every visit, oldest first (id, date, clinic, page). This is your checklist: "visits" must contain exactly these ids.
2. For each visit extract:
A. Appointment details: date, clinic, veterinarian, visit_type, channel, referring/referred-to clinic, weight, costs if shown; reason for visit (anamnes); owner-reported symptoms with onset/recurrence wording; clinical findings (include negative findings); diagnoses; tests and procedures with results; medications (name, dose, duration, reason).
B. Vet comments: every assessment or opinion, kept apart from objective findings. Type: assessment | cause_opinion | prognosis | concern | differential | history_comment | owner_compliance | other. Includes statements on whether a problem is new, recurring, chronic, hereditary/congenital, accident- or disease-related, resolved, and any note on owner delay or declined care ("ägaren har avvaktat", "avböjer utredning").
Instructions to the owner: medication courses, diet, rest, rechecks, tests/referrals to book, "återkom om ej bättre". Give each an id (I1, I2...), what, and by when if stated.
3. Follow-through (C), after all visits: for each instruction decide a status ONLY from explicit text in later entries: followed | partially_followed | not_followed | declined_by_owner | delayed | no_evidence_in_record. A missing later visit is NOT evidence; use no_evidence_in_record. Quote the evidence, give visit_id and date, and the reason if stated.
4. Claim relevance (D): attach claim_relevance (defined below) to every symptom, diagnosis and procedure. It points the handler to the Hedvig claim type; it is not a decision. Undiagnosed symptoms count ("klåda" → Skin Conditions and Allergies; "kräkningar" → Non Chronic gastrointestinal Issues). Routine care (vaccination, chipping, claw trimming) = "Default" + flag routine_care. If unsure: "Default", confidence low. Never invent a link.

# RULES

1. Evidence: copy source_quote exactly from the record (max 25 words) and write it BEFORE interpreting. Required for symptoms, diagnoses, vet comments, instructions, follow-through evidence and certificate remarks. For other items give source_location (page / entry date / heading) only. No evidence = no item.
2. Do not infer. Separate confirmed / suspected ("misstänkt", "?", "DD") / ruled_out ("ej påvisat", "utan anmärkning"). Capture negative findings; a clean health check matters.
3. Dates: ISO 8601 in "date"; original text in "date_as_written"; "date_precision" = day | month | year | unknown. If day/month order is ambiguous, don't guess; flag it. Keep onset wording as written ("sedan 2 veckor"); only calculate a date if explicit and mark it calculated.
4. Language: "text" = the record's wording; "text_sv" = Swedish translation, null if the original is Swedish. Expand abbreviations in text_sv. If unsure what one means, don't guess; flag unclear_abbreviation.
5. Certificates (besiktningsintyg/BI, veterinärintyg): fill "certificate". Remarks verbatim, or the explicit "no remarks" statement verbatim.
6. Privacy: exclude owner personal data (personnummer, address, phone, e-mail, bank). Animal name, chip number, clinic and vet names are fine.
7. Unknown = null, never a guess. Several animals in one record: keep separate.
8. review_flags use only: ambiguous_date | unreadable | unclear_abbreviation | possible_other_animal | likely_copy_forward | conflicting_entries | truncated.

# claim_relevance

{"primary": <claim type>, "also_relevant": [max 2 claim types], "confidence": "high|medium|low", "note_sv": <one sentence, only if confidence is not high>, "terms_flags": [diagnoses only, only if explicit in the text: dental_TR_FORL | tooth_fracture | baby_teeth | hereditary_or_congenital_stated | accident_stated | disease_stated | routine_care | foreign_body_or_poisoning]}

# CLAIM TYPES (exact names)

Arthritis and Joint issues | Asthma | Behavior Therapy | Cancer | Castration | Chronic gastrointestinal Issues | Claw Injury | Cruciate Ligament Injuries | Default | Dental Issues | Diabetes | Dystocia | Ear infections | Endocrine Disorders | Eye Problems | Foreign Body Ingestion | Fractures and Broken Bones | Heart Conditions | Infections | Intestinal Parasites | Kidney and Liver disease | Lacerations and Bite Wounds | Last Rest | Non Chronic gastrointestinal Issues | Pancreatitis | Poisoning | Seizures and Neurological Disorders | Skin Conditions and Allergies | Tooth Fracture and Baby Teeth | Urinary Tract Infections | Plastic Surgery Vet
(TR/FORL and other tooth/mouth disease = Dental Issues; tooth fracture and persistent baby teeth = Tooth Fracture and Baby Teeth.)
"""

# Changes for reading several records at once (Ramsha/Claude, 19 Nov; to be agreed with Desiree)
MULTI_RECORD_ADDITIONS = """
# SEVERAL RECORDS

You receive one or more records as separate attachments, each preceded by its label: [file_1], [file_2], ...
- Every visit carries "source_file" = the label of the attachment it comes from, next to "page".
- "record_meta" is a list with one entry per attachment (its source_file, the animal it describes, the clinics and dates it covers, completeness_note, review_flags).
- Records may overlap: the same visit can appear in two clinics' exports or in two uploads. Keep one visit entry and name both sources in its review_flags as "duplicate_in:<file label>". If two records describe the same visit differently, keep both and flag conflicting_entries.
- Visit ids run across all records (V1, V2, ... oldest first); instruction ids likewise (I1, I2, ...).

# KINDS OF ATTACHMENT

Not every attachment is a journal. Under [claims], each attachment is listed with its file name and the kinds of content it holds.
- An invoice or receipt attachment gives only the visit date, the treatments billed and any instructions printed on it: record a visit with exactly what is stated and infer nothing.
- A veterinary_certificate attachment goes under certificates.
- A lab_or_imaging_report attachment's values go under the findings of the visit on that date, or a visit of its own if there is none.

# THE CLAIMS

Every attachment belongs to one of the claims listed under [claims] (claim id, dates, outcome, claim types, attachments). The handler is assessing the claim marked "current: yes". The outcomes and types of the earlier claims are context, for example to note in note_sv when a visit belongs to an earlier claim that was already paid; do not copy them into the output. claim_relevance is still about Hedvig claim types in general, not a decision about any claim.
"""

PROMPT = DESIREE_PROMPT.replace("Text from ONE record (PDF/OCR/export)", "Text from one or more records (PDF/OCR/export, see SEVERAL RECORDS)") + MULTI_RECORD_ADDITIONS


def stage3(contract_id, contract_row, contract_files, journals, file_bytes, output_dir):
    journal_files = [(f"file_{n}", file_bytes[r["id"]], r["content_type"]) for n, (_, r) in enumerate(journals.iterrows(), start=1)]
    file_index = {f"file_{n}": r["file_name"] for n, (_, r) in enumerate(journals.iterrows(), start=1)}

    # Every claim of the contract, oldest first, with its outcome, claim types and the attachments stage 3 receives from it.
    # The most recently notified claim is the current one (test case; see decisions).
    claims = (contract_files[["claim_id", "claim_notification_date", "claim_date_of_loss", "claim_outcome", "claim_types_by_cost"]]
              .drop_duplicates("claim_id").sort_values("claim_notification_date"))
    current_claim_id = claims.iloc[-1]["claim_id"]
    attachments_of = {}
    for n, (_, r) in enumerate(journals.iterrows(), start=1):
        attachments_of.setdefault(r["claim_id"], []).append(f"file_{n}: {r['file_name']} ({', '.join(r['document_types'])})")
    blocks = []
    for _, c in claims.iterrows():
        blocks.append("\n".join([
            f"claim_id: {c['claim_id']}",
            f"current: {'yes' if c['claim_id'] == current_claim_id else 'no'}",
            f"notification_date: {c['claim_notification_date']}",
            f"date_of_loss: {c['claim_date_of_loss']}",
            f"outcome: {c['claim_outcome'] if pd.notna(c['claim_outcome']) else 'not yet decided'}",
            f"claim_types: {list(c['claim_types_by_cost'])}",
            "attachments: " + ("; ".join(attachments_of.get(c["claim_id"], [])) or "none sent"),
        ]))
    claims_text = f"[claims]\npet_type: {contract_row['pet_type']}\n\n" + "\n\n".join(blocks)
    print(claims_text, "\n")

    if journals["size_mb"].sum() <= MAX_INLINE_MB:
        print(f"One call with {len(journal_files)} journal(s)")
        record, _ = ask_gemini(journal_files, claims_text + "\n\n" + PROMPT, stage="3 read", label="all journals",
                               output_format=RecordExtraction)
    else:
        print(f"{journals['size_mb'].sum():.1f} MB is above MAX_INLINE_MB={MAX_INLINE_MB}: one call per journal, results concatenated")
        parts_out = []
        for jf in journal_files:
            rec, _ = ask_gemini([jf], claims_text + "\n\n" + PROMPT, stage="3 read", label=file_index[jf[0]],
                                output_format=RecordExtraction)
            parts_out.append(rec)
        record = RecordExtraction(
            record_meta=[m for p in parts_out for m in p.record_meta],
            visits=sorted([v for p in parts_out for v in p.visits], key=lambda v: v.date or "9999"),
            follow_through=[f for p in parts_out for f in p.follow_through],
            certificates=[c for p in parts_out for c in p.certificates],
        )

    # add the Hedvig claim-type name next to every claim_relevance, and the file name next to every source_file
    record_json = record.model_dump()

    def add_hedvig_names(obj):
        if isinstance(obj, dict):
            if "primary" in obj and "confidence" in obj:
                obj["primary_hedvig"] = CLAIM_TYPE_TO_HEDVIG.get(obj["primary"])
                obj["also_relevant_hedvig"] = [CLAIM_TYPE_TO_HEDVIG.get(t) for t in obj.get("also_relevant", [])]
            if "source_file" in obj:
                obj["source_file_name"] = file_index.get(obj["source_file"])
            for v in obj.values():
                add_hedvig_names(v)
        elif isinstance(obj, list):
            for v in obj:
                add_hedvig_names(v)

    add_hedvig_names(record_json)
    record_json["contract_id"] = contract_id
    record_json["current_claim"] = {"claim_id": current_claim_id, "claim_types_by_cost": list(claims.iloc[-1]["claim_types_by_cost"])}
    record_json["files"] = file_index
    record_json["claims_text"] = claims_text

    out = output_dir / "record.json"
    out.write_text(json.dumps(record_json, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"{len(record.visits)} visits, {len(record.follow_through)} follow-through entries, {len(record.certificates)} certificates | saved to {out}")

    visits_table = pd.DataFrame([{"id": v.id, "date": v.date, "clinic": v.clinic, "source_file": v.source_file, "page": v.page,
                                  "reason": v.reason_for_visit,
                                  "diagnoses": "; ".join(d.text for d in v.diagnoses),
                                  "claim_types": "; ".join(sorted({d.claim_relevance.primary for d in v.diagnoses} | {s.claim_relevance.primary for s in v.symptoms})),
                                  "flags": "; ".join(v.review_flags)} for v in record.visits])
    visits_table.to_csv(output_dir / "visits.csv", index=False)
    print(visits_table.to_string())
    return record


# ====================================== Stage 4: cost ======================================
def stage4(contract_id, output_dir):
    calls = pd.DataFrame(CALLS)
    total = calls[["input_tokens", "output_tokens", "thinking_tokens", "total_tokens", "cost_usd"]].sum()
    by_stage = calls.groupby("stage")["cost_usd"].sum().round(4).to_dict()
    print(f"\n{len(calls)} Gemini calls for contract {contract_id} ({int(calls['cached'].sum())} answered from the label cache): "
          f"{int(total['total_tokens']):,} tokens, ${total['cost_usd']:.4f}")
    print("by stage:", by_stage)
    calls.to_csv(output_dir / "gemini_calls.csv", index=False)
    return calls, total, by_stage


# ========================================= Main =========================================
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--contract", default=CONTRACT_ID, help="contract id (default: CONTRACT_ID in the config section)")
    contract_id = parser.parse_args().contract
    output_dir = OUTPUT_ROOT / contract_id
    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    print(f"=== Stage 1: gather files for contract {contract_id} ===")
    contract_row, contract_files, file_bytes = stage1(contract_id)

    print("\n=== Stage 2: sort the files ===")
    sorted_files, journals = stage2(contract_files, file_bytes)
    files_out = sorted_files.drop(columns=["bucket", "key"]).copy()
    files_out["document_types"] = files_out["document_types"].apply(";".join)
    files_out["claim_types_by_cost"] = files_out["claim_types_by_cost"].apply(";".join)
    files_out.to_csv(output_dir / "contract_files.csv", index=False)

    print("\n=== Stage 3: read the journals ===")
    if journals.empty:
        print("No files with a JOURNAL_TYPES label; nothing to read.")
        record = None
    else:
        record = stage3(contract_id, contract_row, contract_files, journals, file_bytes, output_dir)

    print("\n=== Stage 4: cost ===")
    calls, total, by_stage = stage4(contract_id, output_dir)

    summary = "\n".join([
        f"contract: {contract_id}",
        f"pet_type: {contract_row['pet_type']}",
        f"run started: {started:%Y-%m-%d %H:%M} UTC; model: {GEMINI_MODEL}; labels_version: {LABELS_VERSION}",
        f"claims: {contract_row['nr_claims']}; files: {contract_row['nr_files']} "
        f"({', '.join(f'{v} {k}' for k, v in contract_files['status'].value_counts().items())})",
        f"labels: " + ", ".join(f"{k} {v}" for k, v in sorted_files["primary_type"].value_counts().items()),
        f"files sent to the reader: {len(journals)} ({round(journals['size_mb'].sum(), 1)} MB)",
        f"visits: {len(record.visits) if record else 0}; instructions followed through: {len(record.follow_through) if record else 0}; "
        f"certificates: {len(record.certificates) if record else 0}",
        f"gemini calls: {len(calls)} ({int(calls['cached'].sum())} from the label cache); tokens: {int(total['total_tokens']):,}; "
        f"cost: ${total['cost_usd']:.4f} (" + ", ".join(f"{k}: ${v}" for k, v in by_stage.items()) + ")",
        f"outputs: {output_dir}/contract_files.csv, record.json, visits.csv, gemini_calls.csv",
    ])
    (output_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    sys.exit(main())
