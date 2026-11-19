# Shortlist data sample

`shortlist_data_sample.csv` is the sample of 100 pet-claim files that Gemini (gemini-3.8-flash, Vertex AI, 19 Nov 2026)
described in its own words. The descriptions were used to derive the `DOCUMENT_TYPES` label list in
`vet-record-reader/contract_journal_reader.ipynb`.

How the 100 were chosen: files on pet claims (`claims_reporting.promise_category = 'Pet'`), uploaded after
2025-01-01, not deleted (`deleted_at IS NULL`), content type PDF/JPEG/PNG, ordered by
`FARM_FINGERPRINT(CAST(f.id AS STRING))`, first 100.

Columns: `id`, `claims_id`, `file_name`, `content_type`, `upload_source`, `uploaded_at` (from
`dbt_prod.stg_claims_service__claim_file`); `what_it_is`, `kind`, `issuer`, `summary`, `language` (Gemini's answer;
`kind = ERROR` means Gemini could not process the file); `size_mb`, `cost_usd`.
