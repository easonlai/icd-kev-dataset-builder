# ICD-10-CM Kev Dataset Builder

Build English synthetic medical notes and turn them into training data for a small model that helps assign ICD-10-CM diagnosis codes.

The aim is to teach Kev to read a note, choose its way through the coding hierarchy, and identify the text supporting the result. Instead of asking one enormous question containing the whole code catalogue, the proposed application asks a series of smaller questions.

This repository builds the data for that experiment. It downloads and checks official CDC references, generates fictional outpatient notes through an Azure OpenAI-compatible deployment, reviews them, and exports training files. It does not train Kev or implement the application that will use it.

The first recorded trial produced 1,460 accepted synthetic notes. The files passed a local Kev format check, but no fine-tuning result or independent clinical validation is reported here.

## Start here

- To understand the idea, read [a worked example](#a-worked-example).
- To understand the training data, read [what one case teaches](#what-one-case-teaches).
- To run the program, start with [installation and official references](#installation-and-official-references).
- To use the exports, read [which files to use](#which-files-to-use) and [training and evaluation](#training-and-evaluation).
- For the recorded results and their limits, read [the trial dataset](#the-trial-dataset).

## Contents

- [What problem this project tackles](#what-problem-this-project-tackles)
- [A worked example](#a-worked-example)
- [What one case teaches](#what-one-case-teaches)
- [How the pipeline builds the dataset](#how-the-pipeline-builds-the-dataset)
- [Installation and official references](#installation-and-official-references)
- [Connect the generation model](#connect-the-generation-model)
- [Run a trial](#run-a-trial)
- [Which files to use](#which-files-to-use)
- [Training and evaluation](#training-and-evaluation)
- [The trial dataset](#the-trial-dataset)
- [Design decisions still to resolve](#design-decisions-still-to-resolve)
- [Tests](#tests)
- [License](#license)

## What problem this project tackles

ICD-10-CM is a system for assigning codes to documented diagnoses. Closely related diagnoses can have different codes: a recorded complication, a disease subtype, or a bleeding status can change the assignment. Good coding also depends on the applicable conventions and instructions, not just similar wording.

A model therefore needs more than a list of disease names. It needs practice reading the details that distinguish one code from another—and knowing when the note does not justify a choice.

This project explores a particular approach:

```text
Read the original note
    → choose a chapter
    → choose a disease group
    → narrow the category as needed
    → select a complete code
    → identify supporting text and any remaining coding checks
```

Think of it as finding a book in a large library: first the subject area, then the section, then the shelf. The route reduces the number of options at each decision. It does not guarantee the right answer: choosing the wrong section early can hide the correct code later.

Kev is the decision model in this design. A separate controller—the program managing the search—must provide its options, follow promising branches, handle mistakes and decide when to stop. Ordinary Kev fine-tuning does not remove its need for supplied options.

### What the experiment is trying to establish

The main question is whether a domain-trained Kev can make better coding choices than default Kev under the same conditions. A second question is whether training lets it work with shorter instructions or option descriptions.

| Question | What would count as useful evidence? |
|---|---|
| Does training improve code selection? | Better complete-code results on the same held-out cases, not just lower training loss |
| Can the model follow the hierarchy? | The correct route remains available from the first decision through the final code |
| Does it distinguish neighboring codes? | Fewer errors on documented subtype, complication and other code-determining details |
| Does it stop or seek review appropriately? | Fewer unsupported codes, wrong stops and unnecessary abstentions |
| Can it point to supporting text? | Evidence judgments agree with independently checked annotations |
| Can the input be shorter? | Comparable or better results with fewer total input tokens and acceptable end-to-end latency |

These are questions for future experiments, not benefits already demonstrated by this repository.

### What it is not for

The initial task concerns one focused diagnosis in a short English outpatient note. It is not full-encounter coding, inpatient principal-diagnosis selection, diagnosis from symptoms, reimbursement optimization, insurance pricing, or a replacement for a qualified coder.

A note may mention other conditions, but the target must be clear. Selecting one code does not establish that every required code for the encounter has been assigned. Synthetic evaluation alone is not evidence that a model is ready for clinical or billing use.

Do not send real patient information or copied patient narratives into this generation workflow.

## A worked example

Here is a fictional note, included only to explain the design:

> The clinician documents type 2 diabetes mellitus with diabetic chronic kidney disease. The stage of chronic kidney disease is not specified in this note.

For the focused diabetes diagnosis, the example reference path is:

```text
Endocrine, nutritional and metabolic diseases
    → Diabetes mellitus (E08–E13)
    → Type 2 diabetes mellitus (E11)
    → Kidney complications (E11.2)
    → Diabetic chronic kidney disease (E11.22)
    → STOP
```

Production labels must be checked against the locked local codebook and applicable instructions. The example does not replace that check. Additional CKD coding requirements are separate from this focused mapping.

### How the note becomes practice questions

Every question retains the original note. What changes is the current position and the available choices.

| Question | Reference answer | Why ask it? |
|---|---|---|
| Which chapter contains the focused diagnosis? | Endocrine, nutritional and metabolic diseases | Do not route solely on the word “kidney” |
| Which group in that chapter? | Diabetes mellitus | Find the relevant disease group |
| Which diabetes category? | E11 | Use the explicitly recorded type |
| Which complication branch? | E11.2 | Recognize the documented kidney complication |
| Which complete code under that branch? | E11.22 | Distinguish the supported code from nearby alternatives |
| Has this focused mapping reached its complete code? | STOP | End the route |

Other diagnoses take fewer or more steps. Five levels are not a requirement.

The same note can also support a wrong-branch exercise: if the current parent is unrelated to the focused diagnosis, the answer is `NO_MATCH`. An evidence exercise asks which sentences directly support the mapping. In this example, the first sentence documents the combination diagnosis; the second identifies a staging limitation rather than cancelling that diagnosis.

### What the user should eventually see

The intended application would show a code, the route it actually selected, the supporting original text, and any further coding checks. That application is not implemented here.

There are three different things to keep separate:

- **Reference path:** the correct route looked up from a dataset label.
- **Selected path:** the decisions actually made during inference.
- **Supporting evidence:** the note text that justifies the assignment.

Looking up a code's ancestors after prediction can show its classification, but it is not a record of how the model reached it. A valid-looking route is not enough if the original note does not support the code.

## What one case teaches

### A case is not the same as a training row

The generator creates a master case: one fictional note and its annotations. The exporter then makes several tasks from it.

| Item | Meaning |
|---|---|
| Planned slot | A requested case with an assigned split, outcome and scenario specification |
| Generation attempt | One attempt to fill a slot; retries or replacements may be needed |
| Candidate | A generated case waiting for acceptance checks |
| Accepted master case | A case retained after the configured checks and model review |
| Routing task | One decision at a reference parent, including the final STOP question |
| Wrong-branch task | A `NO_MATCH` exercise at a selected incorrect parent or leaf |
| Evidence task | Sentence-level support judgments derived from the note |

One case can produce many rows and questions. They are correlated examples, not separate patients. Requested slots, accepted cases, task rows and question counts must be reported separately.

### What Kev receives—and what it must not receive

A routing input contains the original note, current parent, instructions and candidate options. The label tells the training process which option is correct.

The future correct path, final gold answer, evidence answers and review verdict must not be put into model-visible input. Tracking fields such as case ID, split and task type belong outside it too. For example, a visible `routing_no_match` task type can disclose the answer even if `label` has been removed.

A code appearing in the candidate list is not itself leakage. The problem is revealing which candidate is correct. Check the actual loader and request construction rather than assuming a field name guarantees safety.

### Why start with these 50 codes?

The pilot covers five clinical areas. Broad differences test chapter routing; close differences within a family test whether the model reads the details that matter.

| Family | Examples of distinctions to learn |
|---|---|
| Diabetes | Recorded type, glucose complications, kidney complications and neurological complications |
| Respiratory | Asthma classification and exacerbation status, COPD variants, and pneumonia |
| Circulatory | Coronary disease with or without angina, atrial fibrillation type, and heart-failure type |
| Kidney and urinary | CKD stage, stage 3 subdivisions, ESRD and unspecified stage |
| Digestive | Reflux, esophagitis, bleeding, gastritis and ulcer distinctions |

The codes are selected experiment targets, not a claim about the most common diagnoses. The build checks each target against the official code descriptions and supported tabular path, and fails rather than substituting a different code.

| Group | Codes |
|---|---|
| Diabetes | `E10.9`, `E10.65`, `E11.9`, `E11.65`, `E11.21`, `E11.22`, `E11.29`, `E11.40`, `E11.42`, `E11.649` |
| Respiratory | `J45.20`, `J45.21`, `J45.30`, `J45.31`, `J45.40`, `J45.41`, `J44.9`, `J44.1`, `J44.0`, `J18.9` |
| Circulatory | `I10`, `I25.10`, `I25.119`, `I48.0`, `I48.11`, `I48.19`, `I48.21`, `I50.22`, `I50.32`, `I50.42` |
| Kidney and urinary | `N18.1`, `N18.2`, `N18.30`, `N18.31`, `N18.32`, `N18.4`, `N18.5`, `N18.6`, `N18.9`, `N39.0` |
| Digestive | `K21.9`, `K21.00`, `K21.01`, `K29.50`, `K29.51`, `K29.70`, `K29.71`, `K25.9`, `K26.9`, `K59.00` |

The list is defined in `PILOT_TARGET_CODES` in [kb.py](src/icd_kev_dataset_builder/kb.py).

### Variation should change more than the writing style

A useful dataset contains different ways of documenting the same facts, as well as examples where a clinically relevant detail changes the answer.

Changing sentence order or using an ordinary abbreviation should not change a label. Changing an explicitly recorded complication or exacerbation status may do so. Medication, age or a laboratory value alone should not be used to invent a diagnosis or stage. If a coding convention supports a linkage, retain its rule reference.

The planner currently assigns a style, trait and length. Styles include brief SOAP notes, prose, problem lists, telephone follow-ups and abbreviation-heavy notes. Those controls encourage variety, but they do not prove that every important clinical contrast is covered.

A proposed next step is a contrast matrix listing the determining attribute, competing values and expected labels for each family. It would make gaps easier to see. Repeated hobbies, mechanical contradictions or a distinctive style for each code can create shortcuts instead of useful learning.

### Four different ways to stop or change direction

| Choice | Plain-language meaning | Current training coverage |
|---|---|---|
| `STOP` | “This node completes the supported focused mapping.” | Final mapped tasks and wrong-leaf contexts |
| `NO_MATCH` | “The current branch is wrong for this diagnosis.” | Selected incorrect parents or leaves |
| `OUT_OF_SCOPE` | “The supported result is outside this pilot.” | ROOT-level tasks |
| `NEEDS_REVIEW` | “The documentation remains unresolved under the applicable rules.” | ROOT-level tasks |

Missing detail does not automatically require review: an unspecified code or an applicable convention may resolve the assignment.

The present STOP policy teaches recognition of a supplied endpoint. It does not fully teach whether to stop or continue at a node that is both reportable and has descendants. Verify whether and where that situation occurs in the supported release before designing examples for it; do not assume every category with children is reportable.

Review and outside-pilot cases currently stop at ROOT. Preserving a supported prefix is a proposed extension. A note might clearly identify CKD but contradict itself about stage; the system could retain the renal path and request review at the unresolved distinction. If the diagnosis itself is disputed, stopping earlier may be appropriate.

### Evidence means more than finding a matching quote

Master records contain sentence spans, focus IDs and supporting quotes. Native evidence tasks label sentences using focus-sentence membership. Exact-span checks confirm that a quote exists, not that it is clinically sufficient.

The annotation policy needs to be explicit. A useful proposed definition is: a positive sentence directly documents the focused diagnosis or an attribute needed for its code assignment. Symptoms and treatment plans are not automatically positive or negative; their relevance depends on the task.

Some assignments need more than one sentence. Include justified multi-sentence support, negation and unsupported alternatives. A question with no candidate code is evidence selection for the focused diagnosis, not a general verifier for arbitrary codes.

## How the pipeline builds the dataset

```text
Import official files → build and audit references → prepare pilot context
    → plan cases → generate candidates → review and deduplicate
    → audit acceptance → write master files → export Kev tasks
```

Only generation and model review make paid inference calls.

| Stage | Main modules | Responsibility |
|---|---|---|
| Sources and knowledge base | `sources.py`, `kb.py` | Import/download, lock source bytes, parse official facts and audit support |
| Pilot context | `reference.py` | Prepare verified pilot paths, options and reference material |
| Planning and state | `planner.py`, `state.py` | Allocate slots and scenario families; maintain durable SQLite state |
| Generation | `generation.py`, `llm.py` | Build bounded prompts, call the deployment and validate candidates |
| Review | `review.py`, `dedupe.py` | Re-code notes without the intended answer, check duplicates and accept transactionally |
| Reports and finalization | `reports.py` | Report integrity and quota gaps; write master exports and dataset card |
| Kev export | `kev.py`, `kev_native.py` | Derive tasks and convert them to the configured native format |

### Official facts come from files, not the generator

Python establishes code descriptions, reportability, hierarchy paths, source references, IDs and provenance. The model writes the fictional narrative and proposes focus sentences, quotes and related details. Quotes must match the generated text.

The hierarchy follows tabular XML containment, not code-prefix truncation. Coding instructions such as “use additional code” are not parent-child edges. Local and inherited notes matter. A catalogue entry without a supported tabular path is reported as unsupported; no path is invented. Every pilot target must be reportable and have a supported path.

The reference build writes:

| Artifact under `data/reference/` | Purpose |
|---|---|
| `codebook.json` | Official descriptions and reportability information |
| `hierarchy.json` | Tabular nodes, parent-child relationships and coding notes |
| `reference_chunks.jsonl` | Source-addressable index and guideline text |
| `kb_audit.json` | Source binding, pilot validation and unsupported-branch report |
| `pilot_context.json` | Added by `kb pilot`: verified pilot paths, choices and context |

Generation uses bounded reference packets, lexical BM25 retrieval and selected outpatient rules. Code identifiers are sanitized from the generation context as configured. Retrieval helps find relevant material; it does not establish that every applicable rule has been found.

Reference text and model output are treated as data, not instructions.

### Plan before generating

The full plan targets accepted master cases:

| Split | Mapped per code | Mapped total | Outside pilot | Needs review | Total |
|---|---:|---:|---:|---:|---:|
| Train | 50 | 2,500 | 250 | 250 | 3,000 |
| Validation | 8 | 400 | 50 | 50 | 500 |
| Test | 16 | 800 | 100 | 100 | 1,000 |
| Total | 74 | 3,700 | 400 | 400 | 4,500 |

Before a model call, each slot gets a split, outcome, scenario family, style, trait and length. The default seed is `20270` and family size is two. The seed reproduces the plan, not the model's text.

Families stay in one master split. Native export later subdivides validation into calibration and development by case hash; that subdivision can split a family. The recorded overlap is described under [the trial dataset](#the-trial-dataset).

### Generate candidates, then review them

Generation defaults to five slots per batch, processed sequentially. A dry run constructs and checks prompts without model calls. Original slot and family IDs are not sent to the generator. Prompt checks and output checks look for forbidden answer codes, including dotted and undotted pilot forms.

Local checks cover schema, outcome consistency, sentence spans, exact quotes and code leakage. A valid candidate is not accepted immediately: it goes to model review.

The reviewer sees the note and a fixed bank of the 50 pilot codes without the intended answer. It may return a pilot code, an outside code or an unresolved result. Its answer is compared with the stored label. For outside codes, agreement may use the same three-character category as a heuristic; that comparison is not hierarchy inference or proof of exact-code agreement.

This is a blind re-code, not a separate target-conditioned audit of every rule and evidence label. Generator and reviewer may use the same deployment. Their agreement is not independent clinical validation, and the model cannot mark human review complete: `human_review` remains `not_run`.

See the actual [generation prompt](prompts/generate.v1.md), [review prompt](prompts/review.v1.md) and [schemas](src/icd_kev_dataset_builder/schemas.py) for the implemented instructions and fields.

### Reject duplicates and keep an audit trail

Before acceptance, normalized exact duplicates and word-3-gram near duplicates are checked against accepted notes, including other splits. The Jaccard threshold is 0.8. This catches some repetition, but not all paraphrases or reused clinical scenarios. Semantic duplicate detection and independent human review have not been performed.

Acceptance is transactional and unique per slot. Failed cases are not silently relabelled to fill quotas. Each slot gets at most `--max-attempts` attempts, default three; exhausted slots remain visible as shortfalls. Generation also stops after three consecutive batches with no valid candidates.

SQLite at `data/state/plan.sqlite` holds slots, attempts, candidates, review results, duplicate decisions and accepted records. The plan is bound to the release, source manifest, seed, family size and plan hash. An advisory `plan.lock` prevents concurrent generation/review.

An interrupted remote call may have completed or incurred a charge before its result was saved. Such attempts can be marked `uncertain_abandoned`; exactly-once remote execution is not promised.

## Installation and official references

### Set up Python

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[pdf]'
```

The PDF extra installs what is needed to extract guideline text. The OpenAI SDK is a project dependency and is also needed by generation dry runs. For offline tests:

```bash
python -m pip install -e '.[test]'
```

The examples below set `PYTHONPATH=src` when running from a checkout. An installed package can also be used.

### Download the CDC files

The default configuration uses FY2027 ICD-10-CM, October 1, 2026 release. Start from the [official directory](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2027/) or [CDC files page](https://www.cdc.gov/nchs/icd/icd-10-cm/files.html). Exact URLs are configured in [config/default.toml](config/default.toml).

| Source ID | Official file | Use |
|---|---|---|
| `table_and_index` | [Table and Index ZIP](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2027/icd10cm-table-and-index-2027.zip) | Tabular XML hierarchy, coding notes, and Alphabetic Index entries. |
| `code_descriptions` | [Code Descriptions ZIP](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2027/icd10cm-code-descriptions-2027.zip) | Code descriptions and reportable-code catalogue. |
| `guidelines` | [Official Guidelines PDF](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2027/ICD-10-CM-October-1-2026-FY27-Guidelines.pdf) | Extracted, page-referenced guideline text; PDF extraction requires `pypdf`. |
| `addenda` | [Addenda ZIP](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2027/icd10cm-addenda-2027.zip) | Checksummed reference; not reapplied over the main release. |
| `tabular_pdf` | [Readable Tabular PDF](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2027/icd10cm-tabular-2027.pdf) | Optional human-readable reference. |

The first four are required by the current build. The optional tabular PDF is for reading. The addenda are retained as checksummed references, not automatically reapplied over changes already in the main release.

### Import and build

Download the files, replace the example local paths, then run:

```bash
PYTHONPATH=src python -m icd_kev_dataset_builder doctor
PYTHONPATH=src python -m icd_kev_dataset_builder sources import \
  --source-id table_and_index --path /path/to/icd10cm-table-and-index-2027.zip
PYTHONPATH=src python -m icd_kev_dataset_builder sources import \
  --source-id code_descriptions --path /path/to/icd10cm-code-descriptions-2027.zip
PYTHONPATH=src python -m icd_kev_dataset_builder sources import \
  --source-id guidelines --path /path/to/ICD-10-CM-October-1-2026-FY27-Guidelines.pdf
PYTHONPATH=src python -m icd_kev_dataset_builder sources import \
  --source-id addenda --path /path/to/icd10cm-addenda-2027.zip
PYTHONPATH=src python -m icd_kev_dataset_builder kb build
PYTHONPATH=src python -m icd_kev_dataset_builder kb audit
```

Alternatively, explicitly run `icd-data sources download` to download the configured required files; use `--include-optional` for optional sources. Downloads are not automatic. No model calls occur during source import or reference building.

The importer preserves original bytes, checks file signatures and archive safety, and records SHA-256 hashes, source roles, URLs, member names, import times and release evidence in `sources.lock.json`. Downloads have HTTPS, host/redirect restrictions, timeouts, bounded retries and size limits. Failures are reported rather than silently changing release or URL.

Keep the lock with the derived knowledge base and dataset when moving workspaces. It is ignored by Git, but is not disposable. Re-importing one source replaces it atomically and can change the manifest hash; stale reference artifacts must be rebuilt and the plan must remain compatible.

A checksum identifies the bytes used. It does not, by itself, prove that those bytes belong to the claimed release. The recorded guideline-release caveat is preserved in [source provenance](#source-provenance).

## Connect the generation model

Configure the environment or create a local `.env` from `.env.example`. Existing environment variables take precedence. Never commit secrets or put them in logs.

The client calls Chat Completions through the OpenAI SDK, not the Foundry Agent Service or a Foundry project endpoint. In v1 mode it appends `/openai/v1/` to the resource base endpoint. Classic mode uses `AzureOpenAI` with an API version. The deployment name is sent as `model`.

Not every model available in Foundry supports this API and its requested parameters. Host acceptance does not establish compatibility. See Microsoft's [endpoint guide](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/endpoints).

| Variable | Purpose |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | Required HTTPS resource base endpoint, without `/openai/v1/` |
| `AZURE_OPENAI_API_KEY` | Required API key; the current client uses key authentication |
| `AZURE_OPENAI_GENERATION_DEPLOYMENT` | Required generation deployment identifier |
| `AZURE_OPENAI_REVIEW_DEPLOYMENT` | Optional; defaults to generation deployment |
| `AZURE_OPENAI_API_VERSION` | Selects classic mode by default when set; required in classic mode |
| `AZURE_OPENAI_MODE` | Overrides the default choice of classic or v1 |
| `AZURE_OPENAI_TIMEOUT_SECONDS` | Request timeout |
| `AZURE_OPENAI_MAX_RETRIES` | Retry setting |
| `AZURE_OPENAI_MAX_OUTPUT_TOKENS` | Output budget |
| `AZURE_OPENAI_TEMPERATURE` | Optional temperature |
| `AZURE_OPENAI_JSON_MODE` | `auto`, `on` or `off` |

The endpoint host must end in `.openai.azure.com`, `.services.ai.azure.com` or `.cognitiveservices.azure.com`. Variable names retain the `AZURE_OPENAI_` prefix even when the deployment is described as a Foundry deployment.

Unless disabled, the client requests JSON mode. It sends other optional parameters when configured. If a bad-request error identifies an optional parameter, the client may disable that parameter and retry, including JSON response format. Rate limits, 5xx and network errors use bounded backoff. Authentication errors, refusals and content-filter blocks are not treated as transient failures. The client does not bypass provider safety controls.

A deployment name is configuration, not independent verification of the underlying model identity.

## Run a trial

The commands below assume sources and the model connection are ready. Generation and model review are paid stages. Begin with a small limit when setting up a new environment.

### Prepare the workspace

```bash
cd /path/to/icd-kev-dataset-builder
source .venv/bin/activate
export PYTHONPATH=src
```

If continuing a run, do not clean up. Read [stopping and resuming](#stopping-and-resuming).

For a fresh run, first preview the archive operation:

```bash
python scripts/cleanup_previous_run.py --dry-run
python scripts/cleanup_previous_run.py
```

The second command asks you to type `CLEAN`, then moves old state and outputs to `data/archive/`. Never run it while the pipeline is active. It leaves official sources and reference data in place. Non-interactive `--yes` should be used only after inspecting the preview.

### Build the plan

```bash
python -m icd_kev_dataset_builder.cli kb pilot
python -m icd_kev_dataset_builder.cli plan
```

The plan has 4,500 slots. Planning itself does not start generation. An existing plan is resumed only when release, source manifest, seed and family size agree.

### Preview and generate

For a small smoke test, replace `1500` below with `50`. For the trial budget used in this project:

```bash
python -m icd_kev_dataset_builder.cli generate --dry-run --limit 1500 --batch-size 5
python -m icd_kev_dataset_builder.cli generate --limit 1500 --batch-size 5 --max-requests 2000
```

The preview checks prompts without creating candidates or calling the model. It reports the requests needed; `--max-requests` does not limit a dry run.

`--limit 1500` means up to 1,500 eligible slots for this invocation. It does not promise 1,500 accepted notes. Generation can fail and review can reject a candidate. Do not add `--full-run` unless you intend to select all eligible slots in the larger plan.

Use `--outcome MAPPED`, `--outcome OUT_OF_SCOPE` or `--outcome NEEDS_REVIEW` when selecting one outcome is useful.

### Review, audit and export

Wait for generation to finish before starting review:

```bash
python -m icd_kev_dataset_builder.cli review --max-requests 2000
python -m icd_kev_dataset_builder.cli status
python -m icd_kev_dataset_builder.cli audit
```

Read the review summary's `errors`, `reasons` and `stopped_reason`. Check `candidate_status` for `pending_review`; request errors can leave pending candidates even if a summary says `completed`. Run review again to process them.

Check `integrity.passed` and `integrity.records_checked`, but also check the accepted count. Integrity can pass with zero records; it does not mean the quota is filled.

When review and integrity are satisfactory:

```bash
python -m icd_kev_dataset_builder.cli finalize
python -m icd_kev_dataset_builder.cli export-kev
```

The trial will normally be `PARTIAL` against the full 4,500-case quota. That is expected. Master files are in `data/output/`; native files for Kev are in `data/output/kev/native/`.

### Monitor progress

In a second terminal with the same environment, run the read-only command:

```bash
python -m icd_kev_dataset_builder.cli status
```

It is safe while generation runs. Do not start another generation or review process concurrently.

| Field | How to read it |
|---|---|
| `attempts` | Slot-level attempts, including later attempts for a slot; HTTP retries are not separate slot attempts |
| `slots_by_split` | Planned, in-progress, accepted and exhausted slots; generated candidates remain in-progress until reviewed |
| `accepted_records` | Review-accepted cases; usually stays zero during generation |
| `family_split_violations` | Should stay zero at the checked master-split level |

Batches are processed split by split. Seeing early progress only in one split is not, by itself, a sign that the program is stuck.

### Stopping and resuming

Press Ctrl+C and wait for the command to return before doing anything else. SQLite keeps completed work. An interrupted Azure request may still be charged.

To resume review, repeat the review command. Saved decisions remain; pending candidates are processed. Audit afterwards.

To generate more, first clear pending review and inspect status. Do not blindly repeat `generate --limit 1500`: another invocation can select up to another 1,500 eligible slots. If aiming for at most 1,500 accepted cases, use the remaining gap as the next limit after pending review is cleared. Failed slots can still prevent reaching the target.

Generation selects slots once per invocation. A slot released after a generation failure, or rejected during later review, needs another invocation for a further attempt, up to `--max-attempts`.

`--max-requests` is a per-invocation logical-request budget; provider retries can add calls. `--max-tokens` is checked between batches, so the last batch can exceed it. The project does not assume model prices or estimate a bill.

The cleanup helper archives under `data/archive/<timestamp>/` and leaves `data/state/plan.lock` in place. Cleanup and planning make no model calls.

### A full run

Only proceed after reviewing the cost. From a fresh or compatible existing plan:

```bash
python -m icd_kev_dataset_builder.cli generate --full-run --batch-size 5 --max-requests 1200
python -m icd_kev_dataset_builder.cli review --max-requests 5000
python -m icd_kev_dataset_builder.cli audit
```

Inspect summaries and resume the appropriate stage if a budget stops it or candidates remain pending. Claim completion only when `quota_complete: true`; exhausted slots can leave permanent shortfalls. Then finalize and export as above.

## Which files to use

There are several exports because the same accepted notes are represented in different ways. They are not separate collections of generated patients.

### Master records and reports

| Under `data/output/` | Contents |
|---|---|
| `train.master.jsonl`, `validation.master.jsonl`, `test.master.jsonl` | Accepted cases, sorted by record ID |
| `quota_report.json` | Accepted counts and per-cell shortfalls |
| `audit_report.json` | Integrity findings and acceptance status |
| `duplicate_report.json` | Duplicate decisions and related findings |
| `human_review_sample.jsonl` | Sample stratified by split and outcome |
| `dataset_card.json` | Source release, scope, deployment IDs, status, unsupported branches and limitations |
| `config_snapshot.json` | Redacted configuration, without environment values |

A master case carries the note, split/family, sentence spans and focus IDs, outcome/code fields, reference lineage, quotes, source references, and generation/review provenance. The schema is in [schemas.py](src/icd_kev_dataset_builder/schemas.py). It has no generated confidence percentages or invented gold Top-5 ranking.

### Generic task exports

| Under `data/output/kev/` | Contents |
|---|---|
| `{train,validation,test}.described.labelled.jsonl` | Readable option descriptions and answers |
| `{train,validation,test}.described.inference.jsonl` | Described tasks without labels |
| `{train,validation,test}.compact.labelled.jsonl` | IDs/codes without explanatory option descriptions, with answers |
| `{train,validation,test}.compact.inference.jsonl` | Compact tasks without labels |
| `manifest.json` | Source binding, task/native counts, option budget and export notes |

Generic labelled rows carry task ID, case ID, split, input and label. Described and compact variants share the cases, labels and splits.

An “inference” filename does not make the file a serving request. Metadata may still reveal the task, and routing rows use reference parents. Strip tracking information from model-visible input and construct real controller requests separately.

### Native exports for Kev

Use `data/output/kev/native/{train,calibration,development,test}.jsonl` with the relevant Kev tooling. These are labelled native records containing `state`, `questions` and `_meta`.

- Routing becomes a `choice` question. Option names are node IDs or control values; readable descriptions include the code where applicable.
- Evidence becomes one `noul` yes/no question per sentence.
- Choice `label` and one-hot `target` are supervision, not inference input.
- `_meta` supports traceability but is not a replacement for all master-case provenance.
- Only the described variant is converted to native format at present.
- Master validation cases are assigned by case hash to calibration or development; train and test stay separate.

The adapter uses character-based size estimates. It skips rather than truncates oversized records and reports them as `native.skipped_over_limit`. An exact tokenizer check is separate. The recorded interface limits are 255 options and 1,024 tokens per question; verify them against the Kev version actually used. The exporter defaults to 24 options.

### What gets overwritten?

SQLite accumulates attempts, candidates and accepted records during a run. `finalize` atomically rewrites master files and reports from accepted state; it does not append to the old JSONL files. `export-kev` similarly rewrites its fixed-name exports.

Back up or archive state and outputs to keep a prior snapshot. Re-export does not create a versioned copy or clean unrelated files manually placed in the output directory.

The four native files are intended to be shared in the repository. Other data artifacts and the source lock are ignored. Git ignore does not untrack existing files: inspect the index and applicable third-party terms before publishing.

## Training and evaluation

The builder prepares files; use Kev's own training instructions for fine-tuning. Modal training is paid. The guidance referenced here is under `skills/kev-finetune` in the Kev checkout.

### Check compatibility first

Run the local validator on each exported partition:

```bash
cd /path/to/kev
for part in train calibration development test; do
  python3 skills/kev-finetune/scripts/split_data.py \
    "/path/to/icd-kev-dataset-builder/data/output/kev/native/$part.jsonl"
done
```

Without `--out`, this checks rather than splits. Inspect invalid-line, duplicate and conflict counts; a successful exit alone does not mean all are zero. The files are already partitioned by case—do not re-split them inadvertently.

The uploaded README records that `kev@2074cc4` checked each partition on 7 November 2026 and reported zero invalid lines, exact duplicates or conflicting states. That was a per-file format check, not a cross-partition comparison, exact-tokenizer check or training run. New exports need checking again.

### Use each partition for its own job

| File | Job |
|---|---|
| `train.jsonl` | Gradient updates |
| `development.jsonl` | Compare models and choose epochs, learning rate and data mix |
| `calibration.jsonl` | Fit temperature/confidence parameters |
| `test.jsonl` | Final check after tuning decisions are complete |

The referenced Kev guidance recommends at least 40 records in each calibration/development partition and roughly 100 calibration questions for a stable temperature fit. Multiple tasks from one note are not independent clinical cases, so those counts alone do not establish representative coverage.

Every routing question has a one-hot `target`. Under the referenced default cross-entropy loss, that has the same loss as the hard label. It also prevents default augmentation from deleting the correct option or inserting a competing none option; Kev permutes the choices instead. The earlier pilot's offline simulation observed about 40% of questions altered without targets and none with them. These are version-specific observations. Soft-target handling bypasses some optional loss modifiers; inspect the actual training code before using those options.

### Compare the models fairly

Use the same held-out cases, release, hierarchy, controller, candidate policy and branch-search budget for default and fine-tuned checkpoints.

Report both:

1. **Reference-parent task accuracy:** can the model make the right choice when given the correct parent?
2. **End-to-end coding:** can the controller and model start at ROOT and finish with the right code?

The first is useful but easier. It cannot show whether an earlier mistake would prevent the model reaching the right parent.

Measure complete-code Top-1. Report Top-5 only when the controller actually returns five candidates. Also examine accuracy by depth, correct-path retention, first error/stop position, outcome confusion matrices, evidence precision/recall, and total input tokens and latency per case. Keep mapped-code results separate so frequent STOP or NO_MATCH tasks do not dominate the score.

Use case-level results and uncertainty estimates where possible. Thousands of correlated task rows do not provide thousands of independent evaluation cases. Per-code results will be noisy when only a few cases represent a code.

### Test shorter inputs separately

To test input efficiency, compare both checkpoint and option verbosity:

| | Described options | Compact options |
|---|---|---|
| Default Kev | Baseline with explanations | Checks whether default already needs less context |
| Fine-tuned Kev | Tests domain adaptation | Tests the benefit of training under reduced context |

Keep cases, labels and splits matched. Generic compact data exists, but a verified compact-native experiment still needs the corresponding adapter work. Comparing only detailed-default against compact-fine-tuned would mix two changes.

Measure the total across the route. Shorter individual prompts need not reduce total cost or latency if many branches or retries are required.

### Treat confidence as an output to test

Use the option probabilities returned by the actual Kev scoring interface, not a percentage generated as prose. They are conditional on that question's options. Scores from different option sets are not automatically comparable.

A product of conditional path probabilities can be investigated, but it is not automatically an empirically calibrated final-code probability. Validate the complete controller and scoring procedure. Do not fit temperature on development or test, and resolve the family overlap described below before treating calibration and development as independent.

A pipeline smoke test shows that files flow through the tools. Paired model evaluation can show learning. Independently reviewed real or clinician-authored notes are needed to investigate transfer. None of these alone establishes safe deployment.

## The trial dataset

This section preserves the recorded snapshot inspected on 6 November 2026. It is not live status. Recalculate after another run.

The 1,500-slot trial accepted 1,460 synthetic notes. The full plan remained 4,500 cases, so finalization marked the snapshot `PARTIAL`.

| Partition | Cases | Mapped / out-of-scope / review | Mapped pilot codes | Native task rows | Native questions |
|---|---:|---|---:|---:|---:|
| Train | 952 | 786 / 89 / 77 | 50/50 | 6,965 | 9,222 |
| Validation master | 176 | 145 / 20 / 11 | 48/50 | 1,294 | 1,749 |
| ↳ Calibration | 79 | 62 / 13 / 4 | 42/50 | 555 | 748 |
| ↳ Development | 97 | 83 / 7 / 7 | 42/50 | 739 | 1,001 |
| Test | 332 | 272 / 33 / 27 | 50/50 | 2,414 | 3,247 |
| **Total** | **1,460** | **1,203 / 142 / 115** | **50/50** | **10,673** | **14,218** |

Calibration and development are subdivisions of master validation; do not add them again to the total. Counts were calculated using case IDs in the master and native exports.

### What happened to the 1,500 slots?

Of 1,500 slot-level generation attempts, 1,480 produced candidates, 15 were content-filtered and five were invalid. Review accepted 1,460, rejected 20 and left none pending. The manifest reports no estimated-size skips.

These observations do not establish why every provider response failed or guarantee a representative sample. Generator and reviewer deployment IDs were both `gpt-6.1-sol`. That is not independent confirmation of the underlying model or an independent clinical review.

### Coverage and overlap

All 50 target codes have mapped training examples, with 8–24 per code. Master validation covers 48 codes, missing `E11.29` and `N18.30`. Development additionally lacks `E11.21`, `J45.30`, `K26.9`, `K29.71`, `N18.1` and `N39.0`. Represented codes have 1–7 cases in master validation, 1–5 in development, and 2–10 in test.

No case IDs overlap the native partitions. However, 19 scenario families span calibration and development because those sets use case hashes, not family hashes. Treat the two sets as correlated until a family-isolated split is made. Master-level family isolation does not remove this issue.

Of 1,203 mapped cases, 1,124 have one annotated focus sentence, although every mapped note has multiple sentences. Native train contains 832 positive evidence questions out of 3,043. Seven distinct normalized exact first sentences repeat across master records; that narrow check is not a semantic template audit.

Saved audits report zero master cross-split lexical duplicates and family violations. They do not establish semantic independence or clinical correctness.

### What has actually been checked?

| Check | What it establishes | What it does not establish |
|---|---|---|
| Schema and spans | Valid IDs, positions and exact quotes | Correct clinical attribution |
| Codebook and paths | Agreement with the selected references and manifest | Independent confirmation of all source interpretations |
| Export labels | Labels belong to options; one-hot targets agree | No remaining metadata shortcuts |
| Split and lexical duplicate checks | Certain overlaps detected/blocked | Semantic independence; native calibration/development family isolation |
| Code-leakage checks | Dotted/undotted pilot identifiers checked | Absence of semantic shortcuts |
| Blind model re-code | Agreement under the configured review procedure | Human review or a complete rule/evidence audit |
| Local Kev checker | Per-file format, duplicate and conflict checks | Exact tokenizer limits or successful fine-tuning |

The snapshot has no independent human clinical review, real-note evaluation, training result or model-performance comparison. Exact tokenizer validation remains unperformed for this snapshot; an earlier 61-case pilot had separate loader/tokenizer checks. Do not transfer that earlier result to these files.

### Source provenance

The trial's source lock records the following SHA-256 values:

| Source ID | Recorded SHA-256 |
|---|---|
| `table_and_index` | `99b5c2be16ff53eab861cc0be4b1dacba27f687b8ff4bc700679944355037ee0` |
| `code_descriptions` | `93e3ad6004badf470c55bfe679b748ae88fd9b2b421851e409eec382c7713b9a` |
| `guidelines` | `1f0065b609b678e98f85bf047b3b012c6a88ba3222e42be2f2435889e05738ab` |
| `addenda` | `54067ceae07e6472e7162b84d3e75d6d5a5de9e344870a8573029b0b8778388e` |

Combined manifest hash:

```text
55683046cbafa0cfd1d3c9572ffb3e09a2eebb3725c938089e25bbd49792f55d
```

The uploaded README records that the four local files were rehashed on 7 November 2026 and matched the lock. The optional tabular PDF was not imported in that lock.

The existing audit leaves independent release verification of the guidelines PDF unresolved: filename/member metadata alone was not sufficient, although its content parsed. Keep that caveat with the snapshot. Matching hashes identify the files used; they do not resolve the release question. Generated notes are fictional, not CDC patient records.

## Design decisions still to resolve

The project is ready for a trial, but several choices affect what a later model comparison can claim. These are follow-up work, not features added by this README.

### Candidate selection without knowing the answer

The exporter keeps the gold child and hash-ranks other siblings. The default 24-option limit includes three controls, leaving 21 chapter slots at ROOT. The recorded 1,460 ROOT tasks therefore use 22 different chapter sets. Every row contains its correct answer.

That is a training construction policy, not a runtime controller policy. Reference-parent evaluation can disclose that it uses answer-retaining candidates. Real inference cannot use the gold answer to decide what to include.

Where token limits permit, consider complete direct-child sets. ROOT would have 25 options with all chapters and the three controls. Other branches need their own checks; a pilot branch has 24 children. Increasing a limit requires configuration changes and exact tokenizer validation, not just a documentation edit. Chunking and renormalizing options is only a possible design, not an implemented or validated solution.

### Keep related calibration cases together

Assigning whole scenario families to calibration or development would reduce their dependence. Recheck code/outcome coverage afterwards. This can be a repartitioning/export change rather than a reason to regenerate every note.

### Extend stopping and partial routes

Decide how to train stopping at a reportable node with descendants, where applicable. Also decide whether review/outside-pilot cases should retain supported ancestors instead of stopping at ROOT. The present training tasks do not establish these broader abilities.

### Agree on evidence and scenario policies

Clarify whether a positive sentence explicitly states the diagnosis, supplies a required coding attribute, or merely describes related context. Audit the focus-to-evidence conversion under that definition. Add clinical contrast coverage rather than relying only on style/length variation.

### Add compact-native evaluation if input efficiency is a goal

The current native export contains only described options. Add and validate a compact-native variant before claiming the four-way comparison is implemented.

| Priority | Next work |
|---|---|
| Before a definitive comparison | Specify gold-independent candidates, check exact token limits, isolate calibration/development families |
| After initial model errors are available | Refine stopping, partial review routes and evidence annotations |
| Before the next generation round | Plan clinical contrasts and audit scenario/template reuse |
| Before an efficiency claim | Implement compact-native exports and measure total route cost/latency |
| Before considering real use | Independent qualified review and appropriately governed external evaluation |

The export manifest's compatibility wording is generic exporter text, not a substitute for a snapshot-specific validator or tokenizer report.

## Tests

Run the offline suite without an API key:

```bash
PYTHONPATH=src python -m pytest tests -q
```

Focused tests include [test_pipeline.py](tests/test_pipeline.py), [test_kb.py](tests/test_kb.py), [test_planning.py](tests/test_planning.py), and [test_sources.py](tests/test_sources.py). Fake clients test retries, partial batches, leakage, review agreement, duplicates, recovery, locking, finalization and Kev tasks. They do not establish live Azure compatibility.

The [official-source integration test](tests/integration/test_official_sources.py) needs actual CDC files. Set `ICD_OFFICIAL_SOURCE_DIR` to a directory containing the four files with exactly the configured names. Without them it is skipped, not passed as evidence of source compatibility.

This README describes the uploaded implementation and recorded reports. Editing the document does not rerun those checks. Revalidate the current code, source snapshot and exports before a new experiment.

## License

Project code and documentation use the [MIT License](LICENSE). Official references and other third-party materials are not covered by that license. Check their terms before redistributing them.
