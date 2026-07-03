# A1 — Case Study Brief: HealthGuard

**For project:** Operation HealthGuard — AI Model Risk & Compliance Assessment

**Audience:** You, in the role of Chief AI Risk Officer at Klariova Health

**Status:** Pre-launch — model is ready for production but cannot be deployed without your sign-off

---

## 1. Company Context

**Klariova Health** is a 600-person digital-health company headquartered in Dublin, Ireland with a US subsidiary in Boston, Massachusetts. Founded in 2019, Klariova Health builds clinical decision-support tools licensed to hospital networks and large primary-care groups. The company's mission is "to make preventive care decisions faster and more accurate at the point of care." Annual revenue is approximately €120 million; the company is venture-backed and preparing for a Series D round in late 2027.

The company has three product lines: a sepsis early-warning module (in production since 2022), a discharge-risk classifier (in production since 2023), and **HealthGuard** — the diabetes-risk prediction system that is the subject of this assessment. HealthGuard is the first Klariova Health product to be classified under the EU AI Act and is the first one that will deploy in both EU and US markets simultaneously.

---

## 2. System Purpose and Intended Use

**HealthGuard** is a clinical decision-support tool that predicts a patient's risk of developing type-2 diabetes based on data available in the electronic health record (EHR). The model produces a probability score and a binary high-risk / not-high-risk classification.

**Intended use.** Clinicians use the score during primary-care visits to inform preventive care — flagging patients who may benefit from earlier lab testing, lifestyle counseling, or referral to a diabetes-prevention program.

**HealthGuard is a decision-support tool. It does not make autonomous decisions.** A clinician reviews every prediction before any clinical action is taken.

---

## 3. Non-Intended Uses

The following uses are **explicitly prohibited** by Klariova Health's contract terms with hospital licensees and must not be supported by your launch recommendation:

- Insurance underwriting, pricing, or coverage decisions.
- Automated denial of care, referral, or coverage.
- Direct patient-facing communication of risk scores without clinician interpretation.
- Employment screening, occupational health screening, or any non-clinical use.
- Use outside primary-care or endocrinology contexts (e.g., emergency department, ICU).
- Any use that would constitute a regulated medical device decision under MDR 2017/745 (HealthGuard is positioned as decision-support, not as a diagnostic device).

---

## 4. Deployment Regions and Regulatory Jurisdictions

| Region | Status | Primary regulatory regimes that apply |
|---|---|---|
| European Union (EU) | Primary launch — Q3 2027 | EU AI Act (post-staggered-enforcement); GDPR; national medical-device frameworks where applicable |
| United States (US) | Secondary launch — Q4 2027 | HIPAA; CPRA (California-resident data); FDA AI/ML Action Plan considerations; OCR guidance on AI in healthcare |

**EU launch hospitals:** four academic medical centers in Ireland, Germany, the Netherlands, and Sweden.
**US launch hospitals:** three integrated delivery networks in California, Massachusetts, and Texas.

EU patient data is processed in Klariova Health's Frankfurt data center; US patient data is processed in a separate AWS US-East region. **Some inference traffic from US hospitals routes through the EU data center for model-monitoring telemetry** — this is a deliberate design choice that will be relevant in your data-flow analysis.

---

## 5. System Architecture (Named Components)

The HealthGuard system is composed of the following named components. Use these names in your threat model and data-flow analysis — they are the specific attack surfaces and compliance touch-points.

**Component 1 — Data Ingestion Pipeline.** Receives patient records from hospital EHR systems via HL7 FHIR R4 feeds. Authentication is mutual-TLS at the hospital edge. Data is buffered in a Kafka topic before being normalized.

**Component 2 — Preprocessing Layer.** Performs feature engineering (BMI computation, age normalization, missing-value imputation, demographic enrichment from hospital registration records). Joins clinical data with patient demographics including the `insurance_type` field used by the model.

**Component 3 — Model Inference API.** A RESTful FastAPI service that loads the trained model artifact (a scikit-learn pipeline serialized with joblib), accepts a feature vector, and returns the risk probability and binary classification. Requests are rate-limited at 200 requests per minute per hospital.

**Component 4 — Clinician-Facing Dashboard.** A web application clinicians use during patient visits. Displays the risk score, top contributing features (a rough explanation, not full SHAP), patient history, and recommended next actions. Hosts the human-in-the-loop sign-off step.

**Component 5 — Model Registry.** An internal service that stores model artifacts, training-data manifests, and deployment metadata. Contains every model version ever deployed plus the training datasets they were trained on.

**Component 6 — Logging and Monitoring Infrastructure.** Captures every inference request, response, clinician override, and feature-drift metric. Logs are retained for 7 years to support post-market surveillance obligations under EU AI Act Article 72.

---

## 6. Data Flow Description

Patient data flows through the system as follows. Use this narrative when interpreting the pre-drawn data flow diagram (artifact A6):

1. **Origination.** Patient demographic and clinical data originates in the hospital EHR. For EU hospitals, this includes patients across all EU member states whose records may pass through cross-border patient registries.
2. **Transmission.** Records are pushed to Klariova Health via HL7 FHIR feeds over an mTLS-authenticated channel. Hospitals push every 15 minutes; there is no patient consent capture inside Klariova Health's perimeter — consent management is the hospital's responsibility per the data-processing agreement.
3. **Preprocessing.** Demographics are joined with clinical data. The `insurance_type` field is sourced from the hospital's billing system and may differ from the patient's stated coverage if a billing change has not yet been applied.
4. **Inference.** The Model Inference API computes the risk score using all features (clinical + demographic + insurance_type).
5. **Storage.** Predictions, feature vectors, and patient identifiers are stored in the prediction database for 7 years (post-market surveillance retention). Storage location is the regional data center (EU patients → Frankfurt; US patients → AWS US-East).
6. **Display.** The Clinician Dashboard pulls the prediction by patient ID at the point of care.
7. **Feedback.** Clinician overrides (the clinician disagrees with the score) are logged back to the Model Registry to support future retraining.

**Cross-border flow that you should pay attention to:** US inference telemetry (request metadata, *not* full patient records) is sent to the EU data center for model-drift monitoring. You will need to assess whether this constitutes a transfer of personal data under GDPR, and what mechanism (SCCs, Privacy Shield successor framework, adequacy) applies.

---

## 7. Risk Appetite Statement

The Board of Directors of Klariova Health has approved the following risk appetite for the HealthGuard product line. Anchor your launch recommendation against these statements:

> **Patient-safety risk:** Low appetite. We will not deploy a system whose foreseeable failure modes can produce a missed diagnosis or delayed referral for any identifiable patient subgroup at a rate materially worse than standard of care.
>
> **Regulatory non-compliance:** Low appetite. We will not deploy a system that we know to be non-compliant with the EU AI Act, GDPR, HIPAA, or CPRA. We will deploy a system with documented partial compliance only when the gap has a remediation plan and an approved compensating control.
>
> **Operational risk:** Moderate appetite. We accept reasonable infrastructure availability risks (e.g., a single-region outage tolerated up to 4 hours).
>
> **Reputational risk:** Low appetite for any incident that would be reported to a regulator or attract press coverage.
>
> **Financial risk:** Moderate appetite, bounded by the product line's annual operating loss limit.

---

## 8. Model Provenance and Training-Data Sourcing

This section is where the supply-chain story lives. You will reference it in the Risk Register (Task 5) and the Model Card (Task 6).

**Model lineage.**
- The HealthGuard model is a gradient-boosted tree classifier (scikit-learn 1.7) trained in-house by the Klariova Health ML team.
- The training framework is open-source. There is no third-party model component (no pretrained foundation model, no transferred weights from an external source).

**Training data.**
- Source 1 — anonymized EHR extracts from two of Klariova Health's existing customer hospitals (Ireland, Germany), licensed under the data-processing agreement with explicit opt-in for secondary use in model development.
- Source 2 — a synthetic dataset generated to balance demographic representation, in particular to ensure adequate sample sizes for non-majority geographic regions (rural and suburban). The synthetic data was produced by an internal data-engineering team using a generative model trained on the Source 1 data.
- Total training set: approximately 12,000 patient records.

**Known training-data limitations.**
- The synthetic-data generator (Source 2) was trained on data that itself reflects historical care-access disparities — patients from under-served populations are known to be diagnosed at lower rates because they receive fewer encounters with primary care. Any pattern present in Source 1 — including under-diagnosis — is reproduced in Source 2.
- The `insurance_type` field is sourced from hospital billing systems, which may differ from a patient's actual coverage at the moment of care.
- The geographic distribution of the training data (Ireland + Germany) is not representative of the planned US deployment population.

**Model-card precursor information.**
- Model version under assessment: `healthguard-v1.0-rc7`.
- Training run: 2027-03-12.
- Training-data manifest hash: `sha256:7f3c…a82e` (recorded in the Model Registry).
- The model has *not* yet been independently validated by an external party.

---

## 9. What You Have Been Asked to Produce

The Chief Executive Officer and the Chair of the AI Review Board have asked you, as Chief AI Risk Officer, to produce a complete pre-launch GRC review of HealthGuard. The deliverables you produce will be reviewed in a launch decision meeting on the date specified by your project instructions. Your launch recommendation will be one of:

- **Go** — model launches as currently configured.
- **Conditional Go** — model launches subject to specific, measurable conditions being met first.
- **No Go** — model does not launch in its current form.

Work through Tasks 1 through 8 in your project instructions. Each task produces an artifact that feeds the final Launch Decision & Executive Brief.

Good luck. Be rigorous, be specific, and remember that your job is not to gate-keep innovation — it is to make sure the system you ship is one Klariova Health can stand behind.

— *Klariova Health Office of the Chief AI Risk Officer*
