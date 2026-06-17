import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft, ArrowRight, FileText, Handshake, Home, Users,
  ShieldCheck, Briefcase, Scale, PenLine, AlertCircle,
  CheckCircle2, Copy, Download, Sparkles, Check,
} from "lucide-react";
import { generateDocument } from "../api";
import "./GenerateDocumentPage.css";

// ── Document types ──────────────────────────────────────────────
const DOC_TYPES = [
  { id: "nda",               label: "NDA",                icon: ShieldCheck },
  { id: "employment",        label: "Employment Contract", icon: Briefcase },
  { id: "lease",             label: "Lease Agreement",    icon: Home },
  { id: "service",           label: "Service Agreement",  icon: Handshake },
  { id: "partnership",       label: "Partnership Deed",   icon: Users },
  { id: "power_of_attorney", label: "Power of Attorney",  icon: PenLine },
  { id: "affidavit",         label: "Affidavit",          icon: Scale },
  { id: "legal_notice",      label: "Legal Notice",       icon: AlertCircle },
  { id: "custom",            label: "Custom Document",    icon: FileText },
];

const JURISDICTIONS = [
  "India", "United States", "United Kingdom", "Canada", "Australia",
  "UAE", "Singapore", "Germany", "France", "Other",
];

const TONES = [
  { value: "formal",       label: "Formal (Standard Legal)" },
  { value: "semi_formal",  label: "Semi-Formal (Balanced)" },
  { value: "plain",        label: "Plain Language (Readable)" },
];

// ── Step labels ─────────────────────────────────────────────────
const STEPS = ["Document Type", "Parties & Context", "Details", "Review & Generate"];



interface FormData {
  docType: string;
  partyA: string;
  partyB: string;
  context: string;
  jurisdiction: string;
  tone: string;
  specialClauses: string;
  // type-specific
  duration?: string;
  scope?: string;
  propertyAddress?: string;
  rent?: string;
  salary?: string;
  startDate?: string;
}

export default function GenerateDocumentPage() {
  const navigate = useNavigate();
  const [step, setStep]         = useState(0);
  const [dir, setDir]           = useState(1);   // 1 = forward, -1 = backward
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState("");
  const [result, setResult]     = useState("");
  const [copied, setCopied]     = useState(false);

  const [form, setForm] = useState<FormData>({
    docType:       "",
    partyA:        "",
    partyB:        "",
    context:       "",
    jurisdiction:  "India",
    tone:          "formal",
    specialClauses: "",
  });

  // ── Helpers ──────────────────────────────────────────────────
  const update = (key: keyof FormData, val: string) =>
    setForm(prev => ({ ...prev, [key]: val }));

  const goNext = () => { setDir(1); setStep(s => s + 1); };
  const goPrev = () => { setDir(-1); setStep(s => s - 1); };

  const canNext = () => {
    if (step === 0) return form.docType !== "";
    if (step === 1) return form.partyA.trim() !== "" && form.context.trim() !== "";
    return true;
  };

  const selectedDoc = DOC_TYPES.find(d => d.id === form.docType);

  // ── Generate ─────────────────────────────────────────────────
  async function handleGenerate() {
    setError("");
    setLoading(true);
    try {
      const data = await generateDocument(form);
      setResult(data.document_text);
    } catch (e: any) {
      setError(e.message || "Failed to generate document. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  // ── Copy ─────────────────────────────────────────────────────
  function handleCopy() {
    navigator.clipboard.writeText(result).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  // ── Download ─────────────────────────────────────────────────
  function handleDownload() {
    const blob = new Blob([result], { type: "text/plain" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `${selectedDoc?.label ?? "document"}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // ── Step 1 — Type selection ───────────────────────────────────
  const StepType = () => (
    <div className="doc-type-grid">
      {DOC_TYPES.map(({ id, label, icon: Icon }) => (
        <div
          key={id}
          className={`doc-type-tile ${form.docType === id ? "selected" : ""}`}
          onClick={() => update("docType", id)}
          role="button"
          tabIndex={0}
          onKeyDown={e => e.key === "Enter" && update("docType", id)}
        >
          <div className="tile-icon"><Icon size={18} /></div>
          <span className="tile-label">{label}</span>
        </div>
      ))}
    </div>
  );

  // ── Step 2 — Parties & Context ────────────────────────────────
  const StepParties = () => {
    const isLease      = form.docType === "lease";
    const isEmployment = form.docType === "employment";
    const isNDA        = form.docType === "nda";

    return (
      <div className="form-grid">
        <div className="form-group">
          <label className="form-label">
            {isEmployment ? "Employer Name" : "Party A / First Party"}
          </label>
          <input
            id="input-party-a"
            className="form-input"
            placeholder={isEmployment ? "e.g. Acme Corp Pvt. Ltd." : "e.g. ABC Technologies"}
            value={form.partyA}
            onChange={e => update("partyA", e.target.value)}
          />
        </div>

        <div className="form-group">
          <label className="form-label">
            {isEmployment ? "Employee Name" : "Party B / Second Party"}
          </label>
          <input
            id="input-party-b"
            className="form-input"
            placeholder={isEmployment ? "e.g. Rahul Sharma" : "e.g. XYZ Solutions"}
            value={form.partyB}
            onChange={e => update("partyB", e.target.value)}
          />
        </div>

        {isLease && (
          <>
            <div className="form-group">
              <label className="form-label">Property Address</label>
              <input
                id="input-property"
                className="form-input"
                placeholder="e.g. 42 MG Road, Bangalore 560001"
                value={form.propertyAddress ?? ""}
                onChange={e => update("propertyAddress", e.target.value)}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Monthly Rent</label>
              <input
                id="input-rent"
                className="form-input"
                placeholder="e.g. ₹25,000 per month"
                value={form.rent ?? ""}
                onChange={e => update("rent", e.target.value)}
              />
            </div>
          </>
        )}

        {isEmployment && (
          <>
            <div className="form-group">
              <label className="form-label">Monthly Salary / CTC</label>
              <input
                id="input-salary"
                className="form-input"
                placeholder="e.g. ₹80,000 per month"
                value={form.salary ?? ""}
                onChange={e => update("salary", e.target.value)}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Start Date</label>
              <input
                id="input-start-date"
                type="date"
                className="form-input"
                value={form.startDate ?? ""}
                onChange={e => update("startDate", e.target.value)}
              />
            </div>
          </>
        )}

        {isNDA && (
          <>
            <div className="form-group">
              <label className="form-label">Confidentiality Duration</label>
              <input
                id="input-duration"
                className="form-input"
                placeholder="e.g. 2 years from effective date"
                value={form.duration ?? ""}
                onChange={e => update("duration", e.target.value)}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Scope of Confidentiality</label>
              <input
                id="input-scope"
                className="form-input"
                placeholder="e.g. Business strategies, trade secrets"
                value={form.scope ?? ""}
                onChange={e => update("scope", e.target.value)}
              />
            </div>
          </>
        )}

        <div className="form-group span-2">
          <label className="form-label">Purpose / Context</label>
          <textarea
            id="textarea-context"
            className="form-textarea"
            rows={3}
            placeholder="Briefly describe the purpose of this document and any important background context…"
            value={form.context}
            onChange={e => update("context", e.target.value)}
          />
        </div>
      </div>
    );
  };

  // ── Step 3 — Details ──────────────────────────────────────────
  const StepDetails = () => (
    <div className="form-grid">
      <div className="form-group">
        <label className="form-label">Jurisdiction / Country</label>
        <select
          id="select-jurisdiction"
          className="form-select"
          value={form.jurisdiction}
          onChange={e => update("jurisdiction", e.target.value)}
        >
          {JURISDICTIONS.map(j => (
            <option key={j} value={j}>{j}</option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label className="form-label">Document Tone</label>
        <select
          id="select-tone"
          className="form-select"
          value={form.tone}
          onChange={e => update("tone", e.target.value)}
        >
          {TONES.map(t => (
            <option key={t.value} value={t.value}>{t.label}</option>
          ))}
        </select>
      </div>

      <div className="form-group span-2">
        <label className="form-label">Special Clauses / Instructions</label>
        <textarea
          id="textarea-special"
          className="form-textarea"
          rows={4}
          placeholder="Any specific clauses, limitations, or legal instructions you want included…"
          value={form.specialClauses}
          onChange={e => update("specialClauses", e.target.value)}
        />
      </div>
    </div>
  );

  // ── Step 4 — Review ───────────────────────────────────────────
  const StepReview = () => (
    <>
      <div className="review-grid">
        <div className="review-item">
          <div className="review-label">Document Type</div>
          <div className="review-value">{selectedDoc?.label ?? "—"}</div>
        </div>
        <div className="review-item">
          <div className="review-label">Jurisdiction</div>
          <div className="review-value">{form.jurisdiction}</div>
        </div>
        <div className="review-item">
          <div className="review-label">Party A</div>
          <div className="review-value">{form.partyA || "—"}</div>
        </div>
        <div className="review-item">
          <div className="review-label">Party B</div>
          <div className="review-value">{form.partyB || "—"}</div>
        </div>
        <div className="review-item">
          <div className="review-label">Tone</div>
          <div className="review-value">
            {TONES.find(t => t.value === form.tone)?.label ?? "—"}
          </div>
        </div>
        {form.duration && (
          <div className="review-item">
            <div className="review-label">Duration</div>
            <div className="review-value">{form.duration}</div>
          </div>
        )}
        <div className="review-item span-2">
          <div className="review-label">Context</div>
          <div className="review-value">{form.context || "—"}</div>
        </div>
        {form.specialClauses && (
          <div className="review-item span-2">
            <div className="review-label">Special Clauses</div>
            <div className="review-value">{form.specialClauses}</div>
          </div>
        )}
      </div>

      {error && (
        <div className="generate-error">
          <AlertCircle size={16} /> {error}
        </div>
      )}

      {result && (
        <div className="result-container">
          <div className="result-header">
            <div className="result-title">
              <CheckCircle2 size={15} /> Generated Document
            </div>
            <div className="result-actions">
              <button
                id="btn-copy-result"
                className="result-btn btn-copy"
                onClick={handleCopy}
              >
                {copied ? <><Check size={13} /> Copied!</> : <><Copy size={13} /> Copy</>}
              </button>
              <button
                id="btn-download-result"
                className="result-btn btn-download"
                onClick={handleDownload}
              >
                <Download size={13} /> Download
              </button>
            </div>
          </div>
          <div className="result-body">
            <pre className="result-text">{result}</pre>
          </div>
        </div>
      )}
    </>
  );

  const stepComponents = [<StepType />, <StepParties />, <StepDetails />, <StepReview />];
  const stepDescriptions = [
    "Choose the type of legal document you want to generate.",
    "Enter the parties involved and describe the purpose of this document.",
    "Specify jurisdiction, preferred tone, and any special requirements.",
    "Review your inputs and generate the document.",
  ];

  return (
    <div className="generate-page">
      {/* ── Top Bar ── */}
      <div className="generate-topbar">
        <button
          id="btn-back-generate"
          className="back-btn"
          onClick={() => navigate("/")}
        >
          <ArrowLeft size={15} /> Dashboard
        </button>
        <div className="topbar-title">
          <Sparkles size={16} style={{ color: "#e8b84b" }} />
          Generate Legal Document
          <span className="topbar-step-label">· Step {step + 1} of {STEPS.length}</span>
        </div>
        <div style={{ width: 110 }} />
      </div>

      {/* ── Progress Steps ── */}
      <div className="generate-progress">
        <div className="steps-track">
          {STEPS.map((label, i) => (
            <>
              <div className="step-item" key={label}>
                <div className={`step-circle ${i === step ? "active" : i < step ? "completed" : ""}`}>
                  {i < step ? <Check size={14} /> : i + 1}
                </div>
                <span className={`step-label ${i === step ? "active" : i < step ? "completed" : ""}`}>
                  {label}
                </span>
              </div>
              {i < STEPS.length - 1 && (
                <div key={`conn-${i}`} className={`step-connector ${i < step ? "filled" : ""}`} />
              )}
            </>
          ))}
        </div>
      </div>

      {/* ── Form Body ── */}
      <div className="generate-body">
        <div className="generate-card">
          <div className="step-heading">
            <h2>{STEPS[step]}</h2>
            <p>{stepDescriptions[step]}</p>
          </div>

          <AnimatePresence mode="wait" custom={dir}>
            <motion.div
              key={step}
              custom={dir}
              variants={{
                enter:  (d: number) => ({ opacity: 0, x: d * 40 }),
                center: { opacity: 1, x: 0, transition: { duration: 0.32, ease: [0.22, 1, 0.36, 1] } },
                exit:   (d: number) => ({ opacity: 0, x: d * -40, transition: { duration: 0.22, ease: "easeIn" } }),
              }}
              initial="enter"
              animate="center"
              exit="exit"
            >
              {stepComponents[step]}
            </motion.div>
          </AnimatePresence>

          {/* ── Nav Buttons ── */}
          <div className="step-nav">
            <button
              id="btn-step-prev"
              className="btn-prev"
              onClick={goPrev}
              disabled={step === 0}
            >
              <ArrowLeft size={15} /> Back
            </button>

            {step < STEPS.length - 1 ? (
              <button
                id="btn-step-next"
                className="btn-next"
                onClick={goNext}
                disabled={!canNext()}
              >
                Continue <ArrowRight size={15} />
              </button>
            ) : (
              <button
                id="btn-generate-submit"
                className={`btn-next ${loading ? "generating" : ""}`}
                onClick={handleGenerate}
                disabled={loading || result !== ""}
              >
                {loading ? (
                  <><div className="spinner" /> Generating…</>
                ) : result ? (
                  <><CheckCircle2 size={15} /> Generated!</>
                ) : (
                  <><Sparkles size={15} /> Generate Document</>
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
