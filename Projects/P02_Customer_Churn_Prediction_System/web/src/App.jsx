import { useState } from "react";
import evaluation from "../../reports/evaluation.json";
import metadata from "../../models/feature_metadata.json";
import importanceCsv from "../../reports/feature_importance.csv?raw";
import rocCurve from "../../reports/roc_curve.png";

const navigation = [
  { id: "overview", label: "Overview", icon: "grid" },
  { id: "assessment", label: "Customer assessment", icon: "user" },
  { id: "performance", label: "Model performance", icon: "chart" },
];

const fieldGroups = [
  {
    title: "Account details",
    description: "Customer tenure and billing preferences",
    fields: [
      "tenure",
      "Contract",
      "PaperlessBilling",
      "PaymentMethod",
      "SeniorCitizen",
      "Partner",
      "Dependents",
      "gender",
    ],
  },
  {
    title: "Services",
    description: "Current phone, internet, and add-on services",
    fields: [
      "PhoneService",
      "MultipleLines",
      "InternetService",
      "OnlineSecurity",
      "OnlineBackup",
      "DeviceProtection",
      "TechSupport",
      "StreamingTV",
      "StreamingMovies",
    ],
  },
  {
    title: "Monthly charges",
    description: "Current recurring and accumulated charges",
    fields: ["MonthlyCharges", "TotalCharges"],
  },
];

const featureImportance = importanceCsv
  .trim()
  .split(/\r?\n/)
  .slice(1)
  .map((row) => {
    const [feature, mean, deviation] = row.split(",");
    return {
      feature,
      mean: Number(mean),
      deviation: Number(deviation),
    };
  })
  .filter((item) => item.mean > 0)
  .sort((a, b) => b.mean - a.mean)
  .slice(0, 7);

const initialCustomer = Object.fromEntries(
  Object.entries(metadata.reference_values).map(([field, value]) => [
    field,
    Number.isInteger(value) ? String(value) : String(value),
  ]),
);

function Icon({ name, size = 18 }) {
  const paths = {
    grid: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
      </>
    ),
    user: (
      <>
        <circle cx="12" cy="8" r="3.5" />
        <path d="M5 20c.7-3.1 3.2-5 7-5s6.3 1.9 7 5" />
      </>
    ),
    chart: (
      <>
        <path d="M4 19V5M4 19h17" />
        <path d="m7 15 4-4 3 2 6-7" />
      </>
    ),
    arrow: <path d="M5 12h14m-6-6 6 6-6 6" />,
    chevron: <path d="m9 18 6-6-6-6" />,
    spark: (
      <>
        <path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" />
        <path d="m19 15 .9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9L19 15Z" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    info: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v5m0-8h.01" />
      </>
    ),
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
  };

  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[name]}
    </svg>
  );
}

function formatLabel(value) {
  return value.replace(/([a-z])([A-Z])/g, "$1 $2");
}

function demoProbability(customer) {
  let score = 0.13;
  const tenure = Number(customer.tenure);
  const monthlyCharges = Number(customer.MonthlyCharges);

  if (customer.Contract === "Month-to-month") score += 0.25;
  if (customer.Contract === "Two year") score -= 0.12;
  if (tenure < 12) score += 0.2;
  else if (tenure > 48) score -= 0.1;
  if (customer.InternetService === "Fiber optic") score += 0.07;
  if (customer.OnlineSecurity === "No") score += 0.06;
  if (customer.TechSupport === "No") score += 0.05;
  if (monthlyCharges > 85) score += 0.07;
  if (customer.PaperlessBilling === "Yes") score += 0.025;

  return Math.min(0.94, Math.max(0.04, score));
}

function MetricCard({ label, value, note, icon, tone }) {
  return (
    <article className="metric-card">
      <div className={`metric-icon ${tone}`}>
        <Icon name={icon} />
      </div>
      <div className="metric-copy">
        <span className="metric-label">{label}</span>
        <strong>{value}</strong>
        <span className="metric-note">{note}</span>
      </div>
    </article>
  );
}

function PageHeader({ eyebrow, title, description, action }) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </div>
  );
}

function Overview({ onAssess, onPerformance }) {
  const metricCards = [
    {
      label: "Holdout accuracy",
      value: `${(evaluation.accuracy * 100).toFixed(1)}%`,
      note: `${evaluation.test_rows.toLocaleString()} customers evaluated`,
      icon: "check",
      tone: "mint",
    },
    {
      label: "Churn recall",
      value: `${(evaluation.recall * 100).toFixed(1)}%`,
      note: "Customers correctly identified",
      icon: "user",
      tone: "lavender",
    },
    {
      label: "ROC-AUC",
      value: evaluation.roc_auc.toFixed(3),
      note: "Ranking performance",
      icon: "chart",
      tone: "peach",
    },
  ];

  return (
    <>
      <PageHeader
        eyebrow="CUSTOMER INTELLIGENCE"
        title="Churn overview"
        description="A clear view of model performance and the customers who may need attention."
        action={
          <button className="button button-primary" onClick={onAssess}>
            New assessment <Icon name="arrow" size={17} />
          </button>
        }
      />

      <section className="welcome-card">
        <div className="welcome-content">
          <span className="welcome-kicker">
            <Icon name="spark" size={16} /> RETENTION WORKSPACE
          </span>
          <h2>Make every customer<br />conversation count.</h2>
          <p>
            Explore churn signals and understand how your model supports better
            retention decisions.
          </p>
          <button className="button button-light" onClick={onAssess}>
            Assess a customer <Icon name="arrow" size={17} />
          </button>
        </div>
        <div className="welcome-art" aria-hidden="true">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="orbit-core"><Icon name="user" size={31} /></div>
          <span className="orbit-dot dot-one" />
          <span className="orbit-dot dot-two" />
          <span className="orbit-dot dot-three" />
        </div>
        <div className="welcome-model">
          <span className="status-dot" />
          <span>Selected model</span>
          <strong>{evaluation.model}</strong>
        </div>
      </section>

      <div className="metric-grid">
        {metricCards.map((metric) => <MetricCard key={metric.label} {...metric} />)}
      </div>

      <div className="content-grid">
        <section className="panel importance-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">MODEL INSIGHTS</span>
              <h2>What moves the needle</h2>
              <p>Top signals from held-out permutation importance</p>
            </div>
            <span className="subtle-tag">Top features</span>
          </div>
          <FeatureBars compact />
          <p className="chart-footnote">
            Higher values indicate a larger contribution to model ranking performance.
          </p>
        </section>

        <section className="panel snapshot-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">AT A GLANCE</span>
              <h2>Model snapshot</h2>
              <p>Performance on the held-out customer sample</p>
            </div>
            <div className="snapshot-icon"><Icon name="spark" /></div>
          </div>
          <div className="snapshot-score">
            <strong>{(evaluation.f1 * 100).toFixed(1)}%</strong>
            <span>F1 score</span>
          </div>
          <div className="snapshot-meter"><span style={{ width: `${evaluation.f1 * 100}%` }} /></div>
          <div className="snapshot-stats">
            <div><span>Default alert threshold</span><strong>{(evaluation.threshold * 100).toFixed(1)}%</strong></div>
            <div><span>Precision</span><strong>{(evaluation.precision * 100).toFixed(1)}%</strong></div>
            <div><span>Evaluation sample</span><strong>{evaluation.test_rows.toLocaleString()}</strong></div>
          </div>
          <button className="text-button" onClick={onPerformance}>
            Explore model performance <Icon name="arrow" size={16} />
          </button>
        </section>
      </div>
    </>
  );
}

function FeatureBars({ compact = false }) {
  const maxImportance = Math.max(...featureImportance.map((item) => item.mean));
  const items = compact ? featureImportance.slice(0, 5) : featureImportance;

  return (
    <div className={`feature-bars ${compact ? "feature-bars-compact" : ""}`}>
      {items.map((item, index) => (
        <div className="feature-row" key={item.feature}>
          <div className="feature-label">
            <span className="feature-rank">{String(index + 1).padStart(2, "0")}</span>
            <span>{formatLabel(item.feature)}</span>
          </div>
          <div className="feature-track">
            <span style={{ width: `${(item.mean / maxImportance) * 100}%` }} />
          </div>
          <span className="feature-value">{item.mean.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}

function FieldInput({ name, value, onChange }) {
  const info = metadata.features[name];
  const label = formatLabel(name);
  const options = info.options || [];

  if (info.kind === "categorical" || options.length > 0) {
    return (
      <label className="form-field">
        <span>{label}</span>
        <select
          value={value}
          onChange={(event) => onChange(name, event.target.value)}
        >
          {options.map((option) => (
            <option value={option} key={option}>
              {name === "SeniorCitizen" ? (option === 1 ? "Yes" : "No") : option}
            </option>
          ))}
        </select>
      </label>
    );
  }

  return (
    <label className="form-field">
      <span>{label}</span>
      <div className="input-with-unit">
        <input
          type="number"
          min={info.minimum}
          max={info.maximum}
          step={info.integer ? 1 : 0.01}
          value={value}
          onChange={(event) => onChange(name, event.target.value)}
        />
        {name === "tenure" && <span className="input-unit">months</span>}
        {(name === "MonthlyCharges" || name === "TotalCharges") && (
          <span className="input-unit">$</span>
        )}
      </div>
    </label>
  );
}

function Assessment() {
  const [customer, setCustomer] = useState(initialCustomer);
  const [threshold, setThreshold] = useState(evaluation.threshold);
  const [result, setResult] = useState(null);

  function updateCustomer(name, value) {
    setCustomer((current) => ({ ...current, [name]: value }));
  }

  function assessCustomer(event) {
    event.preventDefault();
    setResult(demoProbability(customer));
  }

  return (
    <>
      <PageHeader
        eyebrow="CUSTOMER ASSESSMENT"
        title="Build a customer profile"
        description="Enter the customer's current account and service details to preview the assessment flow."
        action={<span className="subtle-tag"><Icon name="clock" size={15} /> 19 profile fields</span>}
      />

      <div className="demo-notice">
        <span className="notice-icon"><Icon name="info" size={17} /></span>
        <div>
          <strong>Interactive UI demo</strong>
          <span>This preview uses an illustrative score, not the trained Python model.</span>
        </div>
      </div>

      <div className="assessment-layout">
        <form className="panel assessment-form" onSubmit={assessCustomer}>
          {fieldGroups.map((group) => (
            <section className="form-section" key={group.title}>
              <div className="form-section-heading">
                <div className="section-index">{String(fieldGroups.indexOf(group) + 1).padStart(2, "0")}</div>
                <div><h2>{group.title}</h2><p>{group.description}</p></div>
              </div>
              <div className="form-grid">
                {group.fields.map((field) => (
                  <FieldInput
                    key={field}
                    name={field}
                    value={customer[field]}
                    onChange={updateCustomer}
                  />
                ))}
              </div>
            </section>
          ))}
          <div className="assessment-footer">
            <span>Customer IDs are not used in the model.</span>
            <button type="submit" className="button button-primary">
              Preview risk <Icon name="arrow" size={17} />
            </button>
          </div>
        </form>

        <aside className="assessment-aside">
          <section className="panel threshold-panel">
            <span className="eyebrow">DECISION SETTINGS</span>
            <h2>Alert threshold</h2>
            <p>Choose the probability above which the demo flags a customer.</p>
            <div className="threshold-value">{Math.round(threshold * 100)}<span>%</span></div>
            <input
              className="threshold-slider"
              type="range"
              min="0.05"
              max="0.95"
              step="0.01"
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
              aria-label="Alert threshold"
              style={{ "--range-progress": `${((threshold - 0.05) / 0.9) * 100}%` }}
            />
            <div className="range-labels"><span>5%</span><span>95%</span></div>
            <div className="threshold-reference">
              <span>Model default</span><strong>{(evaluation.threshold * 100).toFixed(1)}%</strong>
            </div>
          </section>

          <section className={`panel result-panel ${result === null ? "result-empty" : ""}`}>
            <span className="eyebrow">ASSESSMENT PREVIEW</span>
            {result === null ? (
              <>
                <div className="empty-result-art"><Icon name="spark" size={24} /></div>
                <h2>Your result will appear here</h2>
                <p>Complete the customer profile and preview the illustrative risk score.</p>
              </>
            ) : (
              <>
                <div className={`result-status ${result >= threshold ? "high" : "low"}`}>
                  <span className="result-dot" />
                  {result >= threshold ? "Higher risk" : "Lower risk"}
                </div>
                <div className="result-score">{(result * 100).toFixed(0)}<span>%</span></div>
                <p>Illustrative churn probability</p>
                <div className="result-meter"><span style={{ width: `${result * 100}%` }} /></div>
                <div className="result-threshold">Alert threshold <strong>{Math.round(threshold * 100)}%</strong></div>
                <div className="result-tip">
                  <Icon name="info" size={16} />
                  <span>This demo score is heuristic only. Connect the Python model before using predictions.</span>
                </div>
              </>
            )}
          </section>
        </aside>
      </div>
    </>
  );
}

function Performance() {
  const matrix = evaluation.confusion_matrix;
  const rows = [
    { label: "No churn", values: matrix[0], total: matrix[0][0] + matrix[0][1] },
    { label: "Churn", values: matrix[1], total: matrix[1][0] + matrix[1][1] },
  ];

  return (
    <>
      <PageHeader
        eyebrow="MODEL PERFORMANCE"
        title="Understand the model"
        description="Review held-out results, classification details, and the strongest model signals."
        action={<span className="subtle-tag"><span className="status-dot" /> Held-out evaluation</span>}
      />

      <div className="performance-summary">
        <div className="performance-model">
          <div className="performance-model-icon"><Icon name="spark" size={22} /></div>
          <div><span>Selected model</span><strong>{evaluation.model}</strong></div>
        </div>
        <div className="performance-summary-stat"><span>Alert threshold</span><strong>{(evaluation.threshold * 100).toFixed(1)}%</strong></div>
        <div className="performance-summary-stat"><span>Test customers</span><strong>{evaluation.test_rows.toLocaleString()}</strong></div>
        <div className="performance-summary-stat"><span>ROC-AUC</span><strong>{evaluation.roc_auc.toFixed(3)}</strong></div>
      </div>

      <div className="content-grid performance-grid">
        <section className="panel roc-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">DISCRIMINATION</span><h2>ROC curve</h2><p>How well the model separates churn and non-churn cases</p></div>
            <span className="roc-score">AUC <strong>{evaluation.roc_auc.toFixed(3)}</strong></span>
          </div>
          <img className="roc-image" src={rocCurve} alt="ROC curve for the selected churn model" />
        </section>

        <section className="panel confusion-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">CLASSIFICATION</span><h2>Confusion matrix</h2><p>Predicted outcomes versus actual outcomes</p></div>
          </div>
          <div className="matrix-wrap">
            <div className="matrix-axis matrix-axis-top">PREDICTED</div>
            <div className="matrix-columns"><span>No churn</span><span>Churn</span></div>
            {rows.map((row) => (
              <div className="matrix-row" key={row.label}>
                <div className="matrix-row-label">{row.label}</div>
                {row.values.map((value, index) => (
                  <div
                    key={`${row.label}-${index}`}
                    className={`matrix-cell ${index === (row.label === "Churn" ? 1 : 0) ? "correct" : "incorrect"}`}
                  >
                    <strong>{value.toLocaleString()}</strong>
                    <span>{((value / row.total) * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            ))}
            <div className="matrix-axis-label">ACTUAL</div>
          </div>
          <div className="matrix-legend"><span><i className="legend-correct" /> Correct predictions</span><span><i className="legend-incorrect" /> Missed / false alarm</span></div>
        </section>
      </div>

      <section className="panel full-panel importance-full">
        <div className="panel-heading">
          <div><span className="eyebrow">FEATURE ANALYSIS</span><h2>Permutation importance</h2><p>ROC-AUC decrease when each feature is shuffled on the held-out set</p></div>
          <span className="subtle-tag">Higher = more influential</span>
        </div>
        <FeatureBars />
      </section>

      <p className="disclaimer">
        Metrics describe one held-out split and do not guarantee future performance. Feature importance is an association, not a causal explanation.
      </p>
    </>
  );
}

export default function App() {
  const [activePage, setActivePage] = useState("overview");
  const [activeNav, setActiveNav] = useState(false);

  function navigate(page) {
    setActivePage(page);
    setActiveNav(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  const pageContent = activePage === "assessment"
    ? <Assessment />
    : activePage === "performance"
      ? <Performance />
      : <Overview
          onAssess={() => navigate("assessment")}
          onPerformance={() => navigate("performance")}
        />;

  return (
    <div className="app-shell">
      <aside className={`sidebar ${activeNav ? "sidebar-open" : ""}`}>
        <a className="brand" href="#" onClick={(event) => { event.preventDefault(); navigate("overview"); }}>
          <span className="brand-mark"><Icon name="spark" size={21} /></span>
          <span className="brand-copy"><strong>churnwise</strong><small>INTELLIGENCE</small></span>
        </a>

        <div className="workspace-switcher">
          <div className="workspace-avatar">AC</div>
          <div><strong>Acme Connect</strong><span>Customer success</span></div>
          <Icon name="chevron" size={16} />
        </div>

        <div className="nav-section-label">WORKSPACE</div>
        <nav className="primary-nav" aria-label="Main navigation">
          {navigation.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${activePage === item.id ? "nav-item-active" : ""}`}
              onClick={() => navigate(item.id)}
            >
              <Icon name={item.icon} size={18} />
              <span>{item.label}</span>
              {activePage === item.id && <i />}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="model-status-card">
            <div className="status-card-top"><span className="status-dot" /> MODEL STATUS</div>
            <strong>Ready to assess</strong>
            <span>{evaluation.model} · threshold {(evaluation.threshold * 100).toFixed(0)}%</span>
          </div>
          <div className="user-profile">
            <div className="user-avatar">JD</div>
            <div><strong>Jordan Davis</strong><span>Analyst</span></div>
            <span className="profile-menu">···</span>
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setActiveNav((open) => !open)} aria-label="Toggle navigation">
            <span /><span /><span />
          </button>
          <div className="breadcrumb"><span>Workspace</span><Icon name="chevron" size={14} /><strong>{navigation.find((item) => item.id === activePage)?.label}</strong></div>
          <div className="topbar-right">
            <span className="environment-label"><i /> LOCAL DEMO</span>
            <div className="topbar-divider" />
            <button className="help-button" title="About this demo" aria-label="About this demo">
              <Icon name="info" size={17} />
            </button>
            <div className="topbar-avatar">JD</div>
          </div>
        </header>

        <div className="page-content">{pageContent}</div>
        <footer className="page-footer">
          <span>Customer Churn Intelligence</span>
          <span>Decision support only · Review with customer context</span>
        </footer>
      </main>
      {activeNav && <button className="sidebar-backdrop" onClick={() => setActiveNav(false)} aria-label="Close navigation" />}
    </div>
  );
}
